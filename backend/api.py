"""
api.py  —  Orbit's FastAPI server.

Run with:
    cd backend
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /health           liveness probe
    GET  /quiz             returns the seven quiz questions (no weight leakage)
    POST /match            body: {answers: {q_id: a_id}, budget: 800}
                           returns a ranked list of suburbs with match_score,
                           rent, dimension scores, and 3 positive +
                           2 negative explanation strings each
    GET  /suburb/{name}    full drawer payload for a single suburb:
                           profile, explanations relative to a submitted user
                           vector (or the neutral baseline), and 3 mock
                           Domain listings

Kept deliberately boring: no auth, no rate limiting, no analytics, no db.
`matcher.py` loads suburbs.parquet once at import time, so every request
hits an in-memory pandas dataframe.
"""

import json
import os
import re
import time
from collections import OrderedDict
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from quiz import QUESTIONS, score_user, quiz_payload, DIMENSIONS
from matcher import (
    SUBURBS,
    score_suburbs,
    build_explanations,
    get_listings,
    evidence_for,
)
import metrics

# How many listings each suburb serves to the map/drawer. 3 keeps the map
# from getting cluttered while still giving the drawer enough variety.
LISTINGS_PER_SUBURB = 3
from llm import profile_user, explain_suburb, llm_status
from pois import load_pois, group_counts


app = FastAPI(
    title="Orbit API",
    description="Match Sydney renters to suburbs that fit who they are.",
    version="0.1.0",
)

# Permissive CORS for the demo so the static HTML can hit this from anywhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Telemetry — initialised once per worker. See backend/metrics.py.
# Wrapped so a broken sqlite file (disk full, volume permissions, etc.)
# can't prevent the API from booting — worst case the admin page shows zeros.
# ---------------------------------------------------------------------------
try:
    metrics.init()
except Exception:
    pass


# Endpoints we actually want on the admin dashboard. Static asset hits are
# noisy and uninteresting for the live-room use case.
_TRACKED_PREFIXES = ("/match", "/profile", "/suburb/", "/quiz", "/pois", "/api", "/health", "/polygons")


def _canonical_endpoint(path: str) -> Optional[str]:
    """Map a concrete request path to the route template we want to count
    against. Returns None for things we don't track (static assets, admin
    page itself, the SPA root)."""
    if path.startswith("/suburb/"):
        return "/suburb/{name}"
    for prefix in _TRACKED_PREFIXES:
        if prefix.endswith("/"):
            if path.startswith(prefix):
                return prefix.rstrip("/")
        elif path == prefix:
            return prefix
    return None


@app.middleware("http")
async def _telemetry_middleware(request: Request, call_next):
    """Time the request, hash the client, record the outcome. Must never
    raise out of this function — telemetry failure is strictly lower priority
    than the actual response."""
    endpoint = _canonical_endpoint(request.url.path)
    if endpoint is None:
        return await call_next(request)

    t0 = time.perf_counter()
    status = 500
    error: Optional[str] = None
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            xff = request.headers.get("x-forwarded-for", request.client.host if request.client else "")
            ua = request.headers.get("user-agent", "")
            cache_hit = bool(getattr(request.state, "cache_hit", False))
            metrics.record(
                endpoint=endpoint,
                status=status,
                latency_ms=latency_ms,
                client=metrics.client_hash(xff, ua),
                cache_hit=cache_hit,
                error=error,
            )
        except Exception:
            # Telemetry failure must never surface to the caller.
            pass


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class DimensionScores(BaseModel):
    social_energy: float
    aesthetic:     float
    pace:          float
    outdoor:       float
    culinary:      float
    community:     float


class MatchRequest(BaseModel):
    answers: Optional[Dict[str, str]] = Field(
        None,
        description="Mapping of question_id -> chosen answer_id. Either this "
                    "or user_vector must be provided.",
        example={"weekend_morning": "market_crawl", "nights_out": "few_times"},
    )
    user_vector: Optional[DimensionScores] = Field(
        None,
        description="Direct six-dim taste vector (0-100 per dim). If present, "
                    "scoring uses this and bypasses score_user(). Enables the "
                    "slider-based quiz path without touching scoring logic.",
    )
    budget: float = Field(
        800.0,
        description="Weekly 2BR rent budget in AUD.",
        ge=0,
    )
    limit: int = Field(
        45,
        description="Max number of suburbs to return.",
        ge=1, le=200,
    )


class SuburbMatch(BaseModel):
    suburb:       str
    lat:          float
    lng:          float
    match_score:  float
    rent_2br:     float
    dimensions:   DimensionScores
    positive:     List[str]
    negative:     List[str]
    listings:     List[dict] = []


class MatchResponse(BaseModel):
    user_vector: DimensionScores
    suburbs:     List[SuburbMatch]


class ProfileRequest(BaseModel):
    answers: Optional[Dict[str, str]] = Field(
        None,
        description="Mapping of question_id -> chosen answer_id. Optional when "
                    "user_vector is supplied (slider-mode).",
    )
    user_vector: Optional[DimensionScores] = Field(
        None,
        description="Direct six-dim taste vector; skips score_user when set.",
    )
    ai: bool = Field(
        True,
        description="When true, call the LLM to generate the profile copy. "
                    "The effective behaviour is gated by the admin kill-switch "
                    "(metrics.get_setting('ai_enabled')), so the runtime state "
                    "is: request.ai AND ai_enabled. Flip the kill-switch from "
                    "the admin dashboard if the room is getting hammered.",
    )


class ProfileResponse(BaseModel):
    headline:    str
    summary:     str
    wants:       List[str]
    source:      str
    user_vector: DimensionScores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dim_dict(row: pd.Series) -> DimensionScores:
    return DimensionScores(**{d: float(row[d]) for d in DIMENSIONS})


# /match result cache — keyed on the full request shape so identical quiz
# answers (common under demo load: many users land on the same handful of
# answer combos) skip the scoring + explanation + listings pipeline entirely.
# Bounded OrderedDict used as a crude LRU — popitem(last=False) evicts oldest.
# The CPython GIL makes the dict ops effectively atomic; occasional double
# compute under a race is harmless and cheap.
_MATCH_CACHE: "OrderedDict[tuple, MatchResponse]" = OrderedDict()
_MATCH_CACHE_MAX = 256


def _ai_enabled() -> bool:
    """True iff the admin kill-switch is set to 'true'. Defaults to on so a
    missing setting behaves like the pre-toggle default. Cached in-process
    for _SETTINGS_TTL seconds inside metrics.get_setting."""
    return metrics.get_setting("ai_enabled", "true").lower() == "true"


def _match_cache_key(req: MatchRequest) -> Optional[tuple]:
    """Build a hashable cache key. Returns None when the request is malformed
    (no answers and no vector) — the handler raises 400 downstream."""
    budget = round(float(req.budget), 2)
    if req.answers:
        return ("answers", tuple(sorted(req.answers.items())), budget, req.limit)
    if req.user_vector is not None:
        vec = req.user_vector.model_dump()
        return ("vector", tuple(sorted(vec.items())), budget, req.limit)
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    try:
        from real_listings import DOMAIN_ENABLED
    except Exception:
        DOMAIN_ENABLED = False
    return {
        "status": "ok",
        "n_suburbs": int(len(SUBURBS)),
        "n_questions": len(QUESTIONS),
        "llm": llm_status(),
        "listings_source": "domain" if DOMAIN_ENABLED else "mock",
    }


@app.get("/quiz")
def get_quiz():
    """Return the quiz without exposing answer-to-weight mappings."""
    return {"questions": quiz_payload()}


@app.post("/match", response_model=MatchResponse)
def post_match(req: MatchRequest, request: Request):
    """Score every suburb against the user's taste vector and budget, return
    the ranked list with explanations.

    Two entry points share this endpoint:
      • Answers path (legacy 7-question quiz): answers → score_user → vector.
      • Vector path (slider quiz):             user_vector used directly.
    Scoring logic (score_suburbs) is identical in both cases; only the
    vector origin differs.

    Responses are cached in-process by (answers|vector, budget, limit) — many
    users during a live demo converge on a small set of answer combos, so the
    first hit pays the full cost and the rest are near-free.
    """
    cache_key = _match_cache_key(req)
    if cache_key is not None:
        cached = _MATCH_CACHE.get(cache_key)
        if cached is not None:
            _MATCH_CACHE.move_to_end(cache_key)
            # Flag for the telemetry middleware so the admin dashboard's
            # cache-hit rate reflects what actually happened.
            request.state.cache_hit = True
            return cached

    if req.user_vector is not None:
        user_vec = req.user_vector.model_dump()
    elif req.answers:
        user_vec = score_user(req.answers)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either `answers` or `user_vector`.",
        )

    ranked = score_suburbs(user_vec, budget=req.budget).head(req.limit)

    suburbs: List[SuburbMatch] = []
    for _, row in ranked.iterrows():
        expl = build_explanations(row, user_vec)
        listings = get_listings(
            str(row["suburb"]),
            float(row["rent_2br"]),
            n=LISTINGS_PER_SUBURB,
            suburb_lat=float(row["lat"]),
            suburb_lng=float(row["lng"]),
        )
        suburbs.append(
            SuburbMatch(
                suburb=str(row["suburb"]),
                lat=float(row["lat"]),
                lng=float(row["lng"]),
                match_score=float(row["match_score"]),
                rent_2br=float(row["rent_2br"]),
                dimensions=_dim_dict(row),
                positive=expl["positive"],
                negative=expl["negative"],
                listings=listings,
            )
        )

    response = MatchResponse(
        user_vector=DimensionScores(**user_vec),
        suburbs=suburbs,
    )

    if cache_key is not None:
        _MATCH_CACHE[cache_key] = response
        if len(_MATCH_CACHE) > _MATCH_CACHE_MAX:
            _MATCH_CACHE.popitem(last=False)
    return response


@app.post("/profile", response_model=ProfileResponse)
def post_profile(req: ProfileRequest):
    """Score the quiz and return an LLM-written personality profile.

    Accepts either the legacy answers payload or a direct user_vector (from the
    slider quiz). When only user_vector is supplied, the LLM sees an empty
    answers block — the vector itself is still rich enough to drive the
    profile copy.
    """
    if req.user_vector is not None:
        user_vec = req.user_vector.model_dump()
        answers = req.answers or {}
    elif req.answers:
        user_vec = score_user(req.answers)
        answers = req.answers
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either `answers` or `user_vector`.",
        )
    # Effective LLM usage = per-request opt-in AND global kill-switch on.
    # The admin dashboard toggles the kill-switch; during a room-of-300 spike
    # the operator can flip it off without a redeploy.
    prof = profile_user(user_vec, answers, use_llm=req.ai and _ai_enabled())
    return ProfileResponse(
        headline=prof.get("headline", "Your Orbit profile"),
        summary=prof.get("summary", ""),
        wants=prof.get("wants", []),
        source=prof.get("source", "fallback"),
        user_vector=DimensionScores(**user_vec),
    )


@app.get("/suburb/{name}")
def get_suburb(
    name: str,
    social_energy: Optional[float] = Query(None, ge=0, le=100),
    aesthetic:     Optional[float] = Query(None, ge=0, le=100),
    pace:          Optional[float] = Query(None, ge=0, le=100),
    outdoor:       Optional[float] = Query(None, ge=0, le=100),
    culinary:      Optional[float] = Query(None, ge=0, le=100),
    community:     Optional[float] = Query(None, ge=0, le=100),
    persona:       Optional[str]   = Query(None, description="User persona summary from /profile — grounds LLM copy."),
    ai:            bool            = Query(True,  description="Set true to enrich the drawer with LLM-written copy. Effective state is gated by the admin kill-switch (ai_enabled setting); template explanations always fire regardless."),
):
    """Full drawer payload for a single suburb.

    Query params let the frontend pass the user's vector so explanations can
    be personalised. If any dimension is omitted, the neutral baseline
    (50) is used for that dimension — the explanation then reads like
    'this suburb's standout features' rather than 'why this fits you'.

    If `ai=true` (default) and the LLM is configured, we also return an
    AI-written headline, summary and positive/negative bullets grounded in
    the same evidence blob that drives the template explanations.
    """
    match = SUBURBS[SUBURBS["suburb"].str.lower() == name.lower()]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown suburb: {name}")
    row = match.iloc[0]

    user_vec = {
        "social_energy": social_energy if social_energy is not None else 50,
        "aesthetic":     aesthetic     if aesthetic     is not None else 50,
        "pace":          pace          if pace          is not None else 50,
        "outdoor":       outdoor       if outdoor       is not None else 50,
        "culinary":      culinary      if culinary      is not None else 50,
        "community":     community     if community     is not None else 50,
    }

    expl = build_explanations(row, user_vec, n_positive=4, n_negative=3)
    listings = get_listings(
        str(row["suburb"]),
        float(row["rent_2br"]),
        n=LISTINGS_PER_SUBURB,
        suburb_lat=float(row["lat"]),
        suburb_lng=float(row["lng"]),
    )
    evidence = evidence_for(row, user_vec)

    ai_payload = None
    # Gate on both the request flag AND the admin kill-switch — a per-request
    # `ai=false` still disables AI even when the global switch is on.
    if ai and _ai_enabled():
        ai_payload = explain_suburb(
            evidence,
            profile_summary=(persona or ""),
            template_positive=expl["positive"],
            template_negative=expl["negative"],
        )

    return {
        "suburb":       str(row["suburb"]),
        "lat":          float(row["lat"]),
        "lng":          float(row["lng"]),
        "area_km2":     float(row["area_km2"]),
        "rent_2br":     float(row["rent_2br"]),
        "n_pois":       int(row["n_pois"]),
        "dimensions":   {d: float(row[d]) for d in DIMENSIONS},
        "positive":     expl["positive"],
        "negative":     expl["negative"],
        "listings":     listings,
        # Evidence the LLM was allowed to draw from — exposed so the
        # frontend can show the underlying facts ("12 cafés, 3 parks...")
        # and so judges can audit every claim.
        "evidence":     evidence,
        "ai":           ai_payload,
        # Raw counts so the drawer can show deeper stats if it wants
        "raw": {
            "n_social":     int(row["n_social"]),
            "n_aesthetic":  int(row["n_aesthetic"]),
            "n_outdoor":    int(row["n_outdoor"]),
            "n_culinary":   int(row["n_culinary"]),
            "n_community":  int(row["n_community"]),
            "n_cuisines":   int(row["n_cuisines"]),
            "n_indie":      int(row["n_indie"]),
            "n_chains":     int(row["n_chains"]),
            "indie_ratio":  float(row["indie_ratio"]),
        },
    }


# ---------------------------------------------------------------------------
# Suburb polygon GeoJSON — served from data/suburbs.geojson. Generated once
# by `python3 backend/fetch_suburb_polygons.py`. The frontend hits this
# instead of fetching the file directly, so opening index.html via file:// or
# any other origin still works (CORS is wildcard-open).
# ---------------------------------------------------------------------------
_GEOJSON_PATH = Path(__file__).parent.parent / "data" / "suburbs.geojson"
_GEOJSON_CACHE: Optional[dict] = None


def _load_geojson() -> Optional[dict]:
    global _GEOJSON_CACHE
    if _GEOJSON_CACHE is not None:
        return _GEOJSON_CACHE
    if not _GEOJSON_PATH.exists():
        return None
    try:
        _GEOJSON_CACHE = json.loads(_GEOJSON_PATH.read_text())
    except Exception:
        return None
    return _GEOJSON_CACHE


# ---------------------------------------------------------------------------
# POIs — the "main feature" map layer. Returns a compact list of ~7k POIs
# classified into display groups (cafe, bar, restaurant, park, gym, etc.).
# The frontend fetches this once at session start and renders it as a
# category-filterable overlay on top of the heatmap.
#
# Optional ?groups=cafe,bar filter lets callers request a subset.
# Response shape: {"groups": {group: count}, "pois": [{id, n, la, ln, g, s}]}.
# ---------------------------------------------------------------------------
@app.get("/pois")
def get_pois(groups: Optional[str] = Query(None, description="Comma-separated group filter")):
    pois = load_pois()
    if groups:
        wanted = {g.strip().lower() for g in groups.split(",") if g.strip()}
        pois = [p for p in pois if p["g"] in wanted]
    return {
        "groups": group_counts(),
        "pois": pois,
    }


@app.get("/polygons")
def get_polygons():
    gj = _load_geojson()
    if gj is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "data/suburbs.geojson not found. Run "
                "`python3 backend/fetch_suburb_polygons.py` once to generate it."
            ),
        )
    return JSONResponse(content=gj)


# ---------------------------------------------------------------------------
# API metadata (was at "/" — moved to "/api" so the root can serve the SPA).
# ---------------------------------------------------------------------------
@app.get("/api")
def api_info():
    return {
        "name": "Orbit API",
        "endpoints": [
            "/health", "/quiz", "/match (POST)", "/profile (POST)",
            "/suburb/{name}", "/polygons", "/pois",
        ],
        "suburbs_loaded": int(len(SUBURBS)),
        "polygons_loaded": _load_geojson() is not None,
        "llm": llm_status(),
    }


# ---------------------------------------------------------------------------
# Admin dashboard — `/admin-{token}`.
#
# Gate is `ADMIN_TOKEN` env var. Unset → the route returns 404, effectively
# disabling the admin page. Any mismatched token also returns 404 (rather
# than 403) so the path's existence isn't advertised via a different error.
# ---------------------------------------------------------------------------
import secrets as _secrets  # local import to keep the main import block short


def _admin_token_valid(candidate: str) -> bool:
    expected = os.environ.get("ADMIN_TOKEN", "")
    if not expected:
        return False
    # Constant-time compare so we don't leak token length via timing — cheap
    # defence but free to enable.
    return _secrets.compare_digest(expected, candidate)


def _fmt_duration(seconds: Optional[float]) -> str:
    if not seconds or seconds <= 0:
        return "—"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _render_daily_chart(series: List[Dict]) -> str:
    """Per-day dual-lane SVG chart for the long-term activity panel.

    Same visual grammar as _render_activity_chart so the admin page feels
    coherent, but bucketed by day instead of minute. Shared x-axis labels
    show the first day, middle day, and most recent day as YYYY-MM-DD.
    """
    if not series or not any(p["req"] for p in series):
        return (
            "<div class='muted' style='padding: 32px 0; text-align: center; "
            "font-size: 13px'>no activity in the last 30 days yet — counts "
            "will fill in as the volume collects history</div>"
        )

    n = len(series)
    req_max = max(1, max(p["req"] for p in series))
    usr_max = max(1, max(p["users"] for p in series))

    W, H_LANE, GAP, LABEL_H = 900.0, 70.0, 8.0, 24.0
    total_h = H_LANE * 2 + GAP + LABEL_H
    bar_slot = W / n
    bar_w = max(1.0, bar_slot - 1.0)

    def _bars(key: str, y_top: float, max_val: int, fill: str) -> str:
        chunks: List[str] = []
        for i, p in enumerate(series):
            v = p[key]
            if v <= 0:
                continue
            h = (v / max_val) * (H_LANE - 2)
            if h < 1.5:
                h = 1.5
            x = i * bar_slot + 0.5
            y = y_top + (H_LANE - h)
            d_label = time.strftime("%Y-%m-%d", time.localtime(p["t"]))
            chunks.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" '
                f'height="{h:.2f}" fill="{fill}" rx="1">'
                f'<title>{d_label} — {p["req"]} req, {p["users"]} users</title>'
                f'</rect>'
            )
        return "".join(chunks)

    req_bars = _bars("req", 0.0, req_max, "#6e7681")
    usr_bars = _bars("users", H_LANE + GAP, usr_max, "#4ade80")

    d_start = time.strftime("%b %d", time.localtime(series[0]["t"]))
    d_mid   = time.strftime("%b %d", time.localtime(series[n // 2]["t"]))
    d_end   = time.strftime("%b %d", time.localtime(series[-1]["t"]))

    label_y   = H_LANE * 2 + GAP + 16.0
    lane2_top = H_LANE + GAP
    axis_y1   = H_LANE
    axis_y2   = H_LANE * 2 + GAP
    mono = "ui-monospace,Menlo,Consolas,monospace"

    return (
        f'<svg viewBox="0 0 {W:.0f} {total_h:.0f}" width="100%" '
        f'preserveAspectRatio="none" '
        f'style="display:block;max-height:220px" role="img" '
        f'aria-label="Activity over the last 30 days">'
        f'<text x="6" y="12" fill="#6e7681" font-size="10" '
        f'font-family="{mono}">req/day (peak {req_max})</text>'
        f'{req_bars}'
        f'<line x1="0" y1="{axis_y1:.2f}" x2="{W:.0f}" y2="{axis_y1:.2f}" '
        f'stroke="#1f242c" stroke-width="0.5" />'
        f'<text x="6" y="{lane2_top + 12:.2f}" fill="#6e7681" font-size="10" '
        f'font-family="{mono}">unique users/day (peak {usr_max})</text>'
        f'{usr_bars}'
        f'<line x1="0" y1="{axis_y2:.2f}" x2="{W:.0f}" y2="{axis_y2:.2f}" '
        f'stroke="#1f242c" stroke-width="0.5" />'
        f'<text x="6" y="{label_y:.2f}" fill="#6e7681" font-size="10" '
        f'font-family="{mono}">{d_start}</text>'
        f'<text x="{W/2:.0f}" y="{label_y:.2f}" fill="#6e7681" font-size="10" '
        f'font-family="{mono}" text-anchor="middle">{d_mid}</text>'
        f'<text x="{W - 6:.0f}" y="{label_y:.2f}" fill="#6e7681" font-size="10" '
        f'font-family="{mono}" text-anchor="end">{d_end}</text>'
        f'</svg>'
    )


def _render_activity_chart(series: List[Dict]) -> str:
    """Build an inline SVG dual-lane bar chart for the admin's activity panel.

    Top lane (grey):  requests per minute.
    Bottom lane (green): unique users per minute.

    Each lane is scaled to its own max so the users lane stays legible even
    when it's dwarfed by raw request count (a single user can make 20 req
    in a minute). Shared x-axis labels: window-start, midpoint, now.

    Returns a self-contained HTML snippet — inline SVG, no JS, no external
    libs. One render pass, identical behaviour on every device.
    """
    if not series:
        return (
            "<div class='muted' style='padding: 32px 0; text-align: center; "
            "font-size: 13px'>no activity in the last 60 minutes — come back "
            "after someone uses the app</div>"
        )

    n = len(series)
    req_max = max(1, max(p["req"] for p in series))
    usr_max = max(1, max(p["users"] for p in series))

    # viewBox units — the SVG scales with its container via width=100%.
    W, H_LANE, GAP, LABEL_H = 900.0, 70.0, 8.0, 24.0
    total_h = H_LANE * 2 + GAP + LABEL_H
    bar_slot = W / n
    bar_w = max(1.0, bar_slot - 1.0)

    def _bars(key: str, y_top: float, max_val: int, fill: str) -> str:
        chunks: List[str] = []
        for i, p in enumerate(series):
            v = p[key]
            if v <= 0:
                continue
            h = (v / max_val) * (H_LANE - 2)
            if h < 1.5:  # ensure a visible sliver for any nonzero bucket
                h = 1.5
            x = i * bar_slot + 0.5
            y = y_top + (H_LANE - h)
            t_label = time.strftime("%H:%M", time.localtime(p["t"]))
            chunks.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" '
                f'height="{h:.2f}" fill="{fill}" rx="1">'
                f'<title>{t_label} — {p["req"]} req, {p["users"]} users</title>'
                f'</rect>'
            )
        return "".join(chunks)

    req_bars = _bars("req", 0.0, req_max, "#6e7681")
    usr_bars = _bars("users", H_LANE + GAP, usr_max, "#4ade80")

    # Time-axis labels — first bucket, middle, last bucket.
    t_start = time.strftime("%H:%M", time.localtime(series[0]["t"]))
    t_mid = time.strftime("%H:%M", time.localtime(series[n // 2]["t"]))
    t_end = time.strftime("%H:%M", time.localtime(series[-1]["t"]))

    label_y = H_LANE * 2 + GAP + 16.0
    lane2_top = H_LANE + GAP
    axis_y1 = H_LANE
    axis_y2 = H_LANE * 2 + GAP

    mono = "ui-monospace,Menlo,Consolas,monospace"

    return (
        f'<svg viewBox="0 0 {W:.0f} {total_h:.0f}" width="100%" '
        f'preserveAspectRatio="none" '
        f'style="display:block;max-height:220px" role="img" '
        f'aria-label="Activity over the last 60 minutes">'
        # lane 1 label + bars
        f'<text x="6" y="12" fill="#6e7681" font-size="10" '
        f'font-family="{mono}">req/min (peak {req_max})</text>'
        f'{req_bars}'
        # separator line between lanes
        f'<line x1="0" y1="{axis_y1:.2f}" x2="{W:.0f}" y2="{axis_y1:.2f}" '
        f'stroke="#1f242c" stroke-width="0.5" />'
        # lane 2 label + bars
        f'<text x="6" y="{lane2_top + 12:.2f}" fill="#6e7681" font-size="10" '
        f'font-family="{mono}">unique users/min (peak {usr_max})</text>'
        f'{usr_bars}'
        # x-axis baseline
        f'<line x1="0" y1="{axis_y2:.2f}" x2="{W:.0f}" y2="{axis_y2:.2f}" '
        f'stroke="#1f242c" stroke-width="0.5" />'
        # time axis labels
        f'<text x="6" y="{label_y:.2f}" fill="#6e7681" font-size="10" '
        f'font-family="{mono}">{t_start}</text>'
        f'<text x="{W/2:.0f}" y="{label_y:.2f}" fill="#6e7681" font-size="10" '
        f'font-family="{mono}" text-anchor="middle">{t_mid}</text>'
        f'<text x="{W - 6:.0f}" y="{label_y:.2f}" fill="#6e7681" font-size="10" '
        f'font-family="{mono}" text-anchor="end">{t_end}</text>'
        f'</svg>'
    )


def _render_admin(stats: Dict, token: str = "") -> str:
    """Server-render the admin dashboard as a single HTML doc. Manual refresh
    only (see the /match cache comment: we want the admin view to reflect
    exactly one point-in-time read, not a flickering auto-update).

    The token is needed for form actions (e.g. the AI kill-switch toggle).
    """
    endpoints_rows = "".join(
        f"<tr><td>{_html_escape(e['endpoint'])}</td><td class='num'>{e['count']:,}</td></tr>"
        for e in stats.get("endpoints", [])
    ) or "<tr><td colspan='2' class='muted'>no traffic in the last hour</td></tr>"

    errors_rows = "".join(
        f"<tr><td class='num muted'>{e['age_s']:.0f}s ago</td>"
        f"<td>{_html_escape(e['endpoint'])}</td>"
        f"<td class='num'>{e['status']}</td>"
        f"<td class='err'>{_html_escape(e['error'] or '')}</td></tr>"
        for e in stats.get("errors", [])
    ) or "<tr><td colspan='4' class='muted'>no errors — healthy</td></tr>"

    now_ts = stats.get("generated_at") or time.time()
    window_start = stats.get("window_start")
    window_age = (now_ts - window_start) if window_start else None

    llm = llm_status() or {}
    llm_configured = bool(llm.get("enabled"))
    llm_model = llm.get("model", "—")

    # Runtime kill-switch — independent of whether the LLM is configured.
    # Effective AI = llm_configured AND ai_switch_on. The toggle button lets
    # the demo operator flip ai_switch_on live without a redeploy.
    ai_switch_on = metrics.get_setting("ai_enabled", "true").lower() == "true"
    ai_effective = llm_configured and ai_switch_on
    if not llm_configured:
        ai_state_label = "fallback-only"
        ai_state_sub = f"{_html_escape(str(llm_model))} · no API key"
        ai_dot = "warn"
    elif not ai_switch_on:
        ai_state_label = "disabled"
        ai_state_sub = f"{_html_escape(str(llm_model))} · kill-switch on"
        ai_dot = "warn"
    else:
        ai_state_label = "enabled"
        ai_state_sub = f"{_html_escape(str(llm_model))} · live"
        ai_dot = "ok"

    toggle_btn = ""
    if token and llm_configured:
        toggle_label = "turn off" if ai_switch_on else "turn on"
        toggle_btn = (
            f"<form method='post' action='/admin-{_html_escape(token)}/toggle-ai' "
            f"style='margin:10px 0 0 0'>"
            f"<button class='btn' type='submit'>{toggle_label}</button>"
            f"</form>"
        )

    try:
        worker_pid = os.getpid()
    except Exception:
        worker_pid = 0

    cache_size = len(_MATCH_CACHE)

    init_err = stats.get("init_error")
    init_err_banner = (
        f"<div class='banner err'>metrics init error: {_html_escape(init_err)}</div>"
        if init_err else ""
    )

    activity_chart_svg = _render_activity_chart(stats.get("timeseries") or [])
    daily_chart_svg    = _render_daily_chart(stats.get("daily_timeseries") or [])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <meta name="robots" content="noindex,nofollow" />
  <title>Orbit · admin</title>
  <style>
    :root {{
      --bg: #0b0d10;
      --panel: #12161b;
      --panel-2: #181d24;
      --text: #e6ebf0;
      --muted: #6e7681;
      --accent: #4ade80;
      --warn: #f59e0b;
      --err: #ef4444;
      --border: #1f242c;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--text);
                  font-family: -apple-system, BlinkMacSystemFont, "Inter", system-ui, sans-serif; }}
    header {{ padding: 24px 28px; border-bottom: 1px solid var(--border); display: flex;
              align-items: center; justify-content: space-between; }}
    header h1 {{ margin: 0; font-size: 18px; letter-spacing: 0.02em; }}
    header .sub {{ color: var(--muted); font-size: 13px; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 24px 28px 48px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
             gap: 14px; margin-bottom: 24px; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
             padding: 16px 18px; }}
    .card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase;
                    letter-spacing: 0.06em; margin-bottom: 8px; }}
    .card .value {{ font-size: 28px; font-weight: 600; letter-spacing: -0.01em; }}
    .card .sub {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .section {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
                padding: 18px 20px; margin-bottom: 18px; }}
    .section h2 {{ margin: 0 0 14px; font-size: 14px; color: var(--muted);
                   text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 500; font-size: 12px; text-transform: uppercase;
          letter-spacing: 0.06em; }}
    tr:last-child td {{ border-bottom: none; }}
    td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .muted {{ color: var(--muted); }}
    .err {{ color: var(--err); font-family: ui-monospace, Menlo, monospace; font-size: 12px; }}
    .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
            background: var(--muted); margin-right: 8px; }}
    .dot.ok {{ background: var(--accent); box-shadow: 0 0 6px rgba(74,222,128,0.6); }}
    .dot.warn {{ background: var(--warn); }}
    .dot.err {{ background: var(--err); }}
    .actions {{ display: flex; gap: 10px; align-items: center; }}
    .btn {{ background: var(--panel-2); color: var(--text); border: 1px solid var(--border);
            padding: 8px 14px; border-radius: 6px; font-size: 13px; cursor: pointer;
            text-decoration: none; display: inline-block; }}
    .btn:hover {{ background: #222830; }}
    .banner {{ padding: 10px 14px; border-radius: 8px; margin-bottom: 16px; font-size: 13px; }}
    .banner.err {{ background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); color: var(--err); }}
    .row {{ display: flex; gap: 14px; flex-wrap: wrap; }}
    .row .card {{ flex: 1 1 220px; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Orbit · admin</h1>
      <div class="sub">live activity · generated {_html_escape(time.strftime("%H:%M:%S", time.localtime(now_ts)))}</div>
    </div>
    <div class="actions">
      <a class="btn" href="">refresh</a>
    </div>
  </header>

  <main>
    {init_err_banner}

    <div class="grid">
      <div class="card">
        <div class="label">total unique users</div>
        <div class="value">{stats['unique_users_all']:,}</div>
        <div class="sub">since {_fmt_duration(window_age)} ago</div>
      </div>
      <div class="card">
        <div class="label">total uses</div>
        <div class="value">{stats['total_requests']:,}</div>
        <div class="sub">all-time requests served</div>
      </div>
      <div class="card">
        <div class="label">unique users · last 5m</div>
        <div class="value">{stats['unique_users_5m']:,}</div>
        <div class="sub">{stats['unique_users_1h']:,} in last hour</div>
      </div>
      <div class="card">
        <div class="label">requests · last 5m</div>
        <div class="value">{stats['req_last_5m']:,}</div>
        <div class="sub">{stats['req_per_min_last_5']} req/min avg</div>
      </div>
      <div class="card">
        <div class="label">latency · last 5m</div>
        <div class="value">{stats['p50_ms']:.0f} <span style="font-size:16px;color:var(--muted)">ms</span></div>
        <div class="sub">p95: {stats['p95_ms']:.0f} ms</div>
      </div>
    </div>

    <div class="section">
      <h2>long-term activity</h2>
      <div class="row">
        <div class="card">
          <div class="label">unique users · 24h</div>
          <div class="value">{stats['unique_users_24h']:,}</div>
          <div class="sub">{stats['req_24h']:,} requests</div>
        </div>
        <div class="card">
          <div class="label">unique users · 7d</div>
          <div class="value">{stats['unique_users_7d']:,}</div>
          <div class="sub">{stats['req_7d']:,} requests</div>
        </div>
        <div class="card">
          <div class="label">unique users · 30d</div>
          <div class="value">{stats['unique_users_30d']:,}</div>
          <div class="sub">{stats['req_30d']:,} requests</div>
        </div>
        <div class="card">
          <div class="label">unique users · all-time</div>
          <div class="value">{stats['unique_users_all']:,}</div>
          <div class="sub">{stats['total_requests']:,} total uses</div>
        </div>
      </div>
    </div>

    <div class="section">
      <h2>system health</h2>
      <div class="row">
        <div class="card">
          <div class="label">api</div>
          <div class="value" style="font-size:16px"><span class="dot ok"></span>ok</div>
          <div class="sub">worker pid {worker_pid}</div>
        </div>
        <div class="card">
          <div class="label">ai explanations</div>
          <div class="value" style="font-size:16px">
            <span class="dot {ai_dot}"></span>
            {ai_state_label}
          </div>
          <div class="sub">{ai_state_sub}</div>
          {toggle_btn}
        </div>
        <div class="card">
          <div class="label">/match cache</div>
          <div class="value" style="font-size:16px">{stats['cache_hit_rate']}%</div>
          <div class="sub">{cache_size} keys cached · {stats['match_cache_hits_1h']}/{stats['match_requests_1h']} hit 1h</div>
        </div>
        <div class="card">
          <div class="label">suburbs loaded</div>
          <div class="value" style="font-size:16px">{len(SUBURBS)}</div>
          <div class="sub">in-memory dataframe</div>
        </div>
      </div>
    </div>

    <div class="section">
      <h2>activity · last 60 minutes</h2>
      {activity_chart_svg}
    </div>

    <div class="section">
      <h2>activity · last 30 days</h2>
      {daily_chart_svg}
    </div>

    <div class="section">
      <h2>endpoint traffic · last hour</h2>
      <table>
        <thead><tr><th>endpoint</th><th class="num">requests</th></tr></thead>
        <tbody>{endpoints_rows}</tbody>
      </table>
    </div>

    <div class="section">
      <h2>recent errors</h2>
      <table>
        <thead><tr><th>when</th><th>endpoint</th><th class="num">code</th><th>detail</th></tr></thead>
        <tbody>{errors_rows}</tbody>
      </table>
    </div>

    <p class="muted" style="font-size:12px;margin-top:28px">
      manual refresh only · worker-local cache + shared sqlite events ·
      hidden path, unindexed by design
    </p>
  </main>
</body>
</html>
"""


def _html_escape(s) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


@app.get("/admin-{token}", include_in_schema=False)
def admin_dashboard(token: str):
    """Secret-URL-gated admin page. Token comes from ADMIN_TOKEN env var.
    Returns 404 on missing/mismatched token so the endpoint's existence is
    not leaked through a 403."""
    if not _admin_token_valid(token):
        # 404 response body intentionally mimics StaticFiles so probers can't
        # distinguish this from any other missing path.
        return HTMLResponse(status_code=404, content="Not Found")
    stats = metrics.get_stats()
    return HTMLResponse(content=_render_admin(stats, token=token))


@app.post("/admin-{token}/toggle-ai", include_in_schema=False)
def admin_toggle_ai(token: str):
    """Flip the ai_enabled kill-switch. Form-POST target from the admin UI.
    303-redirects back to the dashboard so the browser reloads with the new
    state visible and a refresh doesn't re-submit the form."""
    if not _admin_token_valid(token):
        return HTMLResponse(status_code=404, content="Not Found")
    current = metrics.get_setting("ai_enabled", "true").lower() == "true"
    metrics.set_setting("ai_enabled", "false" if current else "true")
    return RedirectResponse(url=f"/admin-{token}", status_code=303)


# ---------------------------------------------------------------------------
# Static frontend mount.
#
# Serves `index.html`, the service worker, the manifest, icons, and any other
# assets that live in the project root (one level up from `backend/`). Mounted
# LAST so all real API routes win path-resolution before the catch-all kicks
# in. `html=True` makes "/" auto-resolve to index.html.
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).parent.parent


@app.get("/sw.js")
def service_worker():
    """Service workers must be served from the same origin and scope they
    control. Hand-rolled response so we can set the right content type and
    avoid any caching headers that would freeze users on a stale SW."""
    sw_path = FRONTEND_DIR / "sw.js"
    if not sw_path.exists():
        raise HTTPException(status_code=404, detail="sw.js not found")
    return FileResponse(
        sw_path,
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend",
)
