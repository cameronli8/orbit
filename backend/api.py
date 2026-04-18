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
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from quiz import QUESTIONS, score_user, quiz_payload, DIMENSIONS
from matcher import (
    SUBURBS,
    score_suburbs,
    build_explanations,
    get_listings,
    evidence_for,
)

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
def post_match(req: MatchRequest):
    """Score every suburb against the user's taste vector and budget, return
    the ranked list with explanations.

    Two entry points share this endpoint:
      • Answers path (legacy 7-question quiz): answers → score_user → vector.
      • Vector path (slider quiz):             user_vector used directly.
    Scoring logic (score_suburbs) is identical in both cases; only the
    vector origin differs.
    """
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

    return MatchResponse(
        user_vector=DimensionScores(**user_vec),
        suburbs=suburbs,
    )


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
    prof = profile_user(user_vec, answers)
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
    ai:            bool            = Query(True,  description="Set false to skip LLM calls."),
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
    if ai:
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
# Optional root index so hitting localhost:8000 in a browser isn't a 404.
# ---------------------------------------------------------------------------
@app.get("/")
def index():
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
