"""
api.py  —  Homing's two-endpoint FastAPI server.

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

from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from quiz import QUESTIONS, score_user, quiz_payload, DIMENSIONS
from matcher import (
    SUBURBS,
    score_suburbs,
    build_explanations,
    mock_listings,
)


app = FastAPI(
    title="Homing API",
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
class MatchRequest(BaseModel):
    answers: Dict[str, str] = Field(
        ...,
        description="Mapping of question_id -> chosen answer_id.",
        example={"weekend_morning": "market_crawl", "nights_out": "few_times"},
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


class DimensionScores(BaseModel):
    social_energy: float
    aesthetic:     float
    pace:          float
    outdoor:       float
    culinary:      float
    community:     float


class SuburbMatch(BaseModel):
    suburb:       str
    lat:          float
    lng:          float
    match_score:  float
    rent_2br:     float
    dimensions:   DimensionScores
    positive:     List[str]
    negative:     List[str]


class MatchResponse(BaseModel):
    user_vector: DimensionScores
    suburbs:     List[SuburbMatch]


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
    return {
        "status": "ok",
        "n_suburbs": int(len(SUBURBS)),
        "n_questions": len(QUESTIONS),
    }


@app.get("/quiz")
def get_quiz():
    """Return the quiz without exposing answer-to-weight mappings."""
    return {"questions": quiz_payload()}


@app.post("/match", response_model=MatchResponse)
def post_match(req: MatchRequest):
    """Score every suburb against the user's quiz answers and budget,
    return the ranked list with explanations."""
    if not req.answers:
        raise HTTPException(status_code=400, detail="No quiz answers submitted.")

    user_vec = score_user(req.answers)
    ranked = score_suburbs(user_vec, budget=req.budget).head(req.limit)

    suburbs: List[SuburbMatch] = []
    for _, row in ranked.iterrows():
        expl = build_explanations(row, user_vec)
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
            )
        )

    return MatchResponse(
        user_vector=DimensionScores(**user_vec),
        suburbs=suburbs,
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
):
    """Full drawer payload for a single suburb.

    Query params let the frontend pass the user's vector so explanations can
    be personalised. If any dimension is omitted, the neutral baseline
    (50) is used for that dimension — the explanation then reads like
    'this suburb's standout features' rather than 'why this fits you'.
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
    listings = mock_listings(str(row["suburb"]), float(row["rent_2br"]))

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
# Optional root index so hitting localhost:8000 in a browser isn't a 404.
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return {
        "name": "Homing API",
        "endpoints": ["/health", "/quiz", "/match (POST)", "/suburb/{name}"],
        "suburbs_loaded": int(len(SUBURBS)),
    }
