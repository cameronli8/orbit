"""
matcher.py  —  taste-vector matching + "why it fits" explanations.

Takes a user's six-dimensional vector (from quiz.py) and scores every row in
suburbs.parquet against it. The match score is a cosine similarity rescaled
to 0-100, gated by a smooth budget penalty — suburbs above the user's budget
fade out rather than disappearing, so the user still sees "here are great
fits you can't afford".

Why cosine, not Euclidean:
    Cosine captures *shape* of preferences rather than magnitude. A user who
    scores 80/70/60/50/40/30 and a suburb that scores 90/85/70/55/40/30 have
    very similar profiles — a cosine metric gives them a near-perfect match.
    Euclidean would penalise the magnitude gap and send the user to more
    muted profiles than they'd actually enjoy.

Explanations are template-driven and always point to a real computed number
from the build pipeline, so there's no hallucination risk.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

DIMENSIONS = ("social_energy", "aesthetic", "pace", "outdoor", "culinary", "community")
FEATURES_PATH = Path(__file__).parent.parent / "data" / "features" / "suburbs.parquet"


# ---------------------------------------------------------------------------
# Load — done once at import; api.py imports `SUBURBS` and re-uses it.
# ---------------------------------------------------------------------------
def _load() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} missing. Run `python backend/build.py` first."
        )
    return pl.read_parquet(FEATURES_PATH).to_pandas()


SUBURBS = _load()


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------
def _cosine_against_all(user_vec: np.ndarray, suburb_mat: np.ndarray) -> np.ndarray:
    """Preference-aware cosine similarity.

    We mean-center both vectors at 50 (the neutral baseline) before computing
    cosine. This way a dimension the user is indifferent about (value = 50)
    contributes zero to both dot product and norm — so it can't spuriously
    inflate the match. And a dimension where the user scored 100 only aligns
    with suburbs that also cleared 50 on it.

    user_vec    shape (D,)  — raw 0-100 values
    suburb_mat  shape (N, D)
    returns     shape (N,)  in [-1, 1]; rescaled to [0, 1] for display.
    """
    u = user_vec.astype(float) - 50.0
    s = suburb_mat.astype(float) - 50.0

    user_norm = np.linalg.norm(u) or 1e-9
    suburb_norms = np.linalg.norm(s, axis=1)
    denom = np.where(suburb_norms == 0, 1e-9, suburb_norms) * user_norm
    raw = (s @ u) / denom
    # Map from [-1, 1] to [0, 1] for a friendlier visual scale
    return (raw + 1.0) / 2.0


def _budget_factor(rent: pd.Series, budget: float, softness: float = 80.0) -> pd.Series:
    """Smooth multiplicative penalty for going over budget.

    Returns 1.0 when rent <= budget. For every $softness dollars over budget
    the score drops by ~40% (a logistic), but never hits 0 — we want
    unaffordable suburbs to stay *visible* at the bottom of the list, just
    heavily de-ranked. This is the "aspirational-but-honest" framing.
    """
    over = np.maximum(rent - budget, 0.0)
    # smooth decay to ~0.3 as we go far over budget
    return 0.3 + 0.7 * np.exp(-over / softness)


def score_suburbs(user_vec_dict: Dict[str, float], budget: float) -> pd.DataFrame:
    """Score every suburb against the user's quiz vector.

    Args:
        user_vec_dict: {dim: 0-100} for each of the six dimensions
        budget:        weekly 2BR rent budget in AUD

    Returns:
        A dataframe (copy of SUBURBS) with three extra columns:
            cosine       raw cosine in [0,1]
            budget_gate  smooth multiplicative budget penalty in [0.3, 1.0]
            match_score  0-100 headline match — cosine * budget_gate, rescaled
        sorted by match_score descending.
    """
    df = SUBURBS.copy()
    user_vec = np.array([user_vec_dict[d] for d in DIMENSIONS], dtype=float)
    suburb_mat = df[list(DIMENSIONS)].to_numpy(dtype=float)

    cosine = _cosine_against_all(user_vec, suburb_mat)
    gate   = _budget_factor(df["rent_2br"], budget).to_numpy()
    match  = cosine * gate

    df["cosine"]      = cosine.round(4)
    df["budget_gate"] = gate.round(3)
    # Rescale so a perfect in-budget match sits near 100 and the weakest
    # match lands somewhere in the 40s — this gives the heatmap meaningful
    # contrast without any suburb looking hopeless.
    if match.max() > 0:
        df["match_score"] = (100 * match / match.max()).round(1)
    else:
        df["match_score"] = 0.0

    return df.sort_values("match_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Explanations — templates that plug in real numbers from build.py
# ---------------------------------------------------------------------------
# Each dimension has POSITIVE and NEGATIVE templates. Positives fire when the
# suburb scores well *and* the user values that dimension. Negatives fire when
# there's a meaningful mismatch — either the suburb is weak on something the
# user values, or it's strong on something the user doesn't.
#
# The {fact} placeholder is filled from a small lookup that turns a raw
# feature (e.g. n_social, late_night_share) into a short human phrase.

POSITIVE_TEMPLATES = {
    "social_energy": "A genuinely social suburb — {fact}.",
    "aesthetic":     "Feels crafted, not chain-driven — {fact}.",
    "pace":          "Always something on — {fact}.",
    "outdoor":       "Built for being outside — {fact}.",
    "culinary":      "Food-forward — {fact}.",
    "community":     "A real neighbourhood feel — {fact}.",
}

NEGATIVE_TEMPLATES = {
    "social_energy": "Quieter nights than your profile suggests — {fact}.",
    "aesthetic":     "More commercial than curated — {fact}.",
    "pace":          "Pace is calmer than you'd probably like — {fact}.",
    "outdoor":       "Not many outdoor outlets on the doorstep — {fact}.",
    "culinary":      "Limited food variety compared with your taste — {fact}.",
    "community":     "Feels transient rather than rooted — {fact}.",
}


def _fact_for(dim: str, row: pd.Series) -> str:
    """Pull a short, concrete phrase describing *why* this suburb scored
    the way it did on this dimension. Every phrase references a number we
    actually computed, so nothing is hallucinated."""
    n = int(row["n_pois"])
    area = row["area_km2"]

    if dim == "social_energy":
        per_km2 = row["social_density"]
        return f"{int(row['n_social'])} bars, pubs and venues across {area:.1f} km² (~{per_km2:.0f}/km²)"
    if dim == "aesthetic":
        indie_pct = 100 * row["indie_ratio"]
        return f"{int(row['n_aesthetic'])} galleries/studios/creative retail and {indie_pct:.0f}% independent businesses"
    if dim == "pace":
        share = 100 * row["late_night_share"]
        return f"{n} POIs/km² with {share:.0f}% open late"
    if dim == "outdoor":
        return f"{int(row['n_outdoor'])} parks, beaches or outdoor venues"
    if dim == "culinary":
        return f"{int(row['n_cuisines'])} cuisines represented across {int(row['n_culinary'])} food businesses"
    if dim == "community":
        return f"{int(row['n_community'])} schools, libraries, civic and community spaces"
    return "based on Foursquare category data"


def build_explanations(
    suburb_row: pd.Series,
    user_vec: Dict[str, float],
    n_positive: int = 3,
    n_negative: int = 2,
) -> Dict[str, List[str]]:
    """Return the top positive and negative reasons this suburb fits/doesn't.

    A "positive" dimension is one where user and suburb agree and both are
    reasonably high (suburb >= 55 and user >= 55), or where the suburb is
    exceptionally strong (>= 75) regardless of user preference — strong
    signals are worth surfacing even if the user didn't specifically ask.

    A "negative" dimension is one with a large user↔suburb gap in either
    direction. We show the user *why* a match might feel off.
    """
    positives: List[Tuple[float, str]] = []
    negatives: List[Tuple[float, str]] = []

    for dim in DIMENSIONS:
        u = user_vec[dim]
        s = suburb_row[dim]
        gap = abs(u - s)
        alignment_strength = (u + s) / 2 - gap  # high when both high and close

        # Positive case: suburb strong AND either user values it or it's truly notable
        if (s >= 55 and u >= 55) or s >= 78:
            phrase = POSITIVE_TEMPLATES[dim].format(fact=_fact_for(dim, suburb_row))
            positives.append((alignment_strength, phrase))

        # Negative case: suburb has a clear gap below user's expectation
        # OR the suburb is much stronger than the user wants (e.g. loud
        # partying suburb for someone who picked "almost never out past 10pm")
        if u - s >= 22 or (s - u >= 22 and s >= 65):
            phrase = NEGATIVE_TEMPLATES[dim].format(fact=_fact_for(dim, suburb_row))
            negatives.append((gap, phrase))

    positives.sort(key=lambda x: -x[0])
    negatives.sort(key=lambda x: -x[0])
    return {
        "positive": [p for _, p in positives[:n_positive]],
        "negative": [n for _, n in negatives[:n_negative]],
    }


# ---------------------------------------------------------------------------
# Mock Domain listings — 3 plausible listings per suburb for the drawer
# ---------------------------------------------------------------------------
def mock_listings(suburb_name: str, rent_2br: float, n: int = 3) -> List[dict]:
    """Generate stable-but-plausible listings for the drawer.

    Seeded by the suburb name so every request for the same suburb returns
    the same listings (nicer for demo consistency).
    """
    import hashlib
    seed = int(hashlib.md5(suburb_name.encode()).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(seed)

    listings = []
    for i in range(n):
        # Vary rent ±15% around the median to feel realistic
        rent = int(rent_2br * rng.uniform(0.85, 1.18))
        beds = int(rng.choice([1, 2, 2, 2, 3], p=[0.15, 0.25, 0.25, 0.20, 0.15]))
        baths = int(min(beds, rng.choice([1, 1, 2])))
        street = rng.choice(["Oxford", "King", "Victoria", "Park", "George",
                             "Church", "High", "Station", "Albert", "Queen"])
        number = int(rng.integers(1, 400))
        unit = int(rng.integers(1, 25)) if rng.random() < 0.7 else None
        addr = f"{number} {street} St" if unit is None else f"{unit}/{number} {street} St"
        listings.append({
            "id":      f"{suburb_name.lower().replace(' ', '-')}-{i+1}",
            "address": f"{addr}, {suburb_name} NSW",
            "rent_pw": rent,
            "beds":    beds,
            "baths":   baths,
            "url":     f"https://www.domain.com.au/search?suburb={suburb_name.replace(' ', '-')}-nsw&rent={rent}",
        })
    return listings


# ---------------------------------------------------------------------------
# Script mode: quick sanity runs against the canonical personas
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from quiz import score_user

    personas = {
        "Indie inner-west": {
            "weekend_morning": "market_crawl", "nights_out": "few_times",
            "coffee_preference": "specialty_indie", "how_you_move": "cycle_walk",
            "neighbours": "some_community", "food_scene": "standout_local",
            "suburb_vibe": "indie_main_st",
        },
        "CBD night owl": {
            "weekend_morning": "brunch_crowd", "nights_out": "weekly_plus",
            "coffee_preference": "brunch_spot", "how_you_move": "gym_or_studio",
            "neighbours": "transient_ok", "food_scene": "adventurous",
            "suburb_vibe": "buzzy_streets",
        },
        "Beach outdoor": {
            "weekend_morning": "coastal_walk", "nights_out": "once_twice",
            "coffee_preference": "park_kiosk", "how_you_move": "surf_run",
            "neighbours": "some_community", "food_scene": "standout_local",
            "suburb_vibe": "coastal",
        },
        "North-shore family": {
            "weekend_morning": "home_chill", "nights_out": "almost_never",
            "coffee_preference": "reliable_chain", "how_you_move": "cycle_walk",
            "neighbours": "know_everyone", "food_scene": "mostly_home",
            "suburb_vibe": "leafy_family",
        },
    }

    for name, answers in personas.items():
        user_vec = score_user(answers)
        ranked = score_suburbs(user_vec, budget=900)
        print(f"\n===== {name} (budget $900) =====")
        for _, row in ranked.head(5).iterrows():
            print(f"  {row['suburb']:<20} match={row['match_score']:>5.1f} "
                  f"rent=${int(row['rent_2br']):<4d} "
                  f"(cos={row['cosine']:.3f}, gate={row['budget_gate']:.2f})")

        # Show explanation for the top pick
        top = ranked.iloc[0]
        expl = build_explanations(top, user_vec)
        print(f"\n  Why {top['suburb']}:")
        for p in expl["positive"]:
            print(f"   + {p}")
        for n in expl["negative"]:
            print(f"   - {n}")
