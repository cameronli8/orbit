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
from the build pipeline. Crucially, the {fact} in each template is built from
the per-suburb `breakdowns_json` blob (see categories.py / build.py), so we
only ever cite sub-categories that have a non-zero count in that suburb. This
is what kills the "beaches in Chippendale" class of bug.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

DIMENSIONS = ("social_energy", "aesthetic", "pace", "outdoor", "culinary", "community")
FEATURES_PATH = Path(__file__).parent.parent / "data" / "features" / "suburbs.parquet"

# Pretty labels for every breakdown sub-key. Anything not in here falls back to
# title-cased key. Kept here (not in categories.py) because these are display
# strings, not taxonomy — the LLM also reads them when building prompts.
BREAKDOWN_LABELS: Dict[str, str] = {
    # outdoor
    "beaches": "beaches", "parks": "parks", "trails": "trails",
    "waterfront": "waterfront", "sports_grounds": "sports grounds",
    "pools": "pools", "lookouts": "lookouts", "playgrounds": "playgrounds",
    # social
    "bars": "bars", "pubs": "pubs", "nightclubs": "nightclubs",
    "live_music": "live music venues", "breweries": "breweries",
    "karaoke": "karaoke bars",
    # aesthetic
    "galleries": "galleries", "museums": "museums",
    "vintage": "vintage stores", "indie_retail": "indie retail",
    "cinemas": "cinemas", "bookstores": "bookstores",
    "tattoo_piercing": "tattoo and piercing studios",
    # culinary
    "restaurants": "restaurants", "cafes": "cafés",
    "bakeries": "bakeries", "specialty_food": "specialty food shops",
    "markets": "markets and food trucks", "desserts": "dessert shops",
    # community
    "libraries": "libraries", "schools": "schools",
    "worship": "places of worship",
    "community_centers": "community centres",
    "civic": "civic buildings", "medical": "medical clinics",
}


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
def _preference_match(user_vec: np.ndarray, suburb_mat: np.ndarray) -> np.ndarray:
    """Preference-weighted similarity.

    For each suburb we compute, per dimension, how much it leans in the
    direction the user actually cares about — i.e.  (s - 50) * (u - 50).
    The total is normalised ONLY by the user's L1 deviation from neutral
    (50 * Σ|u - 50|), not by the suburb's norm. That asymmetry is the whole
    point: a user with a spiky profile (e.g. outdoor = 100, everything else
    near 50) gets recommendations that spike on outdoor, instead of a
    balanced cosine-friendly suburb that happens to be "broadly above
    average" across every dimension.

    In words:
        score = preference-weighted average of (s - 50)/50, weighted by
                |u - 50| — so dims the user cares about dominate, and
                dims the user is neutral on contribute zero.

    user_vec    shape (D,)  — raw 0-100 values
    suburb_mat  shape (N, D)
    returns     shape (N,)  in [0, 1] (raw is in [-1, 1], mapped for display).
    """
    u = user_vec.astype(float) - 50.0
    s = suburb_mat.astype(float) - 50.0

    # Σ|u - 50|: total strength of the user's expressed preference. If the
    # user picked every neutral answer this could be zero — guard with 1e-9
    # so we don't divide by zero.
    user_l1 = float(np.abs(u).sum()) or 1e-9

    # Raw score is in [-50*user_l1, +50*user_l1]; normalise to [-1, +1].
    raw = (s @ u) / (50.0 * user_l1)
    raw = np.clip(raw, -1.0, 1.0)

    # Map [-1, 1] → [0, 1] for a friendlier visual scale.
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

    fit  = _preference_match(user_vec, suburb_mat)
    gate = _budget_factor(df["rent_2br"], budget).to_numpy()
    match = fit * gate

    # Kept the column name `cosine` for backwards-compat with anything
    # downstream that still reads it (the LLM evidence blob, debug logs).
    df["cosine"]      = fit.round(4)
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


def _decode_breakdowns(row: pd.Series) -> Dict[str, Dict[str, int]]:
    """Parse the per-suburb breakdown JSON blob into a nested dict, once.
    Returns empty dict on missing/broken data (keeps backwards-compat with
    older parquets that don't have the column yet)."""
    blob = row.get("breakdowns_json") if hasattr(row, "get") else None
    if blob is None or (isinstance(blob, float) and np.isnan(blob)):
        return {}
    try:
        return json.loads(blob) if isinstance(blob, str) else dict(blob)
    except (TypeError, ValueError):
        return {}


def _decode_top_cuisines(row: pd.Series) -> List[Dict]:
    blob = row.get("top_cuisines_json") if hasattr(row, "get") else None
    if blob is None or (isinstance(blob, float) and np.isnan(blob)):
        return []
    try:
        return json.loads(blob) if isinstance(blob, str) else list(blob)
    except (TypeError, ValueError):
        return []


def _phrase_breakdowns(dim: str, row: pd.Series, max_items: int = 3) -> str:
    """Return something like '8 parks, 3 sports grounds and 2 playgrounds',
    built only from sub-categories with a non-zero count in this suburb.
    Returns '' if nothing non-zero is available."""
    breakdowns = _decode_breakdowns(row).get(dim, {})
    present = [(k, v) for k, v in breakdowns.items() if v and v > 0]
    if not present:
        return ""
    present.sort(key=lambda kv: -kv[1])
    parts = [f"{n} {BREAKDOWN_LABELS.get(k, k.replace('_', ' '))}"
             for k, n in present[:max_items]]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _fact_for(dim: str, row: pd.Series) -> str:
    """Pull a short, concrete phrase describing *why* this suburb scored
    the way it did on this dimension. Every phrase references a number we
    actually computed, so nothing is hallucinated.

    We ALWAYS prefer the breakdown-derived phrase when available — it tells
    the truth about what's actually in the suburb (so Chippendale reads as
    "parks and playgrounds" rather than the old lie about beaches). The old
    aggregate phrasing stays as a fallback for older parquets.
    """
    n = int(row["n_pois"])
    area = row["area_km2"]
    breakdown_phrase = _phrase_breakdowns(dim, row)

    if dim == "social_energy":
        if breakdown_phrase:
            return f"{breakdown_phrase} within {area:.1f} km²"
        per_km2 = row["social_density"]
        return f"{int(row['n_social'])} bars, pubs and venues across {area:.1f} km² (~{per_km2:.0f}/km²)"
    if dim == "aesthetic":
        indie_pct = 100 * row["indie_ratio"]
        if breakdown_phrase:
            return f"{breakdown_phrase}, {indie_pct:.0f}% independently owned"
        return f"{int(row['n_aesthetic'])} creative venues and {indie_pct:.0f}% independent businesses"
    if dim == "pace":
        share = 100 * row["late_night_share"]
        return f"{n} places within {area:.1f} km² with {share:.0f}% open late"
    if dim == "outdoor":
        if breakdown_phrase:
            return breakdown_phrase
        return f"{int(row['n_outdoor'])} outdoor spaces"
    if dim == "culinary":
        top_cuisines = _decode_top_cuisines(row)
        top_names = [c["cuisine"] for c in top_cuisines[:3] if c.get("count", 0) > 0]
        bits = []
        if breakdown_phrase:
            bits.append(breakdown_phrase)
        else:
            bits.append(f"{int(row['n_culinary'])} food businesses")
        if top_names:
            bits.append(f"strongest cuisines: {', '.join(top_names)}")
        else:
            bits.append(f"{int(row['n_cuisines'])} cuisines represented")
        return " — ".join(bits)
    if dim == "community":
        if breakdown_phrase:
            return breakdown_phrase
        return f"{int(row['n_community'])} schools, libraries and civic spaces"
    return "based on Foursquare category data"


# ---------------------------------------------------------------------------
# Structured evidence payload — the single source of facts the LLM is allowed
# to use when writing explanations. Anything not in here is off-limits.
# ---------------------------------------------------------------------------
def evidence_for(row: pd.Series, user_vec: Dict[str, float]) -> Dict:
    """Assemble a compact, strictly-factual evidence dict for a suburb.

    This is the sole input the LLM is permitted to draw facts from. Keeping
    it structured (a) lets us inject it into the prompt with zero ambiguity
    and (b) lets us audit the prompt — every claim in the generated text
    should trace back to a number or label in this blob.
    """
    breakdowns = _decode_breakdowns(row)
    top_cuisines = _decode_top_cuisines(row)

    dims = {d: float(row[d]) for d in DIMENSIONS}
    gaps = {d: round(float(row[d]) - user_vec[d], 1) for d in DIMENSIONS}

    # Strip zero entries so the model doesn't see "beaches: 0" and feel
    # tempted to mention beaches. If the key isn't in the payload, it can't
    # be cited.
    pruned_breakdowns = {
        dim: {k: v for k, v in cats.items() if v and v > 0}
        for dim, cats in breakdowns.items()
    }

    return {
        "suburb":        str(row["suburb"]),
        "rent_2br":      float(row["rent_2br"]),
        "area_km2":      float(row["area_km2"]),
        "n_pois":        int(row["n_pois"]),
        "dimensions":    {d: round(v, 1) for d, v in dims.items()},
        "user_vector":   {d: round(float(user_vec[d]), 1) for d in DIMENSIONS},
        "gaps":          gaps,  # suburb minus user: +ve = suburb stronger
        "breakdowns":    pruned_breakdowns,
        "top_cuisines":  [c for c in top_cuisines if c.get("count", 0) > 0][:6],
        "indie_ratio":   round(float(row["indie_ratio"]), 3),
        "late_night_share": round(float(row["late_night_share"]), 3),
        "n_cuisines":    int(row["n_cuisines"]),
    }


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
# Mock listings — 3 plausible listings per suburb, realestate.com-style
# ---------------------------------------------------------------------------
# Stock apartment photos (Unsplash CDN, stable URLs).  Each listing seeds
# into one of these so the image is consistent across reloads.
_LISTING_IMAGES = [
    "https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?w=900&h=600&fit=crop",  # scandi living room
    "https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?w=900&h=600&fit=crop",    # modern bedroom
    "https://images.unsplash.com/photo-1522708323590-d24dbb6b0267?w=900&h=600&fit=crop", # stylish apartment
    "https://images.unsplash.com/photo-1484154218962-a197022b5858?w=900&h=600&fit=crop", # kitchen
    "https://images.unsplash.com/photo-1536376072261-38c75010e6c9?w=900&h=600&fit=crop", # living area
    "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=900&h=600&fit=crop", # white bedroom
    "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=900&h=600&fit=crop", # open plan
    "https://images.unsplash.com/photo-1493809842364-78817add7ffb?w=900&h=600&fit=crop", # warm interior
    "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=900&h=600&fit=crop",   # tidy living
    "https://images.unsplash.com/photo-1560185893-a55cbc8c57e8?w=900&h=600&fit=crop",    # urban apt
    "https://images.unsplash.com/photo-1583847268964-b28dc8f51f92?w=900&h=600&fit=crop", # balcony
    "https://images.unsplash.com/photo-1598928506311-c55ded91a20c?w=900&h=600&fit=crop", # kitchen diner
    "https://images.unsplash.com/photo-1540518614846-7eded433c457?w=900&h=600&fit=crop", # bright living
    "https://images.unsplash.com/photo-1565183997392-2f6f122e5912?w=900&h=600&fit=crop", # dining
    "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=900&h=600&fit=crop", # loft
    "https://images.unsplash.com/photo-1555854877-bab0e564b8d5?w=900&h=600&fit=crop",    # balcony view
]

_PROPERTY_TYPES = ["Apartment", "Apartment", "Unit", "Unit",
                   "Townhouse", "Terrace", "Studio"]

_FEATURES = [
    "an open-plan kitchen and dining area",
    "polished timber floors throughout",
    "a private balcony with leafy outlook",
    "secure undercover parking",
    "split-system air-conditioning",
    "built-in wardrobes in every bedroom",
    "stone benchtops and stainless appliances",
    "a sun-drenched north-facing living area",
    "an internal laundry",
    "a dedicated study nook",
    "floor-to-ceiling windows",
    "a communal rooftop terrace",
    "a gas kitchen with dishwasher",
    "a generous master suite",
    "fresh paint and new carpet throughout",
]

_HEADLINE_TEMPLATES = [
    "Modern {beds}BR {ptype} in the heart of {suburb}",
    "Light-filled {beds}-bedroom {ptype_lc} close to everything",
    "Stylish {ptype_lc} with {feature_a}",
    "Renovated {beds}BR {ptype_lc} in prime {suburb} location",
    "Spacious {ptype_lc} with {feature_a}",
    "Designer {beds}BR {ptype_lc} — {feature_a}",
]

_DESCRIPTION_TEMPLATES = [
    "Tucked into a quiet pocket of {suburb}, this {beds}-bedroom {ptype_lc} offers {feature_a} and {feature_b}. A short walk from local cafés and transport — perfect for tenants who value {vibe}.",
    "Step inside this beautifully presented {ptype_lc} featuring {feature_a}. Moments from {suburb}'s best shops and eateries, with {feature_b} to round it out.",
    "Generous {beds}-bed {ptype_lc} in the heart of {suburb}. {feature_a}, {feature_b}, and a prime location steps from everyday essentials.",
    "A standout rental in {suburb} — this {ptype_lc} combines {feature_a} with {feature_b}. Ready to move in and make your own.",
    "Light-filled and well-proportioned, this {beds}BR {ptype_lc} sits in one of {suburb}'s most sought-after streets. {feature_a} and {feature_b} complete the offering.",
]

_VIBES = [
    "easy everyday living",
    "walkability and lifestyle",
    "quiet charm with city access",
    "space and light",
    "a connected neighbourhood feel",
]


def _realestate_url(suburb: str, beds: int, rent: int) -> str:
    """Build a realestate.com.au filtered rent-search URL.

    Pattern used: query-string filters on the suburb rent page. This is a
    real, working URL — clicking it lands the user on realestate.com.au's
    actual rentals for that suburb, filtered to the pin's beds and a ±$200
    price window so the results shown match what the pin claims.
    """
    slug = suburb.lower().replace(" ", "-")
    price_lo = max(50, rent - 200)
    price_hi = rent + 200
    return (
        f"https://www.realestate.com.au/rent/in-{slug},+nsw/list-1"
        f"?maxBeds={beds}&minBeds={beds}"
        f"&maxPrice={price_hi}&minPrice={price_lo}"
    )


def get_listings(
    suburb_name: str,
    rent_2br: float,
    n: int = 3,
    suburb_lat: float = None,
    suburb_lng: float = None,
) -> List[dict]:
    """Public entry point used by api.py.

    Tries the real Domain API (if DOMAIN_CLIENT_ID/SECRET are set in .env);
    falls back to `mock_listings()` if the real call returns nothing or
    fails. Real listings already carry lat/lng from Domain — mocks need
    the suburb centre to jitter around.

    Per-suburb listings are cached on disk for 24h by real_listings, so
    demo-day /match calls are essentially free after the first hit.
    """
    try:
        # Lazy import so a missing `requests` or import-time error in the
        # Domain module never takes the backend down.
        from real_listings import fetch_listings, DOMAIN_ENABLED
    except Exception:
        fetch_listings = None
        DOMAIN_ENABLED = False

    if DOMAIN_ENABLED and fetch_listings is not None:
        try:
            # Roughly target 2BR listings around the suburb's median rent —
            # Domain's `maxPrice` is strict, so give a generous ceiling.
            real = fetch_listings(
                suburb=suburb_name,
                beds=2,
                max_price=rent_2br * 1.5,
                n=n,
            )
            if real:
                # Backfill lat/lng if Domain omitted it by jittering around
                # the suburb centre — otherwise the pin would land at [0,0].
                if suburb_lat is not None and suburb_lng is not None:
                    import numpy as _np
                    rng = _np.random.default_rng(
                        int.from_bytes(suburb_name.encode()[:4].ljust(4, b"\0"), "big")
                    )
                    for l in real:
                        if l.get("lat") is None or l.get("lng") is None:
                            l["lat"] = round(suburb_lat + float(rng.uniform(-0.003, 0.003)), 6)
                            l["lng"] = round(suburb_lng + float(rng.uniform(-0.0035, 0.0035)), 6)
                return real
        except Exception as e:  # pragma: no cover — real_listings already fails-open
            import logging
            logging.getLogger("orbit.matcher").warning(
                "Real listings failed for %s (%s) — using mocks", suburb_name, e
            )

    return mock_listings(
        suburb_name=suburb_name,
        rent_2br=rent_2br,
        n=n,
        suburb_lat=suburb_lat,
        suburb_lng=suburb_lng,
    )


def mock_listings(
    suburb_name: str,
    rent_2br: float,
    n: int = 3,
    suburb_lat: float = None,
    suburb_lng: float = None,
) -> List[dict]:
    """Generate stable-but-plausible listings for a suburb.

    Each listing carries enough fields for a realestate.com-style map pin:
    lat/lng (jittered around the suburb centre), an image, beds/baths/
    parking, a mock address, a short headline, a 2-sentence listing-style
    description, and a working realestate.com.au deep link that lands on
    the real rent page for this suburb filtered by beds and price.

    Seeded by the suburb name so every request for the same suburb returns
    the same listings.
    """
    import hashlib
    seed = int(hashlib.md5(suburb_name.encode()).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(seed)

    listings = []
    for i in range(n):
        rent = int(rent_2br * rng.uniform(0.85, 1.18))
        beds = int(rng.choice([1, 2, 2, 2, 3], p=[0.15, 0.25, 0.25, 0.20, 0.15]))
        baths = int(min(beds, rng.choice([1, 1, 2])))
        parking = int(rng.choice([0, 1, 1, 2], p=[0.25, 0.40, 0.25, 0.10]))
        ptype = _PROPERTY_TYPES[int(rng.integers(0, len(_PROPERTY_TYPES)))]
        ptype_lc = ptype.lower()

        street = rng.choice(["Oxford", "King", "Victoria", "Park", "George",
                             "Church", "High", "Station", "Albert", "Queen",
                             "Macleay", "Elizabeth", "Darling", "Crown", "Bay"])
        number = int(rng.integers(1, 400))
        unit = int(rng.integers(1, 25)) if rng.random() < 0.7 else None
        addr = f"{number} {street} St" if unit is None else f"{unit}/{number} {street} St"

        # Pick two *different* features deterministically so descriptions
        # don't repeat "open-plan kitchen and an open-plan kitchen".
        f_indices = rng.choice(len(_FEATURES), size=2, replace=False)
        feature_a = _FEATURES[int(f_indices[0])]
        feature_b = _FEATURES[int(f_indices[1])]
        vibe      = _VIBES[int(rng.integers(0, len(_VIBES)))]

        headline = _HEADLINE_TEMPLATES[int(rng.integers(0, len(_HEADLINE_TEMPLATES)))].format(
            beds=beds, ptype=ptype, ptype_lc=ptype_lc,
            suburb=suburb_name, feature_a=feature_a,
        )
        description = _DESCRIPTION_TEMPLATES[int(rng.integers(0, len(_DESCRIPTION_TEMPLATES)))].format(
            beds=beds, ptype_lc=ptype_lc, suburb=suburb_name,
            feature_a=feature_a, feature_b=feature_b, vibe=vibe,
        )

        # Jitter the pin around the suburb centre so the 3 listings don't
        # stack on the same coordinate. Roughly ±300m — staying well inside
        # typical suburb bounds.
        if suburb_lat is not None and suburb_lng is not None:
            dlat = float(rng.uniform(-0.0030, 0.0030))
            dlng = float(rng.uniform(-0.0035, 0.0035))
            lat = round(suburb_lat + dlat, 6)
            lng = round(suburb_lng + dlng, 6)
        else:
            lat = lng = None

        image = _LISTING_IMAGES[int(rng.integers(0, len(_LISTING_IMAGES)))]

        listings.append({
            "id":          f"{suburb_name.lower().replace(' ', '-')}-{i+1}",
            "address":     f"{addr}, {suburb_name} NSW",
            "rent_pw":     rent,
            "beds":        beds,
            "baths":       baths,
            "parking":     parking,
            "property_type": ptype,
            "headline":    headline,
            "description": description,
            "image":       image,
            "lat":         lat,
            "lng":         lng,
            "url":         _realestate_url(suburb_name, beds, rent),
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
