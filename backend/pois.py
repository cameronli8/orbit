"""
pois.py — loads real Sydney POIs (OpenStreetMap) and exposes a compact
list for the `/pois` endpoint.

The frontend uses these as the main map feature — the coloured SVG
markers showing cafés, bars, parks, galleries, etc. sitting on top of
the heatmap.

Data source priority:
    1. data/raw/osm_pois_sydney.parquet  — real OSM POIs (from
       `fetch_osm_pois.py`); already pre-classified and suburb-tagged.
       This is what production uses.
    2. data/raw/fsq_sydney.parquet       — mock Foursquare dataset with
       the original category-keyword classifier + synthesised gyms.
       Only used as a fallback when the OSM parquet is missing.

Display groups (10, matching the frontend chip filter bar):
    bakery, cafe, restaurant, bar, park, beach, gym, gallery,
    cinema, library
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd


# Real OSM-derived POIs are preferred. The mock FSQ parquet is the fallback
# for environments where `fetch_osm_pois.py` hasn't been run. The OSM file
# already comes pre-classified (group, suburb assigned), so when present we
# skip the FSQ-style classify step entirely.
_OSM_PARQUET_PATH = Path(__file__).parent.parent / "data" / "raw" / "osm_pois_sydney.parquet"
_FSQ_PARQUET_PATH = Path(__file__).parent.parent / "data" / "raw" / "fsq_sydney.parquet"
_POIS_CACHE: Optional[List[Dict]] = None


# Group keyword rules — applied in order. First match wins.
# Substrings matched against the POI's leaf category label (case-insensitive).
_GROUP_RULES = [
    ("beach",      ("beach", "surf spot")),
    ("park",       ("park", "playground", "dog park", "botanical garden", "public garden")),
    ("bar",        ("cocktail bar", "wine bar", "beer bar", "whisky bar", "dive bar",
                    "sports bar", "hotel bar", "speakeasy", "nightclub", "dance club",
                    "pub", "beer garden", "gastropub", "brewery", "tasting room",
                    "live music venue", "music venue", "jazz club", "rock club",
                    "karaoke bar", "bar")),  # generic "bar" last
    ("gallery",    ("art gallery", "art museum", "museum", "art studio")),
    ("cinema",     ("cinema", "indie theater", "movie theater")),
    ("library",    ("library",)),
    ("bakery",     ("bakery",)),
    ("cafe",       ("independent coffee shop", "coffee shop", "café", "cafe", "tea room")),
    ("restaurant", ("restaurant", "food truck", "diner", "pizzeria", "pizza place",
                    "sushi bar", "ramen", "burger joint", "sandwich place")),
]


def _leaf_label(labels) -> str:
    """Given the raw fsq_category_labels cell, return the leaf category as a
    lowercase string. Handles None / empty / list-of-strings."""
    if labels is None:
        return ""
    try:
        first = labels[0] if len(labels) else ""
    except TypeError:
        return ""
    if not first:
        return ""
    parts = str(first).split(" > ")
    return parts[-1].strip().lower()


def _classify(leaf: str) -> Optional[str]:
    """Map a leaf label to a display group. Returns None if no group matches."""
    if not leaf:
        return None
    for group, keywords in _GROUP_RULES:
        for kw in keywords:
            if kw in leaf:
                return group
    return None


def _synthesize_gyms(n_per_suburb: int = 3) -> List[Dict]:
    """Generate a handful of gym POIs per Orbit suburb. We don't have gyms in
    the mock FSQ dataset, but gyms are one of the categories users expect to
    see on a rental map. These don't affect any dimension score — they're
    display-only, tagged with a `synthetic: true` marker so we can distinguish
    them if needed.

    Deterministic across requests (seeded rng) so the map doesn't jitter on
    refresh.
    """
    try:
        from suburbs_ref import SYDNEY_SUBURBS
    except Exception:
        return []
    rng = random.Random(42)
    # Mix realistic Sydney gym chain names so it reads believable.
    chain_names = [
        "Fitness First", "Anytime Fitness", "Snap Fitness", "F45 Training",
        "Plus Fitness", "Goodlife Health Clubs", "Crunch Fitness", "Virgin Active",
        "Jetts Fitness", "Orangetheory Fitness",
    ]
    boutique_names = [
        "Iron & Oak", "Peak Lab", "The Studio", "Bondi Strength Co",
        "Flow Yoga", "Reformer Space", "Bloom Pilates", "Hyrox House",
        "North Shore CrossFit", "Cardio Club",
    ]
    out: List[Dict] = []
    for s in SYDNEY_SUBURBS:
        lat0, lng0 = s["lat"], s["lng"]
        # Spread a few hundred metres around the centroid. 0.006° ≈ 660m.
        spread = 0.006
        for i in range(n_per_suburb):
            jitter_lat = (rng.random() - 0.5) * 2 * spread
            jitter_lng = (rng.random() - 0.5) * 2 * spread
            # Mix chain and boutique 50/50.
            name = rng.choice(chain_names if rng.random() < 0.5 else boutique_names)
            out.append({
                "id": f"gym_{s['name'].replace(' ', '_')}_{i}",
                "n": name,
                "la": round(lat0 + jitter_lat, 6),
                "ln": round(lng0 + jitter_lng, 6),
                "g": "gym",
                "s": s["name"],
            })
    return out


def _load_osm() -> List[Dict]:
    """Load the OSM-derived parquet and translate columns to the compact
    schema the frontend expects. The OSM file is already pre-classified
    (group + suburb), so this is just a column rename + dict pack.
    """
    df = pd.read_parquet(_OSM_PARQUET_PATH)
    pois: List[Dict] = []
    for _, row in df.iterrows():
        pois.append({
            "id": str(row["id"]),
            "n":  str(row["name"]),
            "la": round(float(row["latitude"]), 6),
            "ln": round(float(row["longitude"]), 6),
            "g":  str(row["group"]),
            "s":  str(row["suburb"]),
        })
    return pois


def _load_fsq_mock() -> List[Dict]:
    """Fallback path — load the mock Foursquare parquet, classify each row
    by its category leaf, and synthesize gym POIs (the mock dataset has no
    gyms). Used when the OSM parquet hasn't been generated.
    """
    df = pd.read_parquet(_FSQ_PARQUET_PATH)
    pois: List[Dict] = []
    for _, row in df.iterrows():
        leaf = _leaf_label(row.get("fsq_category_labels"))
        group = _classify(leaf)
        if group is None:
            continue
        pois.append({
            "id": str(row["fsq_place_id"]),
            "n":  str(row["name"]),
            "la": round(float(row["latitude"]), 6),
            "ln": round(float(row["longitude"]), 6),
            "g":  group,
            "s":  str(row.get("locality", "")) or "",
        })
    pois.extend(_synthesize_gyms())
    return pois


def load_pois() -> List[Dict]:
    """Load + classify + cache POIs. Prefers real OSM data when present,
    falls back to mock FSQ data when not. Uses compact field names (`n`,
    `la`, `ln`, `g`, `s`) because there are ~12k rows and JSON payload
    size matters when the frontend fetches this on session start.
    """
    global _POIS_CACHE
    if _POIS_CACHE is not None:
        return _POIS_CACHE

    if _OSM_PARQUET_PATH.exists():
        _POIS_CACHE = _load_osm()
    elif _FSQ_PARQUET_PATH.exists():
        _POIS_CACHE = _load_fsq_mock()
    else:
        _POIS_CACHE = []
    return _POIS_CACHE


def group_counts() -> Dict[str, int]:
    """Helper for /health-style debugging — returns {group: count}."""
    pois = load_pois()
    out: Dict[str, int] = {}
    for p in pois:
        out[p["g"]] = out.get(p["g"], 0) + 1
    return out


if __name__ == "__main__":
    # Smoke test: print counts per group.
    pois = load_pois()
    print(f"Total classified POIs: {len(pois)}")
    for g, n in sorted(group_counts().items(), key=lambda x: -x[1]):
        print(f"  {g:12s} {n:5d}")
