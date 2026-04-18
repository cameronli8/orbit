"""
fetch_osm_pois.py — pulls real Sydney POIs from OpenStreetMap (Overpass API)
and writes them to data/raw/osm_pois_sydney.parquet.

Why OSM and not Foursquare?
    The mock dataset was built around Foursquare's category taxonomy because
    real FSQ data is gated behind HF auth. OSM is fully open, no API key,
    and has comprehensive coverage of Sydney for the categories Orbit cares
    about (cafes, bakeries, restaurants, bars, parks, beaches, gyms,
    galleries, cinemas, libraries).

Output schema (one row per POI):
    id           "node/12345" or "way/67890" — OSM type+id, globally unique
    name         human-readable name (POIs without a name are dropped)
    latitude     float, 6dp
    longitude    float, 6dp
    osm_tags     dict — the raw tags so we can re-classify later if needed
    group        one of: cafe, bakery, restaurant, bar, park, beach, gym,
                 gallery, cinema, library
    suburb       nearest of the 45 Orbit suburbs (Haversine to centroid)

Run:
    cd backend
    python3 fetch_osm_pois.py

The script is rerun-safe — overwrites the parquet each time.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from suburbs_ref import SYDNEY_SUBURBS


# ---------------------------------------------------------------------------
# Bounding box covering all 45 Orbit suburbs, with a small buffer so we don't
# clip POIs that sit just outside a suburb centroid but still belong to it.
# ---------------------------------------------------------------------------
_BUFFER_DEG = 0.02  # ~2 km
_LATS = [s["lat"] for s in SYDNEY_SUBURBS]
_LNGS = [s["lng"] for s in SYDNEY_SUBURBS]
BBOX = (
    min(_LATS) - _BUFFER_DEG,   # south
    min(_LNGS) - _BUFFER_DEG,   # west
    max(_LATS) + _BUFFER_DEG,   # north
    max(_LNGS) + _BUFFER_DEG,   # east
)


# ---------------------------------------------------------------------------
# OSM tag → display group mapping. Each group lists (key, value) pairs that
# qualify a feature for that group. Order matters: a feature gets the FIRST
# group that matches, so put the more-specific groups before the catch-alls.
# ---------------------------------------------------------------------------
_GROUP_TAGS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("bakery",     [("shop", "bakery"), ("shop", "pastry")]),
    ("cafe",       [("amenity", "cafe"), ("shop", "coffee")]),
    ("library",    [("amenity", "library")]),
    ("cinema",     [("amenity", "cinema")]),
    ("gallery",    [("tourism", "gallery"), ("tourism", "museum")]),
    ("gym",        [("leisure", "fitness_centre"), ("leisure", "sports_centre"),
                    ("sport", "fitness")]),
    ("beach",      [("natural", "beach"), ("leisure", "beach_resort")]),
    ("park",       [("leisure", "park"), ("leisure", "garden"),
                    ("leisure", "playground"), ("leisure", "dog_park")]),
    ("bar",        [("amenity", "bar"), ("amenity", "pub"),
                    ("amenity", "nightclub"), ("amenity", "biergarten")]),
    ("restaurant", [("amenity", "restaurant"), ("amenity", "fast_food"),
                    ("amenity", "food_court")]),
]


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OUTPUT_PATH  = Path(__file__).parent.parent / "data" / "raw" / "osm_pois_sydney.parquet"


def _build_query() -> str:
    """Compose one Overpass QL query that fetches every category in the bbox.

    `out center;` returns a centroid for ways/relations so we always get a
    single lat/lng per POI. We include nodes, ways, and relations because
    parks and museums are typically tagged on ways/relations.
    """
    south, west, north, east = BBOX
    bbox_str = f"({south},{west},{north},{east})"

    parts: List[str] = []
    for _group, tag_pairs in _GROUP_TAGS:
        for key, val in tag_pairs:
            parts.append(f'  node["{key}"="{val}"]{bbox_str};')
            parts.append(f'  way["{key}"="{val}"]{bbox_str};')
            parts.append(f'  relation["{key}"="{val}"]{bbox_str};')

    body = "\n".join(parts)
    return f"[out:json][timeout:180];\n(\n{body}\n);\nout center tags;"


def _classify(tags: Dict[str, str]) -> Optional[str]:
    """Return the first display group whose tag pair matches, else None."""
    for group, tag_pairs in _GROUP_TAGS:
        for key, val in tag_pairs:
            if tags.get(key) == val:
                return group
    return None


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in km between two points."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _nearest_suburb(lat: float, lng: float) -> str:
    """Map a POI to its nearest Orbit suburb centroid. Ties are extremely
    unlikely with float-precision distances; first-seen-wins is fine."""
    best_name = SYDNEY_SUBURBS[0]["name"]
    best_dist = float("inf")
    for s in SYDNEY_SUBURBS:
        d = _haversine_km(lat, lng, s["lat"], s["lng"])
        if d < best_dist:
            best_dist = d
            best_name = s["name"]
    return best_name


def fetch_overpass(retries: int = 3, backoff: float = 5.0) -> List[Dict]:
    """POST the Overpass query and return the raw `elements` list. Retries
    on 429/504 (Overpass is rate-limited and occasionally times out)."""
    query = _build_query()
    last_err: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[osm] querying Overpass (attempt {attempt}/{retries}) …",
                  file=sys.stderr)
            r = requests.post(OVERPASS_URL, data={"data": query}, timeout=240)
            if r.status_code == 200:
                data = r.json()
                elems = data.get("elements", [])
                print(f"[osm] got {len(elems)} elements", file=sys.stderr)
                return elems
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = str(e)
        # Backoff and retry.
        wait = backoff * attempt
        print(f"[osm] failed ({last_err}); waiting {wait}s before retry …",
              file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError(f"Overpass query failed after {retries} attempts: {last_err}")


def elements_to_pois(elements: List[Dict]) -> List[Dict]:
    """Convert raw Overpass elements into our POI schema. Drops anything
    without a name, without coordinates, or without a matching group."""
    pois: List[Dict] = []
    for el in elements:
        tags = el.get("tags") or {}
        name = (tags.get("name") or "").strip()
        if not name:
            continue
        # Coordinates: nodes have lat/lon; ways/relations have center.
        if "lat" in el and "lon" in el:
            lat, lng = el["lat"], el["lon"]
        elif "center" in el:
            lat = el["center"].get("lat")
            lng = el["center"].get("lon")
        else:
            continue
        if lat is None or lng is None:
            continue
        group = _classify(tags)
        if group is None:
            continue
        # Keep within the bbox proper (Overpass occasionally returns small
        # overlaps near edges — discard so suburb assignment stays accurate).
        south, west, north, east = BBOX
        if not (south <= lat <= north and west <= lng <= east):
            continue
        pois.append({
            "id":        f"{el['type']}/{el['id']}",
            "name":      name,
            "latitude":  round(float(lat), 6),
            "longitude": round(float(lng), 6),
            "osm_tags":  json.dumps(tags, separators=(",", ":")),
            "group":     group,
            "suburb":    _nearest_suburb(lat, lng),
        })
    return pois


def main():
    elements = fetch_overpass()
    pois = elements_to_pois(elements)

    if not pois:
        print("[osm] no POIs after filtering — refusing to write empty parquet",
              file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(pois)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    # Summary
    print(f"[osm] wrote {len(df)} POIs → {OUTPUT_PATH}")
    counts = df["group"].value_counts().sort_values(ascending=False)
    print("[osm] per-group counts:")
    for group, n in counts.items():
        print(f"    {group:11s} {n:5d}")
    n_suburbs = df["suburb"].nunique()
    print(f"[osm] POIs span {n_suburbs} of {len(SYDNEY_SUBURBS)} Orbit suburbs")


if __name__ == "__main__":
    main()
