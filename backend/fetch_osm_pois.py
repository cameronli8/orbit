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

# Point-in-polygon assignment is dramatically more accurate than
# nearest-centroid: with the centroid approach, all POIs sitting between
# two distant suburbs collapse onto whichever centroid is geographically
# nearest, which over-assigns thousands of POIs to suburbs near the
# fringes of the rated set. Falls back to nearest-centroid only if
# shapely / suburbs.geojson is unavailable.
try:
    from shapely.geometry import shape, Point
    from shapely.strtree import STRtree
    _SHAPELY_OK = True
except ImportError:
    _SHAPELY_OK = False


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
    # --- Culinary (shown on map) ----------------------------------------
    ("bakery",     [("shop", "bakery"), ("shop", "pastry")]),
    ("cafe",       [("amenity", "cafe"), ("shop", "coffee")]),
    ("restaurant", [("amenity", "restaurant"), ("amenity", "fast_food"),
                    ("amenity", "food_court")]),
    # --- Aesthetic (shown on map) ---------------------------------------
    ("library",    [("amenity", "library")]),
    ("cinema",     [("amenity", "cinema")]),
    ("gallery",    [("tourism", "gallery"), ("tourism", "museum")]),
    # --- Outdoor (shown on map) -----------------------------------------
    ("gym",        [("leisure", "fitness_centre"), ("leisure", "sports_centre"),
                    ("sport", "fitness")]),
    ("beach",      [("natural", "beach"), ("leisure", "beach_resort")]),
    ("park",       [("leisure", "park"), ("leisure", "garden"),
                    ("leisure", "playground"), ("leisure", "dog_park"),
                    ("leisure", "nature_reserve")]),
    # --- Social (shown on map via "bar") --------------------------------
    ("bar",        [("amenity", "bar"), ("amenity", "pub"),
                    ("amenity", "nightclub"), ("amenity", "biergarten")]),
    # --- Social: breweries and live music (feed scores, map-invisible) --
    ("brewery",    [("craft", "brewery"), ("industrial", "brewery")]),
    ("music_venue",[("amenity", "arts_centre"), ("amenity", "theatre"),
                    ("amenity", "concert_hall")]),
    # --- Outdoor: pools, lookouts, sports grounds, marinas, waterfront --
    ("pool",       [("leisure", "swimming_pool"), ("sport", "swimming")]),
    ("lookout",    [("tourism", "viewpoint"), ("natural", "peak")]),
    ("sports",     [("leisure", "pitch"), ("leisure", "stadium"),
                    ("leisure", "track"), ("leisure", "golf_course"),
                    ("leisure", "sports_hall")]),
    ("marina",     [("leisure", "marina"), ("waterway", "dock"),
                    ("man_made", "pier")]),
    # --- Culinary: markets, specialty shops, ice cream ------------------
    ("market",     [("amenity", "marketplace")]),
    ("specialty_food", [("shop", "deli"), ("shop", "cheese"),
                        ("shop", "wine"), ("shop", "butcher"),
                        ("shop", "seafood"), ("shop", "greengrocer"),
                        ("shop", "chocolate"), ("shop", "confectionery")]),
    ("ice_cream",  [("amenity", "ice_cream"), ("shop", "ice_cream")]),
    # --- Aesthetic: bookstores, vintage, indie retail, tattoo -----------
    ("bookstore",  [("shop", "books")]),
    ("vintage",    [("shop", "second_hand"), ("shop", "antiques"),
                    ("shop", "charity")]),
    ("indie_retail",[("shop", "music"), ("shop", "art"),
                     ("shop", "craft"), ("shop", "musical_instrument")]),
    ("tattoo",     [("shop", "tattoo")]),
    # --- Community: schools, worship, civic, medical --------------------
    ("school",     [("amenity", "school"), ("amenity", "kindergarten"),
                    ("amenity", "college"), ("amenity", "university")]),
    ("worship",    [("amenity", "place_of_worship")]),
    ("community_centre", [("amenity", "community_centre"),
                          ("amenity", "social_centre")]),
    ("civic",      [("amenity", "townhall"), ("amenity", "courthouse"),
                    ("amenity", "post_office")]),
    ("medical",    [("amenity", "clinic"), ("amenity", "doctors"),
                    ("amenity", "dentist"), ("amenity", "pharmacy"),
                    ("amenity", "hospital")]),
]

# The subset of groups the frontend map renders as icons. New groups above
# feed the suburb scoring pipeline but are filtered out in pois.py so the
# map doesn't gain unknown-icon markers for schools/clinics/etc.
DISPLAY_GROUPS = {
    "bakery", "cafe", "restaurant", "bar", "park", "beach",
    "gym", "gallery", "cinema", "library",
}


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OUTPUT_PATH  = Path(__file__).parent.parent / "data" / "raw" / "osm_pois_sydney.parquet"
POLYGONS_PATH = Path(__file__).parent.parent / "data" / "suburbs.geojson"


# Lazily-loaded polygon index, keyed by rated-suburb name. Using an STRtree
# makes point-in-polygon ~O(log N) per query — essential for 40k+ POIs.
_POLY_INDEX: Optional[object] = None
_POLY_GEOMS: List[object] = []
_POLY_NAMES: List[str] = []


# Curated aliases — unrated neighbour polygons that should be merged into a
# rated Orbit suburb. Without these, suburbs whose colloquial area crosses
# official boundaries lose huge chunks of POIs:
#   • "Bondi" (the residential suburb) doesn't contain Bondi Beach —
#     the beach and most cafés are in the separate "BONDI BEACH" suburb.
#   • "Kings Cross" is an officially tiny 0.6 km² sliver; the real KX
#     nightlife strip lives across Potts Point + Woolloomooloo +
#     Elizabeth Bay + Rushcutters Bay.
#   • "Bronte" excludes Tamarama (a tiny adjacent beach).
#   • "Coogee" excludes South Coogee (with South Coogee Beach).
#   • "Manly" excludes the immediately-adjacent Fairlight / Queenscliff
#     heads.
# Values must be unrated in suburbs.geojson — if both sides claim a poly
# the assignment becomes first-match-wins (confusing at best).
_ALIAS_POLYGONS: Dict[str, List[str]] = {
    "Bondi":       ["BONDI BEACH", "NORTH BONDI"],
    "Bronte":      ["TAMARAMA"],
    "Coogee":      ["SOUTH COOGEE"],
    "Manly":       ["FAIRLIGHT", "QUEENSCLIFF"],
    "Kings Cross": ["RUSHCUTTERS BAY", "ELIZABETH BAY", "WOOLLOOMOOLOO"],
    "Parramatta":  ["HARRIS PARK"],
}


def _load_polygon_index() -> None:
    """One-time load of the rated-suburb polygons into a Shapely STRtree.

    Each rated suburb's polygon is unioned with any unrated neighbour
    polygons listed in `_ALIAS_POLYGONS` so colloquial areas (e.g.
    Bondi+Bondi Beach, Kings Cross+Woolloomooloo) are treated as one.
    Aliases must point to UNRATED polygons — otherwise two rated suburbs
    would race to claim the same POI.
    """
    global _POLY_INDEX, _POLY_GEOMS, _POLY_NAMES
    if _POLY_INDEX is not None or not _SHAPELY_OK:
        return
    if not POLYGONS_PATH.exists():
        return
    rated_names = {s["name"].upper() for s in SYDNEY_SUBURBS}
    canonical_by_upper = {s["name"].upper(): s["name"] for s in SYDNEY_SUBURBS}

    # Build a name → geom map across the whole file so we can look up
    # aliases by exact uppercase name. Skip rated polygons here so an
    # alias accidentally pointing to a rated polygon never wins.
    with POLYGONS_PATH.open() as f:
        fc = json.load(f)
    rated_geoms: Dict[str, object] = {}
    unrated_geoms: Dict[str, object] = {}
    for feat in fc.get("features", []):
        props = feat.get("properties") or {}
        name = str(props.get("name", "")).upper()
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        if name in rated_names:
            rated_geoms[name] = geom
        else:
            unrated_geoms[name] = geom

    geoms: List[object] = []
    names: List[str] = []
    n_aliased = 0
    for upper_name, base_geom in rated_geoms.items():
        canonical = canonical_by_upper[upper_name]
        merged = base_geom
        for alias in _ALIAS_POLYGONS.get(canonical, []):
            alias_upper = alias.upper()
            if alias_upper in rated_geoms:
                # Refuse to absorb — would steal POIs from a sibling rated
                # suburb. Loud warn so misconfig is caught early.
                print(f"[osm] WARN alias {alias!r} for {canonical!r} "
                      f"is itself rated — skipped", file=sys.stderr)
                continue
            ag = unrated_geoms.get(alias_upper)
            if ag is None:
                print(f"[osm] WARN alias {alias!r} for {canonical!r} "
                      f"not in geojson — skipped", file=sys.stderr)
                continue
            merged = merged.union(ag)
            n_aliased += 1
        if not merged.is_valid:
            merged = merged.buffer(0)
        geoms.append(merged)
        names.append(canonical)

    _POLY_GEOMS = geoms
    _POLY_NAMES = names
    _POLY_INDEX = STRtree(geoms)
    print(f"[osm] loaded {len(geoms)} rated-suburb polygons "
          f"({n_aliased} alias merges) from {POLYGONS_PATH.name}",
          file=sys.stderr)


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
    """Fallback nearest-centroid assignment. Used only when polygons aren't
    available — the polygon-based path in `_assign_suburb` is the primary."""
    best_name = SYDNEY_SUBURBS[0]["name"]
    best_dist = float("inf")
    for s in SYDNEY_SUBURBS:
        d = _haversine_km(lat, lng, s["lat"], s["lng"])
        if d < best_dist:
            best_dist = d
            best_name = s["name"]
    return best_name


# Coastline / waterfront POIs (beaches, piers, marinas) have OSM center
# coordinates that sit a few metres offshore of the residential polygon.
# Bondi Beach (way/173244595) center = (-33.8923747, 151.2780805) falls
# outside the BONDI BEACH polygon boundary. A small buffer on the distance
# check picks them up without re-inflating the mis-assignment problem.
# 0.002° ≈ 200m at Sydney latitudes.
_COASTAL_BUFFER_DEG = 0.002


def _assign_suburb(lat: float, lng: float) -> List[str]:
    """Return the names of every rated suburb whose polygon contains this
    point. Empty list means the point falls outside every rated polygon —
    drop the POI rather than mis-assign it to the geographically nearest
    centroid (which inflates suburbs near the fringes of the rated set,
    e.g. Cabramatta vacuuming up south-western Sydney).

    Returns multiple matches when rated polygons overlap. The notable case
    is Potts Point / Kings Cross — these have IDENTICAL polygons in the
    geojson because they're the same physical place under two colloquial
    names. Returning both lets each suburb's score reflect the same POI
    set, instead of first-match-wins giving one of them zero POIs.

    Falls back to a small (~200m) buffer when no polygon contains the
    point — this catches beach and waterfront POIs whose OSM centers sit
    just offshore of the residential polygon (e.g. Bondi Beach, piers,
    marinas). The buffer is small enough that an inland POI outside all
    polygons still gets dropped.
    """
    _load_polygon_index()
    if _POLY_INDEX is None:
        # Polygons unavailable — fall back to legacy behaviour so the script
        # still produces output. Pipeline degrades gracefully.
        return [_nearest_suburb(lat, lng)]
    pt = Point(lng, lat)  # GeoJSON is (x=lng, y=lat)
    candidates = _POLY_INDEX.query(pt)
    matches: List[str] = []
    for idx in candidates:
        if _POLY_GEOMS[idx].covers(pt):
            matches.append(_POLY_NAMES[idx])
    if matches:
        return matches
    # Fallback: try a small-buffered point for coastline/waterfront POIs
    # whose centers are a few metres offshore. Query the tree with the
    # buffered geometry so STRtree's bounding-box pre-filter catches near
    # polygons, then use distance as the tiebreak.
    buf = pt.buffer(_COASTAL_BUFFER_DEG)
    near = _POLY_INDEX.query(buf)
    best_name = None
    best_dist = float("inf")
    for idx in near:
        d = _POLY_GEOMS[idx].distance(pt)
        if d <= _COASTAL_BUFFER_DEG and d < best_dist:
            best_dist = d
            best_name = _POLY_NAMES[idx]
    return [best_name] if best_name else []


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
        suburbs = _assign_suburb(lat, lng)
        if not suburbs:
            # Falls outside every rated polygon — drop. This is the whole
            # point of the polygon switch: refuse to fabricate an assignment.
            continue
        # Emit one row per matching suburb. Virtually every POI has exactly
        # one match; Potts Point / Kings Cross share a polygon so POIs
        # there emit twice (once per suburb) — this is correct, they are
        # geographically the same place.
        base_id = f"{el['type']}/{el['id']}"
        osm_tags_json = json.dumps(tags, separators=(",", ":"))
        for suburb in suburbs:
            pois.append({
                # Suburb-suffix the id when a POI lands in multiple rated
                # polygons so downstream joins stay unique.
                "id":        base_id if len(suburbs) == 1 else f"{base_id}@{suburb}",
                "name":      name,
                "latitude":  round(float(lat), 6),
                "longitude": round(float(lng), 6),
                "osm_tags":  osm_tags_json,
                "group":     group,
                "suburb":    suburb,
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
