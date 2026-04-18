"""
osm_to_fsq.py — adapt OSM POIs to the Foursquare-style schema that
`build.py` expects.

Why this exists
---------------
Orbit was originally wired around a mock Foursquare parquet because real
FSQ data is gated behind HuggingFace auth. The keyword sets in
`categories.py` (and the breakdown sub-categories the drawer cites) are
all written as FSQ-shaped label substrings like "Outdoor Recreation > Beach"
or "Food > Restaurant > Vietnamese Restaurant".

`fetch_osm_pois.py` writes real OpenStreetMap data into
`data/raw/osm_pois_sydney.parquet` with a totally different schema
(`id, name, latitude, longitude, osm_tags, group, suburb`). Rather than
rewriting every keyword set in `categories.py` to know about OSM tags,
this script translates OSM rows into FSQ-labelled rows and writes them to
`data/raw/fsq_sydney.parquet`. `build.py` then consumes that file with
zero code changes.

Trade-off: we lose the "real FSQ schema everywhere" purity, but we gain
real data flowing through a classifier/scorer that's been sanity-checked
against the mock. For a hackathon this is the right call.

Run:
    cd backend
    python3 osm_to_fsq.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


OSM_PATH = Path(__file__).parent.parent / "data" / "raw" / "osm_pois_sydney.parquet"
FSQ_PATH = Path(__file__).parent.parent / "data" / "raw" / "fsq_sydney.parquet"


# ---------------------------------------------------------------------------
# Cuisine tag → FSQ cuisine label. Keep this aligned with CUISINE_KEYWORDS
# in categories.py — the substring matcher is what actually counts entropy,
# so the label just needs to contain the keyword.
# ---------------------------------------------------------------------------
_CUISINE_MAP: Dict[str, str] = {
    "italian": "Italian",
    "chinese": "Chinese",
    "japanese": "Japanese",
    "korean":   "Korean",
    "thai":     "Thai",
    "vietnamese": "Vietnamese",
    "indian":   "Indian",
    "mexican":  "Mexican",
    "french":   "French",
    "spanish":  "Spanish",
    "greek":    "Greek",
    "turkish":  "Turkish",
    "lebanese": "Lebanese",
    "middle_eastern": "Middle Eastern",
    "mediterranean":  "Mediterranean",
    "american": "American",
    "australian": "Australian",
    "modern_australian": "Modern Australian",
    "ethiopian": "Ethiopian",
    "moroccan":  "Moroccan",
    "african":   "African",
    "caribbean": "Caribbean",
    "brazilian": "Brazilian",
    "peruvian":  "Peruvian",
    "argentine": "Argentine",
    "sushi":     "Sushi",
    "ramen":     "Ramen",
    "dumpling":  "Dumpling",
    "dim_sum":   "Dim Sum",
    "bbq":       "BBQ",
    "barbecue":  "BBQ",
    "burger":    "Burger",
    "pizza":     "Pizza",
    "sandwich":  "Sandwich",
    "seafood":   "Seafood",
    "steak":     "Steakhouse",
    "steak_house": "Steakhouse",
    "vegetarian": "Vegetarian",
    "vegan":     "Vegan",
    "fusion":    "Fusion",
    "tapas":     "Tapas",
    "breakfast": "Breakfast",
    "brunch":    "Brunch",
    "bakery":    "Bakery",
    "malaysian": "Malaysian",
    "indonesian": "Indonesian",
    "filipino":  "Filipino",
    "singaporean": "Singaporean",
    "cantonese": "Cantonese",
    "sichuan":   "Sichuan",
    "taiwanese": "Taiwanese",
}


def _parse_tags(tags_field) -> Dict[str, str]:
    """OSM tags ship as a JSON blob in the parquet. Be defensive."""
    if tags_field is None:
        return {}
    if isinstance(tags_field, dict):
        return {str(k): str(v) for k, v in tags_field.items()}
    try:
        obj = json.loads(tags_field)
    except Exception:
        return {}
    return {str(k): str(v) for k, v in obj.items()} if isinstance(obj, dict) else {}


def _cuisines_from(tags: Dict[str, str]) -> List[str]:
    """OSM encodes cuisine as `cuisine=italian` or `cuisine=vietnamese;cafe`.
    Return the human-readable names of every recognised cuisine.
    """
    raw = tags.get("cuisine") or tags.get("diet:type") or ""
    if not raw:
        return []
    out: List[str] = []
    for token in raw.replace(",", ";").split(";"):
        key = token.strip().lower().replace(" ", "_").replace("-", "_")
        if key in _CUISINE_MAP:
            out.append(_CUISINE_MAP[key])
    return out


def _labels_for(group: str, tags: Dict[str, str]) -> List[str]:
    """Produce the FSQ-style category labels that should attach to a given
    OSM POI. Emits ONE label per POI (matching the mock FSQ schema) — the
    most specific one available. Substring matching in categories.py then
    handles parent-category rollups automatically (e.g. a "Vietnamese
    Restaurant" label matches both the "Restaurant" keyword and the
    "Vietnamese" cuisine keyword).
    """
    labels: List[str] = []

    # ---- Culinary ----------------------------------------------------
    if group == "bakery":
        if tags.get("shop") == "pastry":
            labels.append("Food > Bakery > Pastry Shop")
        else:
            labels.append("Food > Bakery")

    elif group == "cafe":
        if tags.get("shop") == "coffee":
            labels.append("Food > Coffee Shop")
        else:
            labels.append("Food > Café")

    elif group == "restaurant":
        # Prefer the cuisine-specific label ("Food > Restaurant > Italian
        # Restaurant") so the entropy counter picks up a cuisine AND the
        # parent "Restaurant" keyword still matches via substring. Fall
        # back to the generic label for untagged venues.
        cuisines = _cuisines_from(tags)
        if cuisines:
            labels.append(f"Food > Restaurant > {cuisines[0]} Restaurant")
        elif tags.get("amenity") == "fast_food":
            labels.append("Food > Fast Food Restaurant")
        elif tags.get("amenity") == "food_court":
            labels.append("Food > Food Court")
        else:
            labels.append("Food > Restaurant")

    elif group == "market":
        labels.append("Food > Farmers Market")

    elif group == "specialty_food":
        shop = tags.get("shop") or ""
        mapping = {
            "deli":          "Food > Delicatessen",
            "cheese":        "Food > Cheese Shop",
            "wine":          "Food > Wine Shop",
            "butcher":       "Food > Butcher",
            "seafood":       "Food > Seafood Market",
            "greengrocer":   "Food > Specialty Food Store",
            "chocolate":     "Food > Chocolatier",
            "confectionery": "Food > Specialty Food Store",
        }
        labels.append(mapping.get(shop, "Food > Specialty Food Store"))

    elif group == "ice_cream":
        labels.append("Food > Dessert Shop > Ice Cream Shop")

    # ---- Social ------------------------------------------------------
    elif group == "bar":
        amenity = tags.get("amenity") or ""
        if amenity == "pub":
            labels.append("Dining and Drinking > Pub")
        elif amenity == "nightclub":
            labels.append("Dining and Drinking > Nightclub")
        elif amenity == "biergarten":
            labels.append("Dining and Drinking > Beer Garden")
        else:
            labels.append("Dining and Drinking > Bar")

    elif group == "brewery":
        labels.append("Food > Brewery")

    elif group == "music_venue":
        labels.append("Arts and Entertainment > Music Venue")

    # ---- Outdoor -----------------------------------------------------
    elif group == "park":
        leisure = tags.get("leisure") or ""
        if leisure == "playground":
            labels.append("Outdoor Recreation > Playground")
        elif leisure == "garden":
            labels.append("Outdoor Recreation > Garden")
        elif leisure == "dog_park":
            labels.append("Outdoor Recreation > Dog Park")
        elif leisure == "nature_reserve":
            labels.append("Outdoor Recreation > Nature Reserve")
        else:
            labels.append("Outdoor Recreation > Park")

    elif group == "beach":
        labels.append("Outdoor Recreation > Beach")

    elif group == "pool":
        labels.append("Outdoor Recreation > Swimming Pool")

    elif group == "lookout":
        labels.append("Outdoor Recreation > Scenic Lookout")

    elif group == "sports":
        leisure = tags.get("leisure") or ""
        sport   = (tags.get("sport") or "").lower()
        if leisure == "golf_course" or "golf" in sport:
            labels.append("Outdoor Recreation > Golf Course")
        elif "tennis" in sport:
            labels.append("Outdoor Recreation > Tennis Court")
        elif "basketball" in sport:
            labels.append("Outdoor Recreation > Basketball Court")
        elif "soccer" in sport or "football" in sport:
            labels.append("Outdoor Recreation > Soccer Field")
        elif "skate" in sport:
            labels.append("Outdoor Recreation > Skate Park")
        elif "climb" in sport:
            labels.append("Outdoor Recreation > Rock Climbing")
        else:
            labels.append("Outdoor Recreation > Sports Field")

    elif group == "marina":
        if tags.get("man_made") == "pier":
            labels.append("Outdoor Recreation > Pier")
        elif tags.get("waterway") == "dock":
            labels.append("Outdoor Recreation > Harbor")
        else:
            labels.append("Outdoor Recreation > Marina")

    elif group == "gym":
        # Display-only, doesn't feed any dimension — but needs a label so
        # `/pois` can still ship it if desired. Kept out of any scoring
        # keyword set, so it's inert.
        labels.append("Sports > Gym / Fitness Center")

    # ---- Aesthetic ---------------------------------------------------
    elif group == "gallery":
        if tags.get("tourism") == "museum":
            labels.append("Arts and Entertainment > Museum")
        else:
            labels.append("Arts and Entertainment > Art Gallery")

    elif group == "cinema":
        labels.append("Arts and Entertainment > Cinema")

    elif group == "bookstore":
        labels.append("Retail > Bookstore")

    elif group == "vintage":
        shop = tags.get("shop") or ""
        if shop == "antiques":
            labels.append("Retail > Antique Store")
        elif shop == "charity":
            labels.append("Retail > Thrift Store")
        else:
            labels.append("Retail > Vintage Store")

    elif group == "indie_retail":
        shop = tags.get("shop") or ""
        if shop == "music":
            labels.append("Retail > Record Store")
        elif shop == "musical_instrument":
            labels.append("Retail > Musical Instrument Store")
        elif shop == "craft":
            labels.append("Retail > Arts and Crafts Store")
        else:
            labels.append("Retail > Arts and Crafts Store")

    elif group == "tattoo":
        labels.append("Retail > Tattoo Parlor")

    # ---- Community ---------------------------------------------------
    elif group == "library":
        labels.append("Community > Library")

    elif group == "school":
        amenity = tags.get("amenity") or ""
        if amenity == "university":
            labels.append("Community > School > University")
        elif amenity == "college":
            labels.append("Community > School > College")
        elif amenity == "kindergarten":
            labels.append("Community > Preschool")
        else:
            labels.append("Community > School")

    elif group == "worship":
        religion = (tags.get("religion") or "").lower()
        mapping = {
            "christian": "Community > Church",
            "muslim":    "Community > Mosque",
            "jewish":    "Community > Synagogue",
            "buddhist":  "Community > Buddhist Temple",
            "hindu":     "Community > Temple",
        }
        labels.append(mapping.get(religion, "Community > Place of Worship"))

    elif group == "community_centre":
        labels.append("Community > Community Center")

    elif group == "civic":
        amenity = tags.get("amenity") or ""
        mapping = {
            "townhall":    "Community > Town Hall",
            "courthouse":  "Community > Government Building",
            "post_office": "Community > Post Office",
        }
        labels.append(mapping.get(amenity, "Community > Government Building"))

    elif group == "medical":
        amenity = tags.get("amenity") or ""
        mapping = {
            "hospital":  "Health > Medical Center",
            "clinic":    "Health > Medical Center",
            "doctors":   "Health > Doctor's Office",
            "dentist":   "Health > Dentist",
            "pharmacy":  "Health > Pharmacy",
        }
        labels.append(mapping.get(amenity, "Health > Medical Center"))

    return labels


def _chains_for(tags: Dict[str, str]) -> List[Dict[str, str]]:
    """OSM exposes chain membership via the `brand` tag. We pack it into the
    FSQ-style chains shape so `is_chain()` in build.py can consume it without
    touching anything.
    """
    brand = tags.get("brand") or tags.get("operator") or ""
    if not brand:
        return []
    return [{"name": brand}]


def convert(osm_df: pd.DataFrame) -> pd.DataFrame:
    """Build a FSQ-schema DataFrame from the OSM parquet. One OSM row → one
    FSQ row (possibly with multiple labels in fsq_category_labels)."""
    rows: List[Dict] = []
    dropped_no_label = 0
    for _, r in osm_df.iterrows():
        tags = _parse_tags(r.get("osm_tags"))
        labels = _labels_for(str(r["group"]), tags)
        if not labels:
            dropped_no_label += 1
            continue
        rows.append({
            "fsq_place_id":        str(r["id"]),
            "name":                str(r["name"]),
            "latitude":            float(r["latitude"]),
            "longitude":           float(r["longitude"]),
            "address":             tags.get("addr:street", ""),
            "locality":            str(r["suburb"]),
            "region":              "NSW",
            "country":             "AU",
            "postcode":            tags.get("addr:postcode", ""),
            "fsq_category_labels": labels,
            "fsq_category_ids":    [f"cat_{abs(hash(labels[0])) & 0xFFFF}"],
            "chains":              _chains_for(tags),
            "date_created":        "",
            "date_closed":         "",
        })
    if dropped_no_label:
        print(f"[osm→fsq] dropped {dropped_no_label} POIs with no mapped label")
    return pd.DataFrame(rows)


def main() -> None:
    if not OSM_PATH.exists():
        raise SystemExit(
            f"Missing {OSM_PATH}. Run `python3 fetch_osm_pois.py` first."
        )
    osm = pd.read_parquet(OSM_PATH)
    print(f"[osm→fsq] read {len(osm)} OSM POIs from {OSM_PATH.name}")
    fsq = convert(osm)

    # Small sanity report so we can spot regressions at a glance.
    # Count labels that contain each scoring keyword of interest.
    joined = fsq["fsq_category_labels"].explode().dropna().astype(str)
    probes = ["Beach", "Park", "Restaurant", "Bar", "Café", "Library",
              "School", "Place of Worship", "Community Center",
              "Swimming Pool", "Museum", "Bookstore", "Vintage Store"]
    print("[osm→fsq] label keyword coverage:")
    for p in probes:
        n = joined.str.contains(p, case=False, regex=False).sum()
        print(f"    {p:20s} {n:6d}")

    FSQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    fsq.to_parquet(FSQ_PATH, index=False)
    print(f"[osm→fsq] wrote {len(fsq)} rows → {FSQ_PATH}")


if __name__ == "__main__":
    main()
