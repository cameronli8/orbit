"""
fetch_suburb_polygons.py  —  one-shot downloader for Sydney suburb polygons.

Pulls an all-NSW suburb GeoJSON (ABS-derived, via the tonywr71/GeoJson-Data
mirror), filters to a Greater Sydney bounding box, and writes the result to
data/suburbs.geojson. Every Sydney suburb is included — ones Orbit has
scored are tagged `"rated": true` so the frontend can differentiate.

Usage:
    python3 backend/fetch_suburb_polygons.py

Notes:
    - The primary source covers all ~4500 NSW suburbs. We bbox-filter to
      Greater Sydney so the file stays tractable in the browser.
    - Property names across NSW datasets are inconsistent — we try the
      common ones (SSC_NAME, nsw_loca_2, SAL_NAME21, name).
    - Kings Cross isn't a standalone ABS SSC suburb; we alias it to
      Potts Point's polygon so hover/click still works in that area.
    - If any suburb from suburbs_ref.py is still unmatched after
      bbox-filtering, the script prints close-name suggestions.
"""

import difflib
import json
import ssl
import sys
import urllib.request
from pathlib import Path

from suburbs_ref import SUBURB_NAMES

# Primary: all NSW suburbs. Fallback: the smaller Sydney-only file.
SOURCES = [
    "https://raw.githubusercontent.com/tonywr71/GeoJson-Data/master/suburb-10-nsw.geojson",
    "https://raw.githubusercontent.com/tim-massey/sydney-geojson/master/sydney.geojson",
]

OUT_PATH = Path(__file__).parent.parent / "data" / "suburbs.geojson"

# Greater Sydney bounding box. Loose enough to include Penrith (west),
# Helensburgh (south), Berowra (north), and Cronulla/Palm Beach (east).
BBOX = {
    "lat_min": -34.25,
    "lat_max": -33.45,
    "lng_min": 150.55,
    "lng_max": 151.45,
}

# Common property name fields across NSW suburb GeoJSON flavours.
NAME_FIELDS = ("SSC_NAME", "ssc_name", "nsw_loca_2", "NSW_LOCA_2",
               "SAL_NAME21", "sal_name21", "name", "NAME", "Name")

# Name aliases — left side is what's in suburbs_ref.py, right side is what
# lives in the source dataset. Fill this in as misses appear.
#
# Kings Cross isn't a standalone ABS SSC suburb; it sits inside Potts Point.
# We reuse Potts Point's polygon so hover/click still works in that area —
# they're two features in the output, both carrying their own canonical name.
NAME_ALIASES = {
    "Sydney CBD": "Sydney",
    "Kings Cross": "Potts Point",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ssl_context() -> ssl.SSLContext:
    """macOS' system Python frequently can't find its CA bundle. Prefer
    certifi, fall back to the OS default, and finally unverified for a
    one-shot download of trusted public data."""
    try:
        import certifi  # type: ignore
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    try:
        ctx = ssl.create_default_context()
        ctx.load_default_certs()
        return ctx
    except Exception:
        print("WARN: falling back to unverified SSL.", file=sys.stderr)
        return ssl._create_unverified_context()


def strip_suffix(name: str) -> str:
    """'Abbotsford (NSW)' → 'Abbotsford'."""
    if " (" in name and name.endswith(")"):
        return name.split(" (")[0].strip()
    return name.strip()


def get_feature_name(props: dict) -> str | None:
    for f in NAME_FIELDS:
        v = props.get(f)
        if v:
            return str(v)
    return None


def feature_centroid(geom: dict) -> tuple[float, float] | None:
    """Rough centroid via averaging coordinates of the first/outer ring.
    Good enough for bbox-filtering — we don't need true centroid math."""
    if not geom:
        return None
    coords = geom.get("coordinates")
    if not coords:
        return None
    t = geom.get("type")

    def _avg(pts):
        if not pts:
            return None
        lng_sum = 0.0
        lat_sum = 0.0
        n = 0
        for p in pts:
            if len(p) < 2:
                continue
            lng_sum += p[0]
            lat_sum += p[1]
            n += 1
        if n == 0:
            return None
        return lng_sum / n, lat_sum / n

    if t == "Polygon":
        # outer ring
        return _avg(coords[0]) if coords else None
    if t == "MultiPolygon":
        # first polygon, outer ring
        if coords and coords[0] and coords[0][0]:
            return _avg(coords[0][0])
    return None


def in_sydney(geom: dict) -> bool:
    c = feature_centroid(geom)
    if c is None:
        return False
    lng, lat = c
    return (
        BBOX["lat_min"] <= lat <= BBOX["lat_max"]
        and BBOX["lng_min"] <= lng <= BBOX["lng_max"]
    )


def download(url: str, ctx) -> bytes | None:
    print(f"Downloading {url}…", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=120, context=ctx) as resp:
            raw = resp.read()
        print(f"  downloaded {len(raw) / 1024:.1f} KB", flush=True)
        return raw
    except Exception as e:
        print(f"  failed — {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ctx = _ssl_context()

    raw: bytes | None = None
    for url in SOURCES:
        raw = download(url, ctx)
        if raw:
            break
    if raw is None:
        print(
            "\nERROR: all sources failed.\n"
            "Hint: on macOS this is usually a cert issue. Try:\n"
            "  python3 -m pip install --upgrade certifi\n"
            "or run \"Install Certificates.command\" from your Python install.",
            file=sys.stderr,
        )
        return 1

    gj = json.loads(raw)
    if gj.get("type") != "FeatureCollection":
        print("ERROR: expected FeatureCollection at root.", file=sys.stderr)
        return 1
    features = gj.get("features", [])
    print(f"Source has {len(features)} features.", flush=True)

    # ---------------- Bbox-filter to Greater Sydney ------------------------
    sydney_feats: list[dict] = []
    for f in features:
        geom = f.get("geometry")
        if not geom:
            continue
        if not in_sydney(geom):
            continue
        props = f.get("properties") or {}
        name = get_feature_name(props)
        if not name:
            continue
        sydney_feats.append({
            "type": "Feature",
            "properties": {"name": strip_suffix(name)},
            "geometry": geom,
        })

    print(f"Greater Sydney bbox contains {len(sydney_feats)} suburbs.",
          flush=True)

    # ---------------- Mark which ones we have data for ---------------------
    # Apply aliases: canonical names from suburbs_ref.py point at whatever
    # name lives in the source (e.g. Kings Cross → Potts Point).
    rated_source_names = {
        strip_suffix(NAME_ALIASES.get(s, s)).lower()
        for s in SUBURB_NAMES
    }

    # Build a lookup so we can emit TWO features for aliased pairs
    # (e.g. one "Kings Cross" + one "Potts Point" sharing geometry).
    by_source_name = {f["properties"]["name"].lower(): f for f in sydney_feats}

    # Track which canonical orbit names we managed to find geometry for.
    orbit_matched: dict[str, dict] = {}
    missing: list[str] = []
    for original in SUBURB_NAMES:
        alias = NAME_ALIASES.get(original, original)
        key = strip_suffix(alias).lower()
        feat = by_source_name.get(key)
        if feat is None:
            missing.append(original)
            continue
        orbit_matched[original] = feat

    # ---------------- Build the final FeatureCollection --------------------
    # 1) All Sydney suburbs (un-rated).
    # 2) Override with rated canonical names — and insert extra features for
    #    aliases (Kings Cross, Sydney CBD) that share geometry with their
    #    source suburb.
    output: list[dict] = []
    emitted_names: set[str] = set()

    for f in sydney_feats:
        nm = f["properties"]["name"]
        nm_lower = nm.lower()
        is_rated = nm_lower in rated_source_names
        # If this source feature is rated, we may want to use the ORBIT
        # canonical name on it (e.g. "Sydney" → "Sydney CBD"). If it's an
        # alias target shared by multiple canonical names, we emit one
        # feature per canonical name so hover/click resolves correctly.
        canonical_names = [
            orig for orig in SUBURB_NAMES
            if strip_suffix(NAME_ALIASES.get(orig, orig)).lower() == nm_lower
        ]
        if canonical_names:
            for cn in canonical_names:
                output.append({
                    "type": "Feature",
                    "properties": {"name": cn, "rated": True},
                    "geometry": f["geometry"],
                })
                emitted_names.add(cn.lower())
        else:
            output.append({
                "type": "Feature",
                "properties": {"name": nm, "rated": False},
                "geometry": f["geometry"],
            })
            emitted_names.add(nm_lower)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": output,
    }))
    size_kb = OUT_PATH.stat().st_size / 1024
    n_rated = sum(1 for f in output if f["properties"].get("rated"))
    print(
        f"\nWrote {OUT_PATH.relative_to(OUT_PATH.parents[1])} — "
        f"{len(output)} total suburbs, {n_rated} rated "
        f"(of {len(SUBURB_NAMES)} in suburbs_ref.py), {size_kb:.1f} KB.",
        flush=True,
    )

    if missing:
        all_source_names = sorted({f["properties"]["name"] for f in sydney_feats})
        print("\nStill unmatched (not in the source's Sydney bbox):")
        for m in missing:
            suggestions = difflib.get_close_matches(
                m, all_source_names, n=5, cutoff=0.55
            )
            if suggestions:
                print(f"  - {m}  →  closest names: {', '.join(suggestions)}")
            else:
                print(f"  - {m}  →  no close match")
        print("\nAdd entries to NAME_ALIASES (top of this script) and rerun.")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
