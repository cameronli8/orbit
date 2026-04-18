"""
fetch_hf_samples.py  —  pull a small real-Foursquare sample for each rated suburb.

This is a ONE-SHOT script. It reads the gated Foursquare OS Places parquet
dump from Hugging Face (foursquare/fsq-os-places), narrows to ~1km around
each of the 45 rated suburb centroids, keeps the top 8 well-categorised
venues, and writes data/hf_samples.json. The drawer UI then shows those
venues under a "Live from Foursquare OS Places (Hugging Face)" card —
honest proof that the provided dataset is flowing into the app.

It is intentionally *additive*: it does not touch the scoring pipeline,
the parquet that matcher.py reads, or anything the deployed /match path
depends on. Safe to run (or skip) without risking the demo.

Run once locally with a valid HF_TOKEN in .env, commit the resulting
data/hf_samples.json, and Railway serves it statically. No runtime HF
calls on the demo path.

    python backend/fetch_hf_samples.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path

# Load .env so HF_TOKEN is visible even when invoked outside a shell that
# sourced it. .env lives one directory up from this file.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    # dotenv is listed in requirements.txt; fall back gracefully if someone
    # runs this before `pip install -r`.
    pass

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from suburbs_ref import SYDNEY_SUBURBS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")

# HF glob URI for the FSQ OS Places parquet shards. The dt= subdir is the
# monthly release date; pick one that's been published. Update when a newer
# release rolls out — any valid release works, the data is stable month
# over month for our purposes.
HF_GLOB = (
    "hf://datasets/foursquare/fsq-os-places/"
    "release/dt=2026-04-14/places/parquet/*.parquet"
)

# Sydney metro bbox — used only for predicate pushdown so Polars only streams
# parquet row groups that overlap Sydney. Shrinks the effective scan from
# hundreds of GB to ~1 GB.
S, W, N, E = -34.10, 150.60, -33.55, 151.35

# Per-suburb search radius. ~1.0 km at this latitude → roughly 0.009° lat,
# 0.011° lng. Deliberately tight so each suburb's sample is locally relevant
# (not spillover from a neighbouring main street).
R_LAT = 0.0090
R_LNG = 0.0110

# Top-N POIs to keep per suburb. 8 gives the drawer enough to look rich
# without bloating the JSON.
PER_SUBURB = 8

# Foursquare top-level category buckets a renter would actually care about.
# We sort POIs so entries in these buckets win ties against the noisier
# buckets (Business and Professional Services, Community and Government,
# Travel and Transportation) which are full of offices, funeral homes, and
# conveyancers — true to the source but useless in a drawer.
INTERESTING_BUCKETS = {
    "Dining and Drinking":     0,
    "Arts and Entertainment":  1,
    "Landmarks and Outdoors":  2,
    "Sports and Recreation":   3,
    "Retail":                  4,
    "Event":                   5,
    "Events":                  5,
    "Travel and Transportation": 6,
}
# Anything NOT in the above ranks gets bucket-priority 99 — bumped to the
# back of the list, only surfaced if a suburb has nothing else in radius.

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "hf_samples.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _shorten_category(labels) -> str:
    """Foursquare categories arrive as ['Dining and Drinking > Restaurant > Italian'].
    We keep only the most specific leaf ('Italian') — that's what a renter
    would call the place. Falls back to an empty string if the label is
    missing or malformed."""
    if not labels:
        return ""
    first = labels[0] if isinstance(labels, list) else labels
    if not isinstance(first, str):
        return ""
    leaf = first.split(">")[-1].strip()
    return leaf


def _top_level(labels) -> str:
    """Coarse bucket — 'Dining and Drinking', 'Landmarks and Outdoors',
    etc. Used so the UI can group venues even when the leaf category is
    oddly specific ('Gelato Shop' etc)."""
    if not labels:
        return ""
    first = labels[0] if isinstance(labels, list) else labels
    if not isinstance(first, str):
        return ""
    return first.split(">")[0].strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def fetch() -> None:
    if not HF_TOKEN:
        raise SystemExit(
            "HF_TOKEN not set. Add it to .env or export it before running.\n"
            "Get a read token at https://huggingface.co/settings/tokens "
            "after accepting terms at https://huggingface.co/datasets/foursquare/fsq-os-places."
        )

    print(f"[hf] streaming FSQ OS Places from {HF_GLOB}")
    lf = pl.scan_parquet(HF_GLOB, storage_options={"token": HF_TOKEN})

    # Predicate-push filters into the parquet reader so only Sydney row groups
    # actually download. `collect(streaming=True)` keeps memory bounded.
    sydney = (
        lf
        .filter(pl.col("country") == "AU")
        .filter(pl.col("latitude").is_between(S, N))
        .filter(pl.col("longitude").is_between(W, E))
        .filter(pl.col("date_closed").is_null())
        .filter(pl.col("name").is_not_null())
        .select(
            "fsq_place_id",
            "name",
            "latitude",
            "longitude",
            "fsq_category_labels",
        )
        .collect(streaming=True)
    )
    print(f"[hf] pulled {sydney.height:,} Sydney POIs")

    suburbs_out: dict[str, list[dict]] = {}
    skipped = 0
    for sub in SYDNEY_SUBURBS:
        name = sub["name"]
        lat, lng = sub["lat"], sub["lng"]

        near = sydney.filter(
            pl.col("latitude").is_between(lat - R_LAT, lat + R_LAT)
            & pl.col("longitude").is_between(lng - R_LNG, lng + R_LNG)
            # Require at least one category label so the drawer has something
            # to render next to the venue name.
            & pl.col("fsq_category_labels").is_not_null()
            & (pl.col("fsq_category_labels").list.len() > 0)
        )

        # Materialise rows, rank by (bucket preference, fsq_place_id) so
        # interesting-bucket entries win ties over "Choice Home Loans" and
        # "Capital Conveyancing Services". fsq_place_id provides a stable
        # tiebreaker so re-runs produce identical JSON.
        rows = list(near.iter_rows(named=True))
        def _rank(r):
            bucket = _top_level(r["fsq_category_labels"])
            return (INTERESTING_BUCKETS.get(bucket, 99), r["fsq_place_id"])
        rows.sort(key=_rank)

        entries = []
        for r in rows[:PER_SUBURB]:
            labels = r["fsq_category_labels"] or []
            entries.append({
                "id":       r["fsq_place_id"],
                "name":     r["name"],
                "category": _shorten_category(labels),
                "bucket":   _top_level(labels),
                "lat":      float(r["latitude"]),
                "lng":      float(r["longitude"]),
            })

        if not entries:
            skipped += 1
            print(f"[hf]   {name:<22} (no POIs in radius — skipped)")
            continue

        suburbs_out[name] = entries
        leaf_preview = ", ".join(e["name"] for e in entries[:3])
        print(f"[hf]   {name:<22} {len(entries)} POIs — {leaf_preview}…")

    payload = {
        "source":        "Foursquare Open Source Places (Hugging Face)",
        "dataset":       "foursquare/fsq-os-places",
        "dataset_url":   "https://huggingface.co/datasets/foursquare/fsq-os-places",
        "release":       HF_GLOB.split("dt=")[1].split("/")[0],
        "fetched_at":    _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "radius_km":     1.0,
        "per_suburb":    PER_SUBURB,
        "n_suburbs":     len(suburbs_out),
        "n_pois_total":  sum(len(v) for v in suburbs_out.values()),
        "suburbs":       suburbs_out,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    size_kb = OUT_PATH.stat().st_size / 1024
    print(
        f"[hf] wrote {OUT_PATH.relative_to(OUT_PATH.parent.parent.parent)} "
        f"— {payload['n_pois_total']} POIs across "
        f"{payload['n_suburbs']} suburbs ({size_kb:.1f} KB)"
    )
    if skipped:
        print(f"[hf] {skipped} suburb(s) had no POIs in radius — drawer will skip them.")


if __name__ == "__main__":
    fetch()
