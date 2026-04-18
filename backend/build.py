"""
build.py  —  the Homing feature-engineering pipeline.

Reads:   data/raw/fsq_sydney.parquet       (POI-level)
Writes:  data/features/suburbs.parquet     (one row per suburb, scored)

For each Sydney suburb we compute the six personality dimensions that power
matching: social_energy, aesthetic, pace, outdoor, culinary, community.

Each dimension is the result of a small, explainable formula — a blend of
category densities (POIs of that type per km²) and a couple of derived
signals (chain-to-indie ratio, culinary entropy, late-night share). The
per-suburb raw numbers are then percentile-ranked against the full NSW
distribution so every score lands on the same 0–100 scale and is directly
comparable across dimensions.

The pipeline is deliberately simple: one pass over the POI table, a group-by
on locality, and a vectorized set of pandas operations. No ML, no magic.
Every score traces back to a sum or count a judge can reproduce with a SQL
query against fsq_sydney.parquet.
"""

import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import polars as pl

from suburbs_ref import SYDNEY_SUBURBS, SUBURB_INDEX
from categories import (
    SOCIAL_CATEGORIES,
    AESTHETIC_CATEGORIES,
    OUTDOOR_CATEGORIES,
    CULINARY_CATEGORIES,
    COMMUNITY_CATEGORIES,
    LATE_NIGHT_CATEGORIES,
    CUISINE_KEYWORDS,
    KNOWN_CHAIN_NAMES,
    normalize_label,
    label_matches_any,
    extract_cuisine,
)


ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw" / "fsq_sydney.parquet"
OUT = ROOT / "data" / "features" / "suburbs.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_chain(row) -> bool:
    """A POI is counted as a chain if FSQ tags it OR its name matches our
    30-brand AU fallback list. Either signal is enough."""
    # FSQ `chains` is a list/ndarray of dicts; if non-empty the venue is a chain
    chains = row.get("chains")
    if chains is not None:
        try:
            if len(chains) > 0:
                return True
        except TypeError:
            pass
    name = row.get("name") or ""
    name = name.strip().lower() if isinstance(name, str) else ""
    return any(brand in name for brand in KNOWN_CHAIN_NAMES)


def shannon_entropy(counts: dict) -> float:
    """Classic Shannon entropy. Higher = more diverse.
    Returns 0 if there are no items or only one category."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for n in counts.values():
        if n == 0:
            continue
        p = n / total
        entropy -= p * math.log(p)
    return entropy


def percentile_rank(series: pd.Series) -> pd.Series:
    """Returns each value's rank as a 0-100 percentile.
    Ties get the average rank. Guaranteed monotonic in the input."""
    return series.rank(pct=True, method="average") * 100


# ---------------------------------------------------------------------------
# Per-suburb feature extraction
# ---------------------------------------------------------------------------
def compute_suburb_features(poi_df: pd.DataFrame, suburb: dict) -> dict:
    """Given all POIs in a single suburb, extract the raw features we need."""
    area = suburb["area_km2"]
    n = len(poi_df)

    # Flatten: each POI has a list of category labels. Work at the label level
    # so a "Food > Restaurant > Korean Restaurant" POI contributes to culinary,
    # and a "Food > Restaurant > Korean Restaurant & Karaoke" POI could also
    # contribute to social if it had multiple labels.
    # Each POI's labels as a flat list
    all_labels = []
    for labels in poi_df["fsq_category_labels"]:
        if labels is None:
            continue
        if isinstance(labels, (list, np.ndarray)):
            all_labels.extend([l for l in labels if l])
        elif isinstance(labels, str):
            all_labels.append(labels)

    def count_matches(keyword_set) -> int:
        """How many of this suburb's category labels match `keyword_set`?"""
        return sum(1 for lbl in all_labels if label_matches_any(lbl, keyword_set))

    # --- Category counts (one per dimension) --------------------------------
    social_count     = count_matches(SOCIAL_CATEGORIES)
    aesthetic_count  = count_matches(AESTHETIC_CATEGORIES)
    outdoor_count    = count_matches(OUTDOOR_CATEGORIES)
    culinary_count   = count_matches(CULINARY_CATEGORIES)
    community_count  = count_matches(COMMUNITY_CATEGORIES)
    late_night_count = count_matches(LATE_NIGHT_CATEGORIES)

    # --- Indie vs chain -----------------------------------------------------
    chain_count = 0
    for _, row in poi_df.iterrows():
        if is_chain(row):
            chain_count += 1
    indie_count = n - chain_count
    indie_ratio = indie_count / n if n > 0 else 0.0

    # --- Culinary diversity (Shannon entropy across cuisines) ---------------
    cuisine_counter = Counter()
    for lbl in all_labels:
        c = extract_cuisine(lbl)
        if c:
            cuisine_counter[c] += 1
    culinary_entropy = shannon_entropy(cuisine_counter)
    n_cuisines = len(cuisine_counter)

    # --- Densities (POIs per km²) -------------------------------------------
    # Use log1p to soften the heavy tail (CBD otherwise dominates everything).
    poi_density_raw      = n / area
    social_density       = social_count / area
    aesthetic_density    = aesthetic_count / area
    outdoor_density      = outdoor_count / area
    culinary_density     = culinary_count / area
    community_density    = community_count / area
    late_night_density   = late_night_count / area
    late_night_share     = late_night_count / n if n > 0 else 0.0

    return {
        "suburb": suburb["name"],
        "lat": suburb["lat"],
        "lng": suburb["lng"],
        "area_km2": area,
        "rent_2br": suburb["rent_2br"],
        # counts
        "n_pois": n,
        "n_social": social_count,
        "n_aesthetic": aesthetic_count,
        "n_outdoor": outdoor_count,
        "n_culinary": culinary_count,
        "n_community": community_count,
        "n_late_night": late_night_count,
        "n_chains": chain_count,
        "n_indie": indie_count,
        "n_cuisines": n_cuisines,
        # intermediate signals
        "poi_density_raw": poi_density_raw,
        "social_density": social_density,
        "aesthetic_density": aesthetic_density,
        "outdoor_density": outdoor_density,
        "culinary_density": culinary_density,
        "community_density": community_density,
        "late_night_density": late_night_density,
        "late_night_share": late_night_share,
        "indie_ratio": indie_ratio,
        "culinary_entropy": culinary_entropy,
    }


# ---------------------------------------------------------------------------
# Dimension scoring — turn raw features into 0–100 percentile scores
# ---------------------------------------------------------------------------
def score_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Blend raw features into the six canonical dimension scores.

    Each formula is a convex combination of percentile-ranked signals. The
    weights express *which signals matter most for that dimension*; every
    number in them can be explained in one sentence.
    """
    # We percentile-rank the log1p of each density so the heavy tail doesn't
    # punish normal suburbs relative to CBD. log1p is monotonic, so rank is
    # equivalent to rank of raw density, but we keep the log for clarity when
    # we later visualize the distributions.
    def pr_log(col):
        return percentile_rank(np.log1p(df[col]))

    pr_social_density      = pr_log("social_density")
    pr_aesthetic_density   = pr_log("aesthetic_density")
    pr_outdoor_density     = pr_log("outdoor_density")
    pr_culinary_density    = pr_log("culinary_density")
    pr_community_density   = pr_log("community_density")
    pr_late_night_density  = pr_log("late_night_density")
    pr_poi_density         = pr_log("poi_density_raw")

    pr_late_night_share    = percentile_rank(df["late_night_share"])
    pr_indie_ratio         = percentile_rank(df["indie_ratio"])
    pr_culinary_entropy    = percentile_rank(df["culinary_entropy"])
    pr_n_cuisines          = percentile_rank(df["n_cuisines"])

    # ------------------------------------------------------------------
    # SOCIAL ENERGY  — bars/pubs/venues per km²  (+ late-night mass)
    # Is this a place you'd go out on a Friday night?
    # ------------------------------------------------------------------
    social = 0.70 * pr_social_density + 0.30 * pr_late_night_density

    # ------------------------------------------------------------------
    # AESTHETIC — galleries/vintage/creative retail  (+ indie vs chain)
    # Does the high street feel curated or corporate?
    # ------------------------------------------------------------------
    aesthetic = 0.55 * pr_aesthetic_density + 0.45 * pr_indie_ratio

    # ------------------------------------------------------------------
    # PACE — overall POI density + share of places that stay open late
    # Does the suburb feel busy and alive, or quiet?
    # ------------------------------------------------------------------
    pace = 0.55 * pr_poi_density + 0.45 * pr_late_night_share

    # ------------------------------------------------------------------
    # OUTDOOR — parks, beaches, trails, sports grounds per km²
    # Can you live outside?
    # ------------------------------------------------------------------
    outdoor = pr_outdoor_density

    # ------------------------------------------------------------------
    # CULINARY — how dense the food scene is, AND how diverse
    # Foodie depth = density × variety
    # ------------------------------------------------------------------
    culinary = (
        0.40 * pr_culinary_density
      + 0.35 * pr_culinary_entropy
      + 0.25 * pr_n_cuisines
    )

    # ------------------------------------------------------------------
    # COMMUNITY — libraries, schools, worship, civic spaces per km²
    # Is there infrastructure to put down roots?
    # ------------------------------------------------------------------
    community = pr_community_density

    df = df.copy()
    df["social_energy"] = social.round(1)
    df["aesthetic"]     = aesthetic.round(1)
    df["pace"]          = pace.round(1)
    df["outdoor"]       = outdoor.round(1)
    df["culinary"]      = culinary.round(1)
    df["community"]     = community.round(1)
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print(f"Loading POIs  -> {RAW}")
    if not RAW.exists():
        raise FileNotFoundError(
            f"POI file missing: {RAW}\n"
            f"Run `python backend/mock_fsq.py` first to generate the mock dataset."
        )

    # Polars is faster to load, pandas is friendlier for row-level work in the
    # feature functions. We pay the one-time convert cost for cleaner code.
    lf = pl.scan_parquet(RAW).filter(pl.col("region") == "NSW")
    pdf = lf.collect().to_pandas()
    print(f"  Loaded {len(pdf):,} POIs across {pdf['locality'].nunique()} localities")

    # --- Build per-suburb features -----------------------------------------
    print("Computing features...")
    rows = []
    for suburb in SYDNEY_SUBURBS:
        sub_df = pdf[pdf["locality"] == suburb["name"]]
        if len(sub_df) == 0:
            print(f"  [warn] no POIs for {suburb['name']}, skipping")
            continue
        rows.append(compute_suburb_features(sub_df, suburb))
    feat = pd.DataFrame(rows)
    print(f"  Got features for {len(feat)} suburbs")

    # --- Dimension scoring --------------------------------------------------
    print("Scoring dimensions...")
    scored = score_dimensions(feat)

    # --- Save ---------------------------------------------------------------
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(scored).write_parquet(OUT)
    print(f"  Saved {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")

    # --- Sanity checks — this is the "does it pass the smell test?" ---------
    print("\n== Sanity checks ==")
    def top5(col, label):
        tops = scored.nlargest(5, col)[["suburb", col]]
        print(f"\nTop 5 by {label}:")
        for _, r in tops.iterrows():
            print(f"  {r['suburb']:<20} {r[col]:>5.1f}")

    top5("social_energy", "SOCIAL ENERGY")
    top5("aesthetic",     "AESTHETIC")
    top5("outdoor",       "OUTDOOR")
    top5("culinary",      "CULINARY")
    top5("community",     "COMMUNITY")
    top5("pace",          "PACE")

    print("\n== Spot checks ==")
    def show(name):
        r = scored[scored["suburb"] == name]
        if r.empty:
            print(f"  {name:<18} (missing)")
            return
        r = r.iloc[0]
        print(f"  {name:<18}  soc={r.social_energy:>4.0f} aes={r.aesthetic:>4.0f} "
              f"pace={r.pace:>4.0f} out={r.outdoor:>4.0f} cul={r.culinary:>4.0f} "
              f"com={r.community:>4.0f}  rent=${r.rent_2br:.0f}/wk")

    for s in ["Newtown", "Marrickville", "Bondi", "Manly", "Mosman",
              "Cabramatta", "Parramatta", "Sydney CBD", "Dulwich Hill"]:
        show(s)
    print()


if __name__ == "__main__":
    main()
