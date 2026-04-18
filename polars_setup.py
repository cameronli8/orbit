"""
Foursquare OS Places — Polars Setup
====================================

Dataset: https://huggingface.co/datasets/foursquare/fsq-os-places
~193 GB of global POI data in Parquet format.

Strategy:
    1. Use Polars' LAZY API to filter the full dataset down to a manageable slice
       (e.g., one city or country) without downloading all 193 GB.
    2. Save that slice locally as a Parquet file.
    3. Do the rest of the hackathon analysis on the local slice — fast and free.

Install:
    pip install polars pyarrow huggingface_hub --break-system-packages

    # Optional (for geospatial work later):
    pip install geopandas shapely folium --break-system-packages
"""

import polars as pl
from pathlib import Path


# ---------------------------------------------------------------------------
# Config — adjust these for your target area
# ---------------------------------------------------------------------------

# The release date (check https://huggingface.co/datasets/foursquare/fsq-os-places
# for the latest. Update here if a newer release exists.)
RELEASE_DATE = "dt=2026-04-14"

# Where the Parquet files live on Hugging Face
HF_BASE = "hf://datasets/foursquare/fsq-os-places"
PLACES_GLOB = f"{HF_BASE}/release/{RELEASE_DATE}/places/parquet/*.parquet"
CATEGORIES_GLOB = f"{HF_BASE}/release/{RELEASE_DATE}/categories/parquet/*.parquet"

# Target area for the local slice — edit these!
TARGET_COUNTRY = "US"
TARGET_REGION = "NY"          # US state code, or None to skip
TARGET_LOCALITY = None        # City name, or None to skip (e.g., "New York")

# Where to save the local slice
LOCAL_SLICE = Path("nyc_places.parquet")
CATEGORIES_LOCAL = Path("categories.parquet")


# ---------------------------------------------------------------------------
# Step 1: Peek at the schema (does NOT download the whole dataset)
# ---------------------------------------------------------------------------

def peek_schema():
    """Print the column names and dtypes so you know what you're working with."""
    lf = pl.scan_parquet(PLACES_GLOB)
    print("Places schema:")
    print(lf.collect_schema())
    print()

    lf_cats = pl.scan_parquet(CATEGORIES_GLOB)
    print("Categories schema:")
    print(lf_cats.collect_schema())


# ---------------------------------------------------------------------------
# Step 2: Build a local slice for your target area
# ---------------------------------------------------------------------------

def build_local_slice():
    """
    Lazy-scan the full dataset, filter down to the target area, and save locally.
    Polars pushes the filter down into the Parquet reader so only matching
    row groups are fetched from Hugging Face — much faster than downloading all.
    """
    print(f"Building local slice for {TARGET_COUNTRY}/{TARGET_REGION}...")

    lf = pl.scan_parquet(PLACES_GLOB)

    # Build filter expression step by step
    filter_expr = pl.col("country") == TARGET_COUNTRY
    if TARGET_REGION:
        filter_expr = filter_expr & (pl.col("region") == TARGET_REGION)
    if TARGET_LOCALITY:
        filter_expr = filter_expr & (pl.col("locality") == TARGET_LOCALITY)

    # Only keep rows for currently-open venues (date_closed is null)
    filter_expr = filter_expr & pl.col("date_closed").is_null()

    sliced = lf.filter(filter_expr).collect(streaming=True)

    print(f"  -> {len(sliced):,} places matched")
    sliced.write_parquet(LOCAL_SLICE)
    print(f"  -> saved to {LOCAL_SLICE.resolve()}")

    # Also cache the full categories taxonomy — it's tiny
    cats = pl.scan_parquet(CATEGORIES_GLOB).collect()
    cats.write_parquet(CATEGORIES_LOCAL)
    print(f"  -> categories ({len(cats):,} rows) saved to {CATEGORIES_LOCAL.resolve()}")


# ---------------------------------------------------------------------------
# Step 3: Load the local slice (use this in all follow-up notebooks/scripts)
# ---------------------------------------------------------------------------

def load_places() -> pl.DataFrame:
    """Load the local slice of places. Fast — this is a local file."""
    if not LOCAL_SLICE.exists():
        raise FileNotFoundError(
            f"{LOCAL_SLICE} not found. Run build_local_slice() first."
        )
    return pl.read_parquet(LOCAL_SLICE)


def load_categories() -> pl.DataFrame:
    """Load the Foursquare category taxonomy."""
    if not CATEGORIES_LOCAL.exists():
        raise FileNotFoundError(
            f"{CATEGORIES_LOCAL} not found. Run build_local_slice() first."
        )
    return pl.read_parquet(CATEGORIES_LOCAL)


# ---------------------------------------------------------------------------
# Step 4: Example analyses to get you rolling
# ---------------------------------------------------------------------------

def example_top_categories(df: pl.DataFrame, n: int = 20) -> pl.DataFrame:
    """Which categories appear most often in the slice?"""
    return (
        df.explode("fsq_category_labels")
          .group_by("fsq_category_labels")
          .agg(pl.len().alias("count"))
          .sort("count", descending=True)
          .head(n)
    )


def example_chain_vs_independent(df: pl.DataFrame, n: int = 20) -> pl.DataFrame:
    """Names that appear many times = likely chains."""
    return (
        df.group_by("name")
          .agg(pl.len().alias("locations"))
          .sort("locations", descending=True)
          .head(n)
    )


def example_density_by_locality(df: pl.DataFrame) -> pl.DataFrame:
    """POI counts per city/neighborhood in the slice."""
    return (
        df.group_by("locality")
          .agg(pl.len().alias("poi_count"))
          .sort("poi_count", descending=True)
    )


def example_filter_by_category(df: pl.DataFrame, keyword: str) -> pl.DataFrame:
    """Find all places whose category labels contain `keyword` (case-insensitive)."""
    return df.filter(
        pl.col("fsq_category_labels")
          .cast(pl.List(pl.Utf8))
          .list.eval(pl.element().str.to_lowercase().str.contains(keyword.lower()))
          .list.any()
    )


# ---------------------------------------------------------------------------
# Run from CLI:  python polars_setup.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. First time only — peek at columns
    # peek_schema()

    # 2. First time only — build a local slice (~may take a few minutes over network)
    if not LOCAL_SLICE.exists():
        build_local_slice()

    # 3. From here on, load local and analyze
    df = load_places()
    print(f"\nLoaded {len(df):,} places from {LOCAL_SLICE}")
    print(f"Columns: {df.columns}\n")

    print("Top 10 categories:")
    print(example_top_categories(df, n=10))

    print("\nTop 10 most common names (chains):")
    print(example_chain_vs_independent(df, n=10))
