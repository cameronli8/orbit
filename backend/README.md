# Homing — Backend

Sydney suburb matcher built on Foursquare OS Places. Every suburb is scored on six personality dimensions — social energy, aesthetic, pace, outdoor, culinary, community — and matched against the user's quiz answers via preference-aware cosine similarity with a smooth budget gate.

## Files

```
backend/
  categories.py     # keyword sets that define each of the six dimensions
  suburbs_ref.py    # 45 Sydney suburbs (name, lat/lng, area, median 2BR rent, character archetype)
  mock_fsq.py       # generates a FSQ-schema-compliant mock dataset for dev
  build.py          # POIs  ->  six dimension scores per suburb (writes suburbs.parquet)
  matcher.py        # quiz vector x suburbs.parquet  ->  ranked match list + explanations
  quiz.py           # 7 personality questions with answer-to-dimension weights
  api.py            # FastAPI — /quiz, /match, /suburb/{name}

data/
  raw/fsq_sydney.parquet         # POI-level dataset (from mock_fsq.py or real FSQ slice)
  features/suburbs.parquet       # one row per suburb, scored (the only file the API reads)
```

## First-time setup

```bash
pip install polars pyarrow pandas numpy fastapi "uvicorn[standard]" --break-system-packages
```

## Build the feature file

```bash
cd backend
python3 mock_fsq.py    # generates ~10k POIs across 45 Sydney suburbs
python3 build.py       # computes six dimensions per suburb with percentile-rank normalization
```

`build.py` prints a sanity check: top-5 suburbs for each dimension plus spot checks for canonical archetypes (Newtown aesthetic, Kings Cross social, Cabramatta culinary, Mosman community, Bondi outdoor).

## Swap in real FSQ data

`build.py` accepts any Parquet with the Foursquare columns `name, locality, region, fsq_category_labels, chains, latitude, longitude`. Drop it at `data/raw/fsq_sydney.parquet` and re-run. The mock generator matches this schema exactly, so nothing else changes.

For the real dataset, use `huggingface-cli login` with a token that has access to the gated `foursquare/fsq-os-places` repo, then run `polars_setup.py` with the AU/NSW filter.

## Serve

```bash
cd backend
uvicorn api:app --reload --port 8000
```

- `GET /health` → `{status, n_suburbs, n_questions}`
- `GET /quiz` → 7 questions (no scoring-weight leak)
- `POST /match {answers, budget, limit}` → ranked suburbs with `match_score`, rent, six dimension scores, three positive + two negative explanation strings
- `GET /suburb/{name}?<user_vector>` → drawer payload with dimensions, mock Domain listings, explanations relative to the user's vector

## How matching works

1. The user answers 7 lifestyle questions. Each answer nudges the six-dim user vector from a neutral 50 baseline.
2. Both the user vector and each suburb's vector are mean-centered at 50, so dimensions the user is indifferent about contribute zero to the match.
3. Cosine similarity gives a raw [0, 1] similarity rescaled to a 0-100 `match_score`.
4. A smooth logistic budget penalty down-weights (but doesn't hide) suburbs above budget — unaffordable "great fits" stay visible.
5. Explanations are templated from live computed counts (e.g. "39 galleries/studios/creative retail and 88% independent businesses"). No LLM, no hallucination.

## Sanity checks the system passes

- Kings Cross, Potts Point, Darlinghurst top social energy.
- Enmore, Erskineville, Petersham, Dulwich Hill top aesthetic.
- Coogee, Bronte, Manly, Bondi top outdoor.
- Cabramatta scores 74 on culinary (Vietnamese/Chinese density + cuisine entropy).
- Mosman, Lane Cove, Neutral Bay score high on community.
- A hardcore beach persona ranks Bondi #1, Manly #2.
- A balanced "beach but also values food and community" persona ranks Tempe #1 — the system correctly flags that this user would be better served by a balanced inner-west suburb than a spiky pure-beach one.
