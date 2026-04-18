# Orbit — Backend

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
pip install -r backend/requirements.txt --break-system-packages
```

### AI layer (optional but recommended)

Orbit writes suburb-specific copy and user personality profiles with OpenAI.
Set up a `.env` in the project root:

```bash
cp .env.example .env
# edit .env and paste your OpenAI key into OPENAI_API_KEY=
```

Without a key the backend still works — every LLM response falls back to the
deterministic template strings. Hit `GET /health` and look at the `llm` field
to see whether the AI layer is live.

### Live rental listings (Domain API)

Out of the box the map shows generated mock listings (with deep links to
the matching `realestate.com.au` filter). To swap them for real Sydney
rentals, sign up at [developer.domain.com.au](https://developer.domain.com.au/),
create a project with the `Listings` package, and paste the OAuth credentials
into `.env`:

```env
DOMAIN_CLIENT_ID=your-client-id
DOMAIN_CLIENT_SECRET=your-client-secret
```

The first time `/match` is called for a suburb, `real_listings.fetch_listings`
exchanges credentials for a Bearer token at `auth.domain.com.au/v1/connect/token`,
hits `POST /v1/listings/residential/_search` filtered to that suburb + a
2BR price band, maps the response into Orbit's listing schema, and writes
the result to `data/listings_cache.json`. Subsequent calls within 24h
(`ORBIT_LISTINGS_CACHE_H`) return cached results — keeping demo-day usage
well inside the free tier.

`GET /health` returns `"listings_source": "domain"` when the live path is
active, `"mock"` otherwise. Any failure (auth error, network timeout, empty
response) silently falls back to mocks so the demo never breaks.

Smoke test from the CLI:

```bash
cd backend
python3 real_listings.py Bondi 2
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

For the real dataset, set `HF_TOKEN` in `.env` (requires accepting terms on the gated `foursquare/fsq-os-places` repo) and run `python3 backend/fetch_hf_samples.py`. That ships ~8 real venues per rated suburb into `data/hf_samples.json`, which the drawer surfaces as a "Real places nearby" card. The scoring pipeline stays on the OSM-derived `fsq_sydney.parquet`.

## Serve

```bash
cd backend
uvicorn api:app --reload --port 8000
```

- `GET /health` → `{status, n_suburbs, n_questions, llm}`
- `GET /quiz` → 7 questions (no scoring-weight leak)
- `POST /match {answers, budget, limit}` → ranked suburbs with `match_score`, rent, six dimension scores, three positive + two negative explanation strings
- `POST /profile {answers}` → `{headline, summary, wants, source, user_vector}` — LLM-written renter persona
- `GET /suburb/{name}?<user_vector>&persona=…` → drawer payload with dimensions, mock Domain listings, template explanations, the strict evidence blob, **and an `ai` block** (LLM headline, summary, positive/negative bullets) grounded in the same evidence

Then open `../index.html` in a browser. The frontend is a single static file that talks to `http://localhost:8000` — CORS is wildcard-open so no proxy is needed.

## How matching works

1. The user answers 7 lifestyle questions. Each answer nudges the six-dim user vector from a neutral 50 baseline.
2. Both the user vector and each suburb's vector are mean-centered at 50, so dimensions the user is indifferent about contribute zero to the match.
3. Cosine similarity gives a raw [0, 1] similarity rescaled to a 0-100 `match_score`.
4. A smooth logistic budget penalty down-weights (but doesn't hide) suburbs above budget — unaffordable "great fits" stay visible.
5. Template explanations are computed from live counts and per-suburb sub-category breakdowns (e.g. "8 parks and 2 playgrounds within 0.6 km²"). Because the templates pull from `breakdowns_json` in `suburbs.parquet`, they only ever cite sub-categories with a non-zero count in that specific suburb — so Chippendale never claims beaches.
6. On top of that, `llm.py` asks GPT-4o-mini to write a second-person persona and tailored suburb copy. The model sees only the same strict evidence blob (`matcher.evidence_for`) and is forbidden from citing any feature not in it. If the API is unreachable or the key is missing, the frontend gracefully falls back to the template strings.

## Sanity checks the system passes

- Kings Cross, Potts Point, Darlinghurst top social energy.
- Enmore, Erskineville, Petersham, Dulwich Hill top aesthetic.
- Coogee, Bronte, Manly, Bondi top outdoor.
- Cabramatta scores 74 on culinary (Vietnamese/Chinese density + cuisine entropy).
- Mosman, Lane Cove, Neutral Bay score high on community.
- A hardcore beach persona ranks Bondi #1, Manly #2.
- A balanced "beach but also values food and community" persona ranks Tempe #1 — the system correctly flags that this user would be better served by a balanced inner-west suburb than a spiky pure-beach one.
