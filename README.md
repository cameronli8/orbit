# Orbit

Sydney renter suburb-matching that starts with *who you are*, not what you can afford.

Most rent tools ask you to pick a suburb and sort by price. Orbit inverts that: answer a 7-question personality quiz, and the whole Sydney metro colours from red (poor fit) through grey (neutral) to green (your kind of place). Budget is a soft gate, not a filter — aspirational suburbs stay visible, they just rank lower.

Built as a solo-dev submission for Data-Hack 2026.

## The six dimensions

Every Sydney suburb and every quiz-taker gets scored on the same axes:

- **social_energy** — nightlife, bar density, late-night venues
- **aesthetic** — indie cafés, galleries, vintage, chain-to-indie ratio
- **pace** — walkability, density, 24/7 venues vs residential calm
- **outdoor** — beaches, parks, surf, bushwalks, sports clubs
- **culinary** — cuisine diversity (Shannon entropy), restaurant density
- **community** — schools, libraries, community venues, low churn

The vectors are mean-centred at 50 before cosine — dimensions you don't care about don't spuriously inflate matches, so a "chill beach + food" persona doesn't get Kings Cross because Kings Cross is spiky on things they're neutral on.

## User flow

1. Quiz overlay fetches `GET /quiz` and asks 7 lifestyle questions with large tappable cards, ending on a weekly rent slider.
2. `POST /match` returns two parallel lists: the top-ranked **45 curated suburbs** (with full detail, explanations, mock Domain listings) and a **473-suburb heatmap payload** (all of Sydney, compact, budget-free).
3. A Leaflet `L.GridLayer` renders a per-pixel gaussian-interpolated heatmap across the metro — a 9-stop red→grey→green gradient keyed to personality fit, not price.
4. Clicking any suburb opens a right-side drawer with all six dimension bars, template explanations pulled from real POI counts, an LLM-written persona paragraph, three mock Domain listings, and a **"Real places nearby"** card with venues from the Foursquare OS Places dataset (Hugging Face).
5. A toggle flips the heatmap to commute time from the CBD — same UI, different lens.

## Architecture

```
Leaflet frontend (index.html)
        │
        │   fetch() against same origin
        ▼
FastAPI (backend/api.py)
        │
        ├── /quiz           quiz.py           → 7 questions
        ├── /match          matcher.py        → ranked 45 + heatmap 473
        ├── /suburb/{name}  matcher + llm     → drawer payload + AI copy
        ├── /pois           pois.py           → OSM POIs for the map layer
        └── /profile        llm.py            → persona from GPT-4o-mini
                │
                ▼
        data/features/suburbs.parquet   (only file the request path reads)
```

The feature parquet is rebuilt offline by a pipeline that runs once, not per request:

```
OSM (Overpass)  →  fetch_osm_pois.py  →  osm_pois_sydney.parquet
                                             │
                                             ▼
                                      osm_to_fsq.py
                                             │
                                             ▼
                                      fsq_sydney.parquet
                                             │
                                             ▼
                                        build.py
                                             │
                                             ▼
                              features/suburbs.parquet
```

Matching is deterministic: preference-aware cosine similarity, smooth-logistic budget gate (0.3–1.0 multiplier), template explanations grounded in per-suburb breakdown counts so the copy never hallucinates a beach in Chippendale. LLM copy from `llm.py` sees the same strict evidence blob and is instructed to cite only what's in it.

## Tech stack

Vanilla JavaScript + Leaflet on the frontend. FastAPI + Polars + pandas on the backend. Parquet for the feature store. OpenAI GPT-4o-mini for persona and per-suburb copy (optional — falls back to templates). Deployed on Railway via Nixpacks; `index.html` is served as a static file by FastAPI's `StaticFiles` mount so the frontend and API share an origin.

## Quickstart

```bash
git clone https://github.com/cameronli8/orbit.git
cd orbit
pip install -r backend/requirements.txt --break-system-packages
cd backend
uvicorn api:app --reload --port 8000
```

Open `http://localhost:8000/` — the FastAPI app serves `index.html` from the repo root, so the frontend and API are same-origin and no CORS or proxy config is needed.

### Optional environment variables

Copy `.env.example` to `.env` and fill in whichever of these you want:

- `OPENAI_API_KEY` — unlocks the LLM-written persona and suburb copy. Without it the frontend shows the deterministic template strings.
- `DOMAIN_CLIENT_ID` / `DOMAIN_CLIENT_SECRET` — swaps the mock rental listings for live Sydney 2BR Domain listings (cached for 24 h).
- `HF_TOKEN` — required only if you want to regenerate `data/hf_samples.json` from scratch. Needs access to the gated `foursquare/fsq-os-places` dataset on Hugging Face.

`GET /health` reports which data layers are live (`llm`, `listings_source`).

## Data sources

Orbit is honest about where its data comes from:

- **Foursquare Open Source Places** (via Hugging Face, `foursquare/fsq-os-places`) — the dataset provided for the hackathon. Orbit uses it directly for the drawer's "Real places nearby" card, sampled at the 45 rated suburbs and committed as `data/hf_samples.json`. The one-shot fetcher is `backend/fetch_hf_samples.py`.
- **OpenStreetMap** (via the Overpass API) — the scoring pipeline's actual POI source. OSM gives every-suburb coverage without HF auth, and `osm_to_fsq.py` reshapes its output into the Foursquare schema `build.py` expects. Used because the full FSQ slice is gated and Railway's free tier can't host hundreds of gigabytes.
- **ABS SAL suburb polygons** — pulled once by `fetch_suburb_polygons.py` into `data/suburbs.geojson` for the frontend's hover/click targets and suburb boundary rendering.
- **NSW Fair Trading Rent and Sales Report** (December 2025 quarter) — postcode-level 2BR medians hand-curated into `suburbs_ref.py`.
- **Domain Listings API** — live 2BR listings when credentials are configured; otherwise mock listings deep-linking to the matching realestate.com.au filter.

## Repo layout

```
index.html                    # Leaflet SPA — quiz overlay, heatmap, drawer
sw.js, manifest.json, icons/  # PWA wrapper so it installs to a phone home screen
backend/
  api.py                      # FastAPI app + StaticFiles mount
  quiz.py                     # 7 questions + answer-to-dimension weights
  matcher.py                  # cosine + budget gate + template explanations
  build.py                    # pipeline: POIs → suburbs.parquet
  categories.py               # keyword sets defining the six dimensions
  suburbs_ref.py              # 45 rated suburbs (lat/lng, area, rent, character)
  pois.py, real_listings.py   # /pois and Domain listings endpoints
  llm.py                      # GPT-4o-mini persona + suburb copy
  fetch_osm_pois.py           # one-shot: Overpass → osm_pois_sydney.parquet
  osm_to_fsq.py               # one-shot: OSM schema → FSQ schema
  fetch_suburb_polygons.py    # one-shot: ABS SAL → suburbs.geojson
  fetch_hf_samples.py         # one-shot: Hugging Face FSQ → hf_samples.json
  mock_fsq.py                 # offline fallback POI generator
data/
  features/suburbs.parquet    # the scored suburbs — only file served by the API
  raw/                        # upstream POI parquets
  suburbs.geojson             # polygon boundaries for the frontend
  hf_samples.json             # Foursquare venues for the drawer card
Procfile, railway.json, runtime.txt, requirements.txt   # Railway deploy
```

For deeper notes on the scoring pipeline, sanity checks, and LLM integration, see [`backend/README.md`](backend/README.md).

## API surface

- `GET /health` — `{status, n_suburbs, n_questions, llm, listings_source}`
- `GET /quiz` — 7 questions, scoring weights stripped
- `POST /match` — body `{answers, budget, limit}`, returns `{user_vector, suburbs, heatmap}`
- `POST /profile` — body `{answers}`, returns `{headline, summary, wants, source, user_vector}`
- `GET /suburb/{name}?<user_vector query params>&persona=…` — full drawer payload including AI block
- `GET /pois` — compact OSM POI list for the map layer

## Sanity checks the backend passes

Kings Cross, Potts Point, and Darlinghurst top social energy. Enmore, Erskineville, Petersham, and Dulwich Hill top aesthetic. Coogee, Bronte, Manly, and Bondi top outdoor. Cabramatta tops culinary on Vietnamese-Chinese density plus cuisine entropy. Mosman, Lane Cove, and Neutral Bay top community. A hardcore beach persona ranks Bondi first and Manly second; a balanced "beach but also values food and community" persona correctly prefers Tempe over spiky pure-beach suburbs, which is the behaviour that justifies the mean-centring trick.

## Data attribution

- **Foursquare Open Source Places** (ODbL) — [huggingface.co/datasets/foursquare/fsq-os-places](https://huggingface.co/datasets/foursquare/fsq-os-places) — powers the drawer's "Real places nearby" card.
- **OpenStreetMap** (ODbL) — © OpenStreetMap contributors — powers the scoring pipeline (7,382 POIs across the Sydney metro) via the public [Overpass API](https://overpass-api.de/).
- **ABS Suburbs and Localities** (CC BY 4.0) — [Australian Bureau of Statistics](https://www.abs.gov.au/) — polygon boundaries for hover/click targets.
- **NSW Fair Trading Rent and Sales Report** — postcode-level 2BR median rents (December 2025 quarter).
- **Domain Listings API** — live 2BR listings when credentials are configured; otherwise mock listings that deep-link to [realestate.com.au](https://www.realestate.com.au/).

## Open source dependencies

Orbit is built on the shoulders of a lot of excellent open source work. Everything listed here is either in `backend/requirements.txt` or loaded directly by `index.html`.

**Backend (Python)**

- [FastAPI](https://github.com/tiangolo/fastapi) (MIT) — web framework
- [Uvicorn](https://github.com/encode/uvicorn) (BSD-3-Clause) — ASGI server
- [Pydantic](https://github.com/pydantic/pydantic) (MIT) — request/response validation
- [Polars](https://github.com/pola-rs/polars) (MIT) — data pipeline (Parquet, group-by, percentile ranking)
- [PyArrow](https://github.com/apache/arrow) (Apache-2.0) — Parquet I/O backend
- [pandas](https://github.com/pandas-dev/pandas) (BSD-3-Clause) — some dataframe ops in the build pipeline
- [NumPy](https://github.com/numpy/numpy) (BSD-3-Clause) — vector math in the matcher
- [openai-python](https://github.com/openai/openai-python) (Apache-2.0) — LLM client for persona/suburb copy
- [python-dotenv](https://github.com/theskumar/python-dotenv) (BSD-3-Clause) — env var loading
- [Python](https://www.python.org/) (PSF License) and [SQLite](https://www.sqlite.org/) (Public Domain) — runtime + telemetry store

**Frontend (vanilla JS)**

- [Leaflet 1.9.4](https://github.com/Leaflet/Leaflet) (BSD-2-Clause) — map rendering, custom `GridLayer` for the per-pixel heatmap
- [Space Grotesk](https://fonts.google.com/specimen/Space+Grotesk) (SIL Open Font License 1.1) — display font
- [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) (SIL Open Font License 1.1) — monospace font for data/metrics

**Infrastructure**

- [Railway](https://railway.com/) — hosting via [Nixpacks](https://github.com/railwayapp/nixpacks) (MIT).
- [CARTO basemaps](https://carto.com/basemaps/) — the "Positron" light_nolabels tile layer, rendered from OpenStreetMap data and styled by CARTO. Attribution is surfaced in the Leaflet attribution control on the map: *"© OpenStreetMap · © CARTO"*.

## AI acknowledgement

Orbit was built with substantial help from AI coding assistants — primarily Anthropic's Claude (via Claude Code and Cursor) — across the stack: the preference-weighted matcher, FastAPI cross-worker telemetry and admin kill-switch, the Leaflet per-pixel heatmap shader, the drawer UX, and large portions of the data pipeline. Product direction, the six-dimension scoring design, mean-centring and the preference-weighted reformulation, the choice of Shannon entropy and indie-to-chain ratio as derived features, and all final decisions were the author's.

Runtime AI usage is narrow and gated. OpenAI's GPT-4o-mini writes the optional persona paragraph (`POST /profile`) and per-suburb copy (`GET /suburb/{name}`) when `OPENAI_API_KEY` is set and the live admin kill-switch is enabled. Every number surfaced in the UI — the six dimension bars, the "9 parks / 43% indie / 2 beaches"-style explanations, the match scores — comes from the deterministic matcher in `matcher.py`; the LLM only paraphrases an evidence blob it is handed and the frontend falls back to template strings whenever the LLM path is disabled, unavailable, or errors out. No user-facing claim originates from the model.

## License

Built for a hackathon and not licensed for redistribution. Ask before using.
