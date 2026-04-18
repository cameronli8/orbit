# Homing — Technical Requirements

Everything you need to go from empty folder to working demo. Assumes Python 3.11+, a laptop with ~3 GB free disk, and a decent internet connection for the one-time data pulls.

---

## 1. Environment

Python 3.11 is the floor. 3.12 is fine. Polars's streaming engine gets meaningfully faster on 3.11+ and a couple of the libraries we use drop 3.10 soon.

Work in a virtual environment so your system Python stays clean:

```
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

Install the full dependency set in one go. None of these are heavy by hackathon standards:

```
pip install polars pyarrow huggingface_hub \
            geopandas shapely \
            fastapi "uvicorn[standard]" \
            pydantic numpy scipy \
            pandas
```

Polars handles the FSQ slice and per-suburb aggregation. Geopandas plus Shapely does the one-time spatial join from POI points to suburb polygons (we bake the result into a column and never touch the polygons again at serving time, so geopandas doesn't need to be in the API process). FastAPI plus Uvicorn is the web layer. NumPy and SciPy cover cosine similarity and normalization. Pandas is kept for the API because FastAPI loves dataframes — the final Parquet is small enough that Pandas is the simpler choice server-side.

Sanity check the install:

```
python -c "import polars, geopandas, fastapi, numpy; print('ok')"
```

---

## 2. Data acquisition

Three sources. Two are one-time downloads, one is a per-build pull that's fast after first cache.

**Foursquare OS Places — Sydney slice.** Already scripted in `polars_setup.py`. Change the config to `TARGET_COUNTRY = "AU"`, `TARGET_REGION = "NSW"`, `TARGET_LOCALITY = None` so we get all of Greater Sydney plus some surrounds (the suburb polygon join will discard anything outside Sydney). Running the script lazy-scans Hugging Face, pulls matching row groups, and writes `nsw_places.parquet`. Expect 3–5 minutes over a home connection and a file around 600–900 MB. You need a Hugging Face account and a read token — set it with `huggingface-cli login` once. The dataset path template in the script (`release/dt=2026-04-14/places/parquet/*.parquet`) should be updated to whatever the latest release is on the dataset page; Foursquare ships roughly monthly.

**ABS Statistical Areas — suburb polygons.** The ABS publishes "Suburbs and Localities" (SAL) boundaries as a GeoJSON in the ASGS 2021 release. Download the NSW-filtered GeoJSON from the ABS website under ASGS Edition 3 → SAL. It's around 40 MB. Save it to `data/raw/sal_nsw.geojson`. This file never changes for a hackathon — download once, commit it if you want.

**NSW Rent and Sales Report.** NSW Fair Trading publishes quarterly. The relevant sheet is "Median weekly rents by suburb." Find the latest quarterly Excel file on the NSW Communities & Justice website, save the rent-by-suburb sheet as a CSV, and drop it in `data/raw/nsw_rent.csv`. Only columns you need are suburb name, bedroom count, and median weekly rent. A small one-time cleanup script strips footnotes and normalizes suburb names to uppercase stripped of extra whitespace.

Put all three files under `data/raw/` and never touch them again after download. Everything downstream reads from there and writes to `data/features/`.

---

## 3. The build pipeline — `build.py`

One script, top to bottom, idempotent. Reruns overwrite `data/features/suburbs.parquet`. The script is five logical blocks.

The **first block** loads the three raw sources. FSQ via Polars, SAL via geopandas (cast to the same CRS as FSQ coordinates — WGS84, EPSG:4326), rent CSV via Polars. Normalize suburb names in the rent data to match the SAL naming convention (ABS uses title case, e.g., "Dulwich Hill" — the NSW rent report uses uppercase; a `.str.title()` fixes it).

The **second block** does the point-in-polygon spatial join. The cleanest way is to convert the FSQ dataframe to a geopandas GeoDataFrame, call `gpd.sjoin(places, suburbs, how="inner", predicate="within")`, and get back a dataframe with a new `suburb_name` column. For ~100k NSW POIs this runs in under ten seconds. If geopandas feels slow, the fallback is shapely's STRtree manually — but you shouldn't need it.

The **third block** computes per-suburb category counts. Explode `fsq_category_labels` (it's a list column), group by suburb and category, pivot wide so you get one column per category. Keep only top-level categories (FSQ labels are slash-delimited: `"Food > Cafe > Independent Coffee Shop"` — split on ` > ` and take the second element for the grouping level we want). You end up with a suburb × ~120 matrix of counts.

The **fourth block** computes the six dimension scores. This is the single piece of code that defines the product. Each dimension is a formula over category counts plus a final normalization against the Sydney-wide distribution.

For **social energy**: sum counts in categories like Bar, Cocktail Bar, Nightclub, Beer Garden, Cafe, Restaurant, Event Space; divide by suburb area in square kilometres (from the SAL geometry); take the log to tame the long tail; then min-max scale across all NSW suburbs to 0–100.

For **aesthetic**: compute `indie_ratio = independent_count / (independent_count + chain_count)` where you define chain membership by FSQ's `chains` field or by a hand-curated list of the 40 most common chain names (Woolworths, Coles, McDonald's, KFC, etc. — ten minutes of effort, catches 95% of the signal). Multiply by the share of galleries, studios, tattoo parlors, vintage stores. Normalize.

For **pace**: without reliable opening-hours data, proxy with category signals — fraction of POIs in "late" categories (Bar, Nightclub, Late-night Food, 24-hour venues) plus overall POI density. A buzzier suburb has more things generally. Normalize.

For **outdoor orientation**: count Park, Beach, Trail, Surf Spot, Bike Shop, Outdoor Sports Venue. Divide by total POIs (so it's a *share* of character, not an absolute count — Centennial Park being huge shouldn't drown out its surroundings). Normalize.

For **culinary depth**: Shannon entropy over the restaurant sub-categories (Italian, Vietnamese, Korean BBQ, Sichuan, Ethiopian, etc.) — higher entropy means more diverse food scene. Add independent food businesses as a share of total food. Normalize.

For **community tightness**: sum Library, Community Center, Place of Worship, School, Family Service, Medical Center, Senior Services. Divide by total POIs. Normalize.

Save the result with one row per suburb and columns `suburb_name, social, aesthetic, pace, outdoor, culinary, community, rent_2br, poi_total, area_km2`, plus a raw `category_counts` JSON column if you want the drawer to show "3× metro-average indie cafes" style facts (compute those at query time from this column, not in build).

The **fifth block** writes `data/features/suburbs.parquet` with `df.write_parquet()`. File should be well under a megabyte. Done.

A full run of `build.py` targets sixty to ninety seconds end-to-end after first data pull.

---

## 4. The quiz — `quiz.py`

A pure Python module. Two data structures and one function.

The **first structure** is `QUESTIONS` — a list of seven question dicts. Each has an `id`, a `prompt`, a list of `answers`, and each answer carries a `weights` dict that adds to the user's six-dimension vector. Weights are small integers (−2 to +2) per dimension; the final user vector is the elementwise sum across all answered questions, then normalized to 0–100 per dimension so it lives in the same space as suburb scores.

Example of how a weights dict looks conceptually: the answer "Saturday morning I want to wake up and walk straight to the beach" carries `{outdoor: +2, pace: -1, social: 0}` — it pushes strongly outdoor, slightly calm, neutral on social. Tuning these weights is hand-craft; expect an hour of iteration after you see how the output maps to suburbs you know.

The **second structure** is `CHAIN_LIST` — the forty most common chain brand names, lowercased, used by `build.py` for the indie/chain ratio. It lives here because the quiz and the indie dimension are the two places chain membership matters conceptually.

The **one function** is `score_user(answers: dict[str, str]) -> dict[str, float]` that takes answer choices (by id) and returns the six-dimension user vector.

---

## 5. The API — `api.py`

Single file. Imports `suburbs.parquet` into a pandas dataframe at startup. Exposes two endpoints.

`POST /match` takes a body of `{answers: {...}, budget: int}`. Calls `quiz.score_user(answers)` to get the user vector. For each suburb, computes cosine similarity between the user vector and the suburb's six-dimension vector — NumPy does this as a single matrix multiply across all 700 rows, sub-millisecond. Applies the budget gate: multiply score by `1.0` if `rent_2br <= budget`, smoothly decay to `0.3` over the range `budget → 1.5*budget`. Sorts by final score, returns top N with score, rent, dimension profile, and three templated explanation strings each.

`GET /suburb/{name}` returns the full profile. Six dimension scores with interpretation labels ("high social energy" when score > 70, "quiet" when < 30, "balanced" otherwise), three positive explanation strings, two negative ones, and three faked listings (just median rent ± small offsets, with plausible dummy addresses). The response shape should match exactly what the frontend drawer expects — keep them isomorphic to avoid remapping code.

Explanation templates live in a constant at the top of `api.py`. They look like `"{multiple}× Sydney-average cafe density"` where `multiple` is filled from the ratio of this suburb's cafe count to the metro median. Templates are chosen by which dimension has the biggest positive or negative delta versus the user vector, pulled from a small lookup table.

CORS needs to be open for the frontend to call it from a file:// or different-port origin. One middleware line:

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

Fine for a demo, obviously not for production.

Serve with `uvicorn api:app --reload --port 8000`.

---

## 6. Frontend wiring

`homing_demo.html` currently uses hardcoded `suburbs` array. Two surgical edits replace it with live data from the API.

The quiz UI doesn't exist yet in the frontend. For the demo you have two options. The quick option is reusing the existing sliders as the quiz — each slider becomes one dimension input, directly writing to the user vector. The real option is building a step-by-step quiz modal that runs before the map loads, collects answers, posts to `/match`, and populates the map with the response. The real option is better for the pitch but costs most of a day; the quick option is acceptable if time is tight.

On map load, call `fetch('http://localhost:8000/match', ...)` with the collected answers, receive the ranked suburb list, replace the `suburbs` array's `fit` values with the returned scores, and call `fitLayer.redraw()`. The heatmap rerenders instantly.

On suburb click, call `GET /suburb/{name}` and use the response body to populate the drawer — dimension profile replaces the current "archetype" tag, the returned explanation arrays become the fact list, the listings array populates the grid.

---

## 7. The tricky bits

Four things are likely to burn time. Worth thinking about them before you hit them.

**Suburb name normalization.** FSQ's `locality` field is user-entered by venue owners and is wildly inconsistent — you'll see "Dulwich Hill," "dulwich hill," "Dulwich hill", and "Dulwich Hill NSW." Always join on the SAL polygon match rather than the locality string. Use locality only as a tiebreaker for edge-case points near suburb boundaries.

**Category normalization.** FSQ has a rich hierarchical taxonomy but labels can appear at different depths. A place might be tagged just `"Food"` or `"Food > Restaurant > Korean > Korean BBQ"`. Pick a target depth (usually level 2) and truncate; anything at shallower depth gets dropped or bucketed into "Other." Don't try to make the taxonomy perfect — a 10% of points landing in "Other" is fine.

**Dimension score calibration.** Your first pass at the six dimensions will feel wrong. Dulwich Hill might score medium on aesthetic when you expect high; Mosman might score higher on outdoor than you expect because of the harbour. The fix is iterative — print the top five suburbs by each dimension, eyeball them, adjust the formula. Budget two hours for this; it's the single highest-leverage polish pass in the whole project.

**Cosine similarity and zero-variance.** If a user gives answers that land at exactly (50, 50, 50, 50, 50, 50), cosine similarity becomes undefined because the vector has no direction. Handle by either forcing at least one nonzero dimension (e.g., subtract 50 first so the vector is centered) or by using a small epsilon. Centered cosine similarity is cleaner — every dimension becomes −50 to +50, the math works, and the matches are more differentiated.

---

## 8. Testing without writing tests

Hackathon scope doesn't justify unit tests, but two ten-minute validation passes save hours.

First, a **sanity eyeball** on dimension scores. Print `df.nlargest(5, 'outdoor')` and `df.nlargest(5, 'aesthetic')` and so on for each dimension. If Manly, Bondi, and Coogee don't dominate "outdoor," your formula has a bug. If Surry Hills, Newtown, and Redfern don't dominate "aesthetic," same.

Second, a **round-trip smoke test** on matching. Hardcode a user vector that represents a caricature — say, "pure Kreuzberg": high aesthetic, high culinary, high social, medium pace, low outdoor, low community. Hit `/match` with that vector and check that Marrickville, Newtown, and Surry Hills sit at the top. If they don't, either the dimension formulas are wrong or the matching math is wrong. Swap vectors to represent "pure Mosman" — low aesthetic, medium culinary, low social, low pace, high outdoor, high community — and check that North Shore suburbs dominate. Two user profiles, sanity-checked against two mental models. If both pass, the system is demo-ready.

---

## 9. Serving the demo

Uvicorn runs the API on localhost:8000. The frontend is currently a single HTML file — open it directly in Chrome and it'll hit the API via the CORS-enabled fetch. No need for a separate static server.

If you want one-command startup, a tiny `Makefile`:

```
setup:
	pip install -r requirements.txt

build:
	python build.py

serve:
	uvicorn api:app --reload --port 8000

demo: build
	uvicorn api:app --reload --port 8000 &
	open homing_demo.html
```

Freeze your installed packages to `requirements.txt` with `pip freeze > requirements.txt` so a judge can reproduce the environment in one command.

---

## 10. What you'll be glad you did

If there's time, record a one-minute screen capture of the demo — quiz answers flowing in, the map shifting from mostly-grey to green-pooled, a drawer opening on the top match, a second quiz profile flipping the map to a different corner of Sydney. Judging often happens in rooms too loud for live narration; a short video is a backup plan and a social-share asset in one.

Two items worth having on a single sheet of paper for the pitch: the six dimensions with a one-line definition each, and two example quiz profiles with the suburbs they return. Everything the product does reduces to those, and being able to point at them when asked "what's your moat" changes the tone of the conversation.
