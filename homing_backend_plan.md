# Homing — Backend Plan (solo dev, personality-first)

Rewritten to center on the one feature that actually makes this product worth building: **suburbs matched to who you are, not just what you can afford.** Everything else is a filter layered on top.

This plan is scoped for one developer over a hackathon weekend. Features that don't directly serve the personality-match story are cut, not deferred.

---

## 1. The one big idea

Every property portal in Sydney asks what you want (bedrooms, budget, postcode). Homing asks who you are and then shows you where that person lives. The user answers a short personality quiz, Homing translates their answers into a six-dimensional "taste vector," and the map colors every suburb green to red by how close its own taste vector sits to theirs.

The taste vector is the spine of the product. If you can explain what the six dimensions are and how both users and suburbs get a score on them, you've explained the whole system.

## 2. The six personality dimensions

Each dimension is chosen so it can be computed from FSQ Places data alone. This is important — it means every suburb can be profiled automatically, no hand-tagging, no third-party data for the core feature.

The first dimension is **social energy**, running from solitary to highly social. A suburb's score comes from bar density, cafe density with seating, and venues-per-capita. The second is **aesthetic**, from polished to creative-raw, derived from the chain-to-independent ratio and the share of galleries, studios, and vintage-style retail. Third is **pace**, from calm to buzzy, measured by the proportion of POIs open after 10pm plus overall POI density per square kilometre. Fourth is **outdoor orientation**, from indoor to outdoor, built from parks, beaches, trails, outdoor sports venues and waterfront access. Fifth is **culinary depth**, from convenience to foodie, scored on restaurant category diversity (Shannon entropy over cuisine tags), specialty grocery, and independent food businesses. Sixth and last is **community tightness**, from transient to rooted, read from community centres, libraries, places of worship, family services, and long-established venues as a share of total POIs.

Each dimension is normalized to a 0–100 score against the Sydney metro distribution so they're directly comparable and directly interpretable.

## 3. The quiz

Seven questions, each tagged with which dimension(s) it feeds and by how much. The quiz is the UX glue between the user and the taste vector — it has to feel like a personality quiz, not a filter form.

Questions aren't worded as property preferences. They're lifestyle scenarios: "Your ideal Saturday morning — which picture feels right?" with four images representing different pace/outdoor/social positions. "How many people in your calendar next week?" slides across transient/rooted plus social energy. "Pick three words for your ideal coffee spot" taps aesthetic and culinary depth simultaneously.

Each answer carries pre-computed weights that sum into the user's six-dim vector. The math is trivial; the craft is in making the quiz feel like it understood something about the person.

The quiz doubles as the demo hook. It's what the judge does first, and it's what they remember.

## 4. The data — deliberately minimal

**Foursquare OS Places** is the only dataset the core product depends on. Every dimension, every suburb score, every matching calculation comes from here. Sliced to `country == "AU"` and `region == "NSW"` via Polars lazy scan, it's a ~800 MB Parquet file on disk and fits entirely in memory for a hackathon.

**NSW Rent and Sales Report** is a secondary source used only as a budget filter. Suburbs above the user's budget are greyed out on the map, not excluded entirely — the user can still see they exist and are good fits, they just can't afford them. This framing is a small product choice that matters.

Everything else from the original plan — ABS Census, GTFS, crime data, Domain API, cross-city embeddings — is cut. Not deferred: cut. A solo dev adding any one of those eats a day, and the core personality-match experience doesn't need them to feel real.

The single exception is Domain listings in the drawer. Those are faked for the demo — three plausible listings generated per suburb from median rent ± noise. A judge clicking "view on Domain" can see the link would work; no live API needed.

## 5. The pipeline

One Python script does the whole thing end-to-end. `build.py` runs for about a minute on first invocation and writes a single `suburbs.parquet` file. That file is the backend.

The script scans the FSQ Sydney slice, joins each POI to a suburb polygon (using ABS SAL boundaries as a one-time GeoJSON download, but the join is done once and baked into the feature file — the backend never touches the polygons again). For each suburb it computes raw feature counts, then six dimension scores on the 0–100 metro-normalized scale. It joins NSW Rent data by normalized suburb name. The output has one row per suburb and roughly twenty columns — suburb name, six dimension scores, rent, category counts, and a few derived stats used in the "why it fits" explanations.

Rerunning `build.py` is the entire deployment process. There's no database, no migration, no ORM. Parquet files and an in-memory dataframe.

## 6. The scoring

With both the user and every suburb living in the same six-dimensional space, matching is one cosine similarity call. The API endpoint takes the user's vector, computes similarity against every row of `suburbs.parquet`, multiplies by a budget-gate factor (1.0 if in budget, smoothly decaying to 0.3 as rent exceeds budget), and returns the sorted list.

The score is the cosine, rescaled 0–100 for display. Nothing more complicated. A solo dev cannot afford a scoring engine that takes more than ten lines to explain — and in this case, ten lines is enough.

Explanations are templated off the per-dimension deltas. If the user scored high on outdoor and the suburb scored high on outdoor, the explanation string pulls the actual feature that drove that dimension: "3× Sydney's park density and direct Cooks River access." If a dimension mismatches, the neg-fact template fires: "lower nightlife density than your profile suggests." Every sentence traces back to a computed number in `suburbs.parquet`. No LLM, no hallucination, no judge asking "how did you calculate that?"

## 7. The API

Two endpoints is the whole server.

`POST /match` takes the quiz answers and the budget, returns a ranked list of suburbs with match score, rent, and three generated explanation strings each. The heatmap on the frontend reads this.

`GET /suburb/{name}` returns the full profile for the drawer — all six dimension scores with interpretation labels, the rent number, three positive and two negative explanation sentences, and three faked Domain listings. This is what the drawer opens.

FastAPI, one file, loads the Parquet into a pandas dataframe at startup. No auth, no sessions, no rate limiting, no analytics. Boring is correct here.

## 8. What's in the build order

Hackathon timelines are merciless and one person has no redundancy. This is the order of operations, from first commit to demo-ready.

The first day is data and scoring. Get the FSQ slice. Get ABS SAL boundaries and do the spatial join. Compute six dimension scores for every Sydney suburb. Join NSW Rent. Save `suburbs.parquet`. Write the scoring function and verify by hand that Dulwich Hill scores near the "creative-raw, outdoor-ish, foodie" corner and Mosman scores toward "polished, rooted, calm." If those two sanity checks fail, the dimension definitions are wrong and it's better to know on day one.

The second day is the quiz and the API. Write the seven quiz questions with pre-computed answer weights. Wire them into a user vector. Stand up the two-endpoint FastAPI. Connect it to the existing frontend — the heatmap already reads suburb scores, so this is replacing mock data with real data from `/match`.

The third day — assuming there is one — is polish. Explanation strings that feel crafted, not templated-looking. A persuasive quiz UX (images, not dropdowns). One really good example in the demo script: "I'm going to take the quiz as someone who loves Berlin's Kreuzberg. Watch where the map lights up green." That moment is the entire pitch.

If the timeline slips, the ordering means the cut line falls on polish rather than on substance. The worst acceptable demo still has real suburb scoring against real personality input.

## 9. What this plan deliberately does not have

Cross-city embeddings. Archetype clustering with k-means. Temporal FSQ snapshot diffs to detect gentrification. Live Domain API. GTFS commute isochrones. SEIFA-based demographic filters. Saved user profiles. Account login. Mobile optimization. Any kind of ML model beyond dot products and normalization. Any data pipeline that doesn't run from a single script.

Most of these were in the original plan. Most of them are good ideas. All of them are v2 — the difference between a demo that actually ships and one that's three-quarters built.

The test for any feature during the weekend is whether it makes the personality-to-place story more vivid. Budget filtering yes, because a suburb you can't afford isn't a real match. Explanation strings yes, because the "why it fits" moment is half the magic. Cross-city embeddings no, because "suburbs like Kreuzberg" is a different product — a great one, but one you build next week.

## 10. The repo

```
data/
  raw/
    fsq_sydney.parquet       # from polars_setup.py
    nsw_rent.parquet         # from NSW Fair Trading CSV
    sal_boundaries.geojson   # from ABS
  features/
    suburbs.parquet          # the only file the API reads
build.py                     # runs the whole pipeline
api.py                       # the two-endpoint FastAPI
quiz.py                      # questions + answer-to-weight tables
homing_demo.html             # existing frontend, wired to /match
```

One command to build: `python build.py`. One command to serve: `uvicorn api:app`. A judge can run this locally if they want to poke at it — and that itself is a small but real credibility signal for a data hackathon.
