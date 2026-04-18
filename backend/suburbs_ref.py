"""
Sydney suburbs reference data.

For the hackathon demo we ship a curated list of 45 Sydney suburbs spanning
inner-west, eastern beaches, north shore, western Sydney, and a handful of
south and southwest areas. Each entry has:
    - name           (canonical spelling matching NSW Fair Trading)
    - lat, lng       (centroid)
    - area_km2       (approximate "walkable / lived-in" area — closer to ABS
                      SAL residential figures than to the full official
                      polygon. POIs are attributed via the polygon in
                      data/suburbs.geojson, but density is computed against
                      this residential approximation because the dimension
                      scoring is tuned against it: dividing by the full
                      polygon area would crush suburbs like Mascot or
                      Manly whose polygons include airport/headland/water.)
    - rent_2br       (median 2BR weekly rent, NSW Rent and Sales Report,
                      December 2025 quarter, postcode-level Total×2BR)
    - character      (a short archetype hint used for mock POI generation)
"""

# character: describes the POI mix this suburb should have (for mock data)
#   inner_west_creative : indie cafes, galleries, vintage, low chain
#   inner_east_bougie   : high-end restaurants, wine bars, boutique retail
#   cbd_buzz            : dense everything, high nightlife
#   beach_outdoor       : beaches, surf, outdoor sports, casual food
#   north_shore_family  : community, schools, parks, low nightlife
#   western_commuter    : chains, family services, community anchors
#   cultural_hub        : diverse restaurants, community venues
#   student_buzz        : budget food, cafes, libraries, cheap bars

SYDNEY_SUBURBS = [
    # Inner-west creative belt
    {"name": "Dulwich Hill",  "lat": -33.906, "lng": 151.140, "area_km2": 2.45, "rent_2br": 705, "character": "inner_west_creative"},
    {"name": "Marrickville",  "lat": -33.912, "lng": 151.156, "area_km2": 4.32, "rent_2br": 795, "character": "inner_west_creative"},
    {"name": "Newtown",       "lat": -33.896, "lng": 151.180, "area_km2": 1.84, "rent_2br": 860, "character": "inner_west_creative"},
    {"name": "Enmore",        "lat": -33.901, "lng": 151.173, "area_km2": 0.86, "rent_2br": 860, "character": "inner_west_creative"},
    {"name": "Petersham",     "lat": -33.895, "lng": 151.155, "area_km2": 1.30, "rent_2br": 780, "character": "inner_west_creative"},
    {"name": "Stanmore",      "lat": -33.893, "lng": 151.163, "area_km2": 1.08, "rent_2br": 720, "character": "inner_west_creative"},
    {"name": "Leichhardt",    "lat": -33.884, "lng": 151.156, "area_km2": 2.10, "rent_2br": 850, "character": "inner_west_creative"},
    {"name": "Erskineville",  "lat": -33.899, "lng": 151.187, "area_km2": 1.05, "rent_2br": 1050, "character": "inner_west_creative"},
    {"name": "Camperdown",    "lat": -33.888, "lng": 151.175, "area_km2": 1.22, "rent_2br": 940, "character": "student_buzz"},
    {"name": "Chippendale",   "lat": -33.886, "lng": 151.199, "area_km2": 0.48, "rent_2br": 1100, "character": "student_buzz"},
    # Inner-east bougie
    {"name": "Surry Hills",   "lat": -33.884, "lng": 151.213, "area_km2": 1.55, "rent_2br": 1085, "character": "inner_east_bougie"},
    {"name": "Paddington",    "lat": -33.884, "lng": 151.230, "area_km2": 1.50, "rent_2br": 1100, "character": "inner_east_bougie"},
    {"name": "Darlinghurst",  "lat": -33.878, "lng": 151.218, "area_km2": 1.16, "rent_2br": 1085, "character": "cbd_buzz"},
    # Potts Point and Kings Cross share an identical polygon in
    # data/suburbs.geojson; KX additionally absorbs Woolloomooloo /
    # Elizabeth Bay / Rushcutters Bay via aliasing in fetch_osm_pois.py.
    # Those absorbed chunks are real nightlife ground truth, so we bump
    # KX's area to roughly the merged footprint (~1.4 km²) to keep its
    # density scores honest.
    {"name": "Potts Point",   "lat": -33.871, "lng": 151.223, "area_km2": 0.56, "rent_2br": 1150, "character": "cbd_buzz"},
    {"name": "Kings Cross",   "lat": -33.873, "lng": 151.223, "area_km2": 1.38, "rent_2br": 1150, "character": "cbd_buzz"},
    {"name": "Redfern",       "lat": -33.893, "lng": 151.204, "area_km2": 1.30, "rent_2br": 950, "character": "inner_east_bougie"},
    {"name": "Waterloo",      "lat": -33.901, "lng": 151.206, "area_km2": 2.00, "rent_2br": 1100, "character": "inner_east_bougie"},
    {"name": "Alexandria",    "lat": -33.909, "lng": 151.195, "area_km2": 3.58, "rent_2br": 928, "character": "inner_east_bougie"},
    {"name": "Glebe",         "lat": -33.880, "lng": 151.185, "area_km2": 1.62, "rent_2br": 1000, "character": "inner_west_creative"},
    # CBD
    {"name": "Sydney CBD",    "lat": -33.868, "lng": 151.209, "area_km2": 2.80, "rent_2br": 1300, "character": "cbd_buzz"},
    {"name": "Pyrmont",       "lat": -33.870, "lng": 151.195, "area_km2": 0.94, "rent_2br": 1000, "character": "cbd_buzz"},
    # Eastern beaches
    {"name": "Bondi",         "lat": -33.892, "lng": 151.275, "area_km2": 1.30, "rent_2br": 1175, "character": "beach_outdoor"},
    {"name": "Bondi Junction","lat": -33.893, "lng": 151.251, "area_km2": 1.64, "rent_2br": 1200, "character": "inner_east_bougie"},
    {"name": "Bronte",        "lat": -33.903, "lng": 151.264, "area_km2": 1.00, "rent_2br": 1100, "character": "beach_outdoor"},
    {"name": "Coogee",        "lat": -33.920, "lng": 151.259, "area_km2": 1.88, "rent_2br": 1025, "character": "beach_outdoor"},
    {"name": "Randwick",      "lat": -33.914, "lng": 151.241, "area_km2": 3.75, "rent_2br": 935, "character": "beach_outdoor"},
    {"name": "Maroubra",      "lat": -33.949, "lng": 151.247, "area_km2": 3.52, "rent_2br": 900, "character": "beach_outdoor"},
    # Northern beaches
    {"name": "Manly",         "lat": -33.797, "lng": 151.285, "area_km2": 1.54, "rent_2br": 1160, "character": "beach_outdoor"},
    {"name": "Dee Why",       "lat": -33.754, "lng": 151.290, "area_km2": 3.00, "rent_2br": 823, "character": "beach_outdoor"},
    # North Shore
    {"name": "Mosman",        "lat": -33.828, "lng": 151.243, "area_km2": 8.64, "rent_2br": 850, "character": "north_shore_family"},
    {"name": "Neutral Bay",   "lat": -33.832, "lng": 151.218, "area_km2": 1.74, "rent_2br": 890, "character": "north_shore_family"},
    {"name": "Cremorne",      "lat": -33.827, "lng": 151.229, "area_km2": 2.11, "rent_2br": 900, "character": "north_shore_family"},
    {"name": "North Sydney",  "lat": -33.840, "lng": 151.207, "area_km2": 1.52, "rent_2br": 1000, "character": "cbd_buzz"},
    {"name": "Chatswood",     "lat": -33.797, "lng": 151.183, "area_km2": 4.39, "rent_2br": 930, "character": "cultural_hub"},
    {"name": "Lane Cove",     "lat": -33.814, "lng": 151.169, "area_km2": 2.08, "rent_2br": 760, "character": "north_shore_family"},
    # Cultural / diverse
    {"name": "Ashfield",      "lat": -33.888, "lng": 151.125, "area_km2": 3.06, "rent_2br": 700, "character": "cultural_hub"},
    {"name": "Burwood",       "lat": -33.877, "lng": 151.104, "area_km2": 2.74, "rent_2br": 880, "character": "cultural_hub"},
    {"name": "Strathfield",   "lat": -33.880, "lng": 151.085, "area_km2": 2.38, "rent_2br": 740, "character": "cultural_hub"},
    {"name": "Cabramatta",    "lat": -33.894, "lng": 150.936, "area_km2": 3.32, "rent_2br": 480, "character": "cultural_hub"},
    # Inner-south transit
    {"name": "Mascot",        "lat": -33.931, "lng": 151.189, "area_km2": 3.12, "rent_2br": 1030, "character": "western_commuter"},
    {"name": "Tempe",         "lat": -33.926, "lng": 151.161, "area_km2": 1.28, "rent_2br": 850, "character": "inner_west_creative"},
    # Western commuter
    {"name": "Parramatta",    "lat": -33.815, "lng": 151.003, "area_km2": 4.13, "rent_2br": 690, "character": "western_commuter"},
    {"name": "Penrith",       "lat": -33.751, "lng": 150.694, "area_km2": 7.23, "rent_2br": 550, "character": "western_commuter"},
    {"name": "Blacktown",     "lat": -33.768, "lng": 150.906, "area_km2": 6.70, "rent_2br": 550, "character": "western_commuter"},
    # Southern
    {"name": "Hurstville",    "lat": -33.967, "lng": 151.103, "area_km2": 3.43, "rent_2br": 750, "character": "cultural_hub"},
]

# Index by name for quick lookup
SUBURB_INDEX = {s["name"]: s for s in SYDNEY_SUBURBS}

# Fast list of just the names
SUBURB_NAMES = [s["name"] for s in SYDNEY_SUBURBS]
