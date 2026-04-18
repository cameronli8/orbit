"""
Mock Foursquare OS Places generator.

Produces a Parquet file with the exact schema Foursquare's real dataset uses,
populated with plausible Sydney POIs distributed across 45 suburbs. Used for
development when HF auth isn't available and for reproducible demos — the
scoring pipeline treats mock and real data identically.

Each suburb's `character` field (from suburbs_ref.py) determines its POI mix:
    - inner_west_creative  → indie cafes, galleries, vintage, record stores
    - inner_east_bougie    → wine bars, fine dining, designer boutiques
    - cbd_buzz             → dense everything, heavy on bars & nightlife
    - beach_outdoor        → beaches, surf shops, casual food
    - north_shore_family   → parks, schools, community, low nightlife
    - western_commuter     → chains, groceries, family services
    - cultural_hub         → diverse restaurants (Vietnamese, Korean, Lebanese)
    - student_buzz         → cheap eats, cafes, libraries, books

Output schema mirrors Foursquare's real columns, so switching to real data
is a one-line change in build.py.
"""

import random
import math
from pathlib import Path

import polars as pl

from suburbs_ref import SYDNEY_SUBURBS


# ---------------------------------------------------------------------------
# Per-character POI templates
# ---------------------------------------------------------------------------
# Each entry is (fsq_category_label, weight, is_chain_bias)
# Where is_chain_bias is probability this POI is from a chain (0-1).

CATEGORY_TEMPLATES = {
    "inner_west_creative": [
        # Food & drink (indie heavy)
        ("Food > Cafe > Independent Coffee Shop", 12, 0.05),
        ("Food > Restaurant > Modern Australian", 5, 0.10),
        ("Food > Restaurant > Vietnamese Restaurant", 4, 0.05),
        ("Food > Restaurant > Italian Restaurant", 4, 0.15),
        ("Food > Restaurant > Pizza Place", 3, 0.25),
        ("Food > Restaurant > Thai Restaurant", 3, 0.10),
        ("Food > Restaurant > Japanese Restaurant", 3, 0.15),
        ("Food > Restaurant > Korean Restaurant", 2, 0.05),
        ("Food > Restaurant > Vegan and Vegetarian Restaurant", 3, 0.05),
        ("Food > Bakery", 4, 0.20),
        ("Food > Brewery", 2, 0.10),
        ("Nightlife > Bar > Cocktail Bar", 3, 0.05),
        ("Nightlife > Bar > Wine Bar", 3, 0.05),
        ("Nightlife > Bar > Pub", 2, 0.20),
        ("Nightlife > Bar > Beer Bar", 2, 0.10),
        ("Nightlife > Live Music Venue", 2, 0.10),
        # Aesthetic / creative
        ("Arts and Entertainment > Art Gallery", 3, 0.02),
        ("Arts and Entertainment > Museum", 1, 0.05),
        ("Arts and Entertainment > Cinema", 1, 0.30),
        ("Retail > Bookstore > Independent Bookstore", 2, 0.05),
        ("Retail > Vintage Store", 3, 0.05),
        ("Retail > Record Store", 2, 0.05),
        ("Retail > Tattoo Parlor", 2, 0.05),
        ("Retail > Arts and Crafts Store", 1, 0.15),
        # Outdoor
        ("Outdoor Recreation > Park", 2, 0.0),
        ("Outdoor Recreation > Dog Park", 1, 0.0),
        ("Outdoor Recreation > Bike Trail", 1, 0.0),
        # Community
        ("Community > Library", 1, 0.0),
        ("Community > Community Center", 1, 0.0),
        # Retail (some chains)
        ("Retail > Grocery Store", 2, 0.85),
        ("Retail > Pharmacy", 2, 0.75),
        ("Retail > Fashion Retail", 3, 0.35),
    ],
    "inner_east_bougie": [
        ("Food > Cafe > Coffee Shop", 10, 0.25),
        ("Food > Restaurant > Modern Australian", 6, 0.15),
        ("Food > Restaurant > French Restaurant", 3, 0.10),
        ("Food > Restaurant > Italian Restaurant", 5, 0.20),
        ("Food > Restaurant > Japanese Restaurant", 4, 0.15),
        ("Food > Restaurant > Steakhouse", 2, 0.20),
        ("Food > Restaurant > Seafood Restaurant", 2, 0.15),
        ("Food > Restaurant > Tapas Restaurant", 2, 0.10),
        ("Food > Bakery", 3, 0.25),
        ("Food > Gourmet Shop", 2, 0.10),
        ("Nightlife > Bar > Cocktail Bar", 4, 0.10),
        ("Nightlife > Bar > Wine Bar", 5, 0.10),
        ("Nightlife > Bar > Whisky Bar", 2, 0.10),
        ("Arts and Entertainment > Art Gallery", 2, 0.05),
        ("Retail > Fashion Retail > Boutique", 6, 0.20),
        ("Retail > Jewelry Store", 2, 0.30),
        ("Retail > Cosmetics Shop", 2, 0.60),
        ("Outdoor Recreation > Park", 2, 0.0),
        ("Community > Library", 1, 0.0),
        ("Retail > Grocery Store", 2, 0.80),
        ("Retail > Pharmacy", 2, 0.75),
    ],
    "cbd_buzz": [
        ("Food > Cafe > Coffee Shop", 18, 0.45),
        ("Food > Restaurant > Modern Australian", 8, 0.20),
        ("Food > Restaurant > Japanese Restaurant", 6, 0.15),
        ("Food > Restaurant > Chinese Restaurant", 6, 0.15),
        ("Food > Restaurant > Italian Restaurant", 5, 0.20),
        ("Food > Restaurant > Thai Restaurant", 4, 0.15),
        ("Food > Restaurant > Korean Restaurant", 3, 0.10),
        ("Food > Restaurant > Vietnamese Restaurant", 3, 0.10),
        ("Food > Restaurant > Sushi Restaurant", 3, 0.15),
        ("Food > Restaurant > Burger Joint", 4, 0.60),
        ("Food > Restaurant > Sandwich Place", 5, 0.50),
        ("Food > Bakery", 3, 0.35),
        ("Nightlife > Bar > Cocktail Bar", 8, 0.10),
        ("Nightlife > Bar > Pub", 6, 0.20),
        ("Nightlife > Bar > Wine Bar", 4, 0.08),
        ("Nightlife > Bar > Sports Bar", 3, 0.20),
        ("Nightlife > Nightclub", 4, 0.10),
        ("Nightlife > Live Music Venue", 3, 0.05),
        ("Nightlife > Bar > Dive Bar", 2, 0.05),
        ("Arts and Entertainment > Concert Hall", 1, 0.0),
        ("Arts and Entertainment > Cinema", 2, 0.35),
        ("Arts and Entertainment > Museum", 2, 0.0),
        ("Retail > Fashion Retail", 10, 0.50),
        ("Retail > Department Store", 2, 0.90),
        ("Community > Library", 1, 0.0),
        ("Outdoor Recreation > Park", 2, 0.0),
        ("Retail > Grocery Store", 3, 0.85),
        ("Retail > Pharmacy", 3, 0.80),
    ],
    "beach_outdoor": [
        ("Outdoor Recreation > Beach", 2, 0.0),
        ("Outdoor Recreation > Park", 3, 0.0),
        ("Outdoor Recreation > Surf Spot", 2, 0.0),
        ("Outdoor Recreation > Trail", 2, 0.0),
        ("Outdoor Recreation > Bike Trail", 1, 0.0),
        ("Outdoor Recreation > Swimming Pool", 1, 0.0),
        ("Outdoor Recreation > Scenic Lookout", 1, 0.0),
        ("Outdoor Recreation > Tennis Court", 1, 0.0),
        ("Retail > Sporting Goods Shop > Surf Shop", 2, 0.20),
        ("Retail > Sporting Goods Shop", 2, 0.40),
        ("Food > Cafe > Coffee Shop", 10, 0.25),
        ("Food > Restaurant > Seafood Restaurant", 4, 0.10),
        ("Food > Restaurant > Modern Australian", 4, 0.15),
        ("Food > Restaurant > Burger Joint", 3, 0.50),
        ("Food > Restaurant > Pizza Place", 3, 0.30),
        ("Food > Restaurant > Thai Restaurant", 2, 0.15),
        ("Food > Restaurant > Italian Restaurant", 2, 0.20),
        ("Food > Bakery", 3, 0.30),
        ("Food > Juice Bar", 2, 0.30),
        ("Nightlife > Bar > Pub", 3, 0.25),
        ("Nightlife > Bar > Cocktail Bar", 2, 0.10),
        ("Community > Library", 1, 0.0),
        ("Community > Community Center", 1, 0.0),
        ("Retail > Grocery Store", 2, 0.85),
        ("Retail > Pharmacy", 2, 0.80),
    ],
    "north_shore_family": [
        ("Outdoor Recreation > Park", 4, 0.0),
        ("Outdoor Recreation > Playground", 2, 0.0),
        ("Outdoor Recreation > Tennis Court", 2, 0.0),
        ("Outdoor Recreation > Bike Trail", 1, 0.0),
        ("Community > School", 4, 0.0),
        ("Community > Preschool", 2, 0.0),
        ("Community > Library", 2, 0.0),
        ("Community > Community Center", 1, 0.0),
        ("Community > Place of Worship > Church", 3, 0.0),
        ("Community > Medical Center", 3, 0.05),
        ("Community > Dentist", 2, 0.15),
        ("Food > Cafe > Coffee Shop", 6, 0.35),
        ("Food > Restaurant > Modern Australian", 3, 0.20),
        ("Food > Restaurant > Italian Restaurant", 3, 0.25),
        ("Food > Restaurant > Chinese Restaurant", 2, 0.25),
        ("Food > Restaurant > Thai Restaurant", 2, 0.20),
        ("Food > Bakery", 2, 0.35),
        ("Nightlife > Bar > Pub", 1, 0.40),
        ("Nightlife > Bar > Wine Bar", 1, 0.15),
        ("Retail > Grocery Store", 3, 0.90),
        ("Retail > Pharmacy", 2, 0.80),
        ("Retail > Fashion Retail", 2, 0.50),
    ],
    "western_commuter": [
        ("Retail > Grocery Store", 6, 0.90),
        ("Retail > Pharmacy", 3, 0.85),
        ("Retail > Department Store", 2, 0.95),
        ("Retail > Fashion Retail", 4, 0.70),
        ("Food > Restaurant > Fast Food Restaurant", 6, 0.85),
        ("Food > Restaurant > Burger Joint", 3, 0.75),
        ("Food > Restaurant > Pizza Place", 3, 0.60),
        ("Food > Restaurant > Chinese Restaurant", 3, 0.30),
        ("Food > Restaurant > Vietnamese Restaurant", 2, 0.10),
        ("Food > Restaurant > Thai Restaurant", 2, 0.20),
        ("Food > Cafe > Coffee Shop", 5, 0.55),
        ("Food > Bakery", 2, 0.55),
        ("Community > School", 4, 0.0),
        ("Community > Preschool", 2, 0.0),
        ("Community > Library", 1, 0.0),
        ("Community > Place of Worship > Church", 2, 0.0),
        ("Community > Medical Center", 3, 0.10),
        ("Outdoor Recreation > Park", 2, 0.0),
        ("Outdoor Recreation > Playground", 2, 0.0),
        ("Nightlife > Bar > Pub", 1, 0.50),
    ],
    "cultural_hub": [
        ("Food > Restaurant > Vietnamese Restaurant", 6, 0.05),
        ("Food > Restaurant > Chinese Restaurant", 5, 0.10),
        ("Food > Restaurant > Korean Restaurant", 4, 0.05),
        ("Food > Restaurant > Thai Restaurant", 4, 0.08),
        ("Food > Restaurant > Japanese Restaurant", 3, 0.12),
        ("Food > Restaurant > Sushi Restaurant", 3, 0.12),
        ("Food > Restaurant > Malaysian Restaurant", 2, 0.05),
        ("Food > Restaurant > Indonesian Restaurant", 2, 0.05),
        ("Food > Restaurant > Filipino Restaurant", 2, 0.05),
        ("Food > Restaurant > Lebanese Restaurant", 3, 0.05),
        ("Food > Restaurant > Middle Eastern Restaurant", 2, 0.10),
        ("Food > Restaurant > Indian Restaurant", 3, 0.10),
        ("Food > Restaurant > Dumpling Restaurant", 2, 0.05),
        ("Food > Restaurant > Ramen Restaurant", 1, 0.05),
        ("Food > Restaurant > Dim Sum Restaurant", 2, 0.05),
        ("Food > Cafe > Coffee Shop", 6, 0.40),
        ("Food > Bakery", 3, 0.25),
        ("Food > Specialty Food Store", 3, 0.10),
        ("Food > Farmers Market", 1, 0.0),
        ("Nightlife > Bar > Pub", 2, 0.40),
        ("Nightlife > Bar > Karaoke Bar", 1, 0.10),
        ("Community > Library", 1, 0.0),
        ("Community > Place of Worship > Buddhist Temple", 1, 0.0),
        ("Community > Place of Worship > Church", 2, 0.0),
        ("Community > Place of Worship > Mosque", 1, 0.0),
        ("Community > School", 3, 0.0),
        ("Community > Community Center", 2, 0.0),
        ("Retail > Grocery Store", 3, 0.70),
        ("Retail > Pharmacy", 2, 0.75),
        ("Outdoor Recreation > Park", 2, 0.0),
    ],
    "student_buzz": [
        ("Food > Cafe > Independent Coffee Shop", 10, 0.10),
        ("Food > Restaurant > Vietnamese Restaurant", 4, 0.05),
        ("Food > Restaurant > Thai Restaurant", 4, 0.15),
        ("Food > Restaurant > Burger Joint", 4, 0.55),
        ("Food > Restaurant > Pizza Place", 3, 0.40),
        ("Food > Restaurant > Dumpling Restaurant", 2, 0.05),
        ("Food > Restaurant > Ramen Restaurant", 2, 0.10),
        ("Food > Restaurant > Sandwich Place", 3, 0.50),
        ("Food > Bakery", 2, 0.30),
        ("Nightlife > Bar > Pub", 4, 0.25),
        ("Nightlife > Bar > Beer Bar", 2, 0.10),
        ("Nightlife > Bar > Dive Bar", 2, 0.05),
        ("Nightlife > Live Music Venue", 2, 0.05),
        ("Retail > Bookstore > Independent Bookstore", 2, 0.05),
        ("Retail > Vintage Store", 2, 0.05),
        ("Community > Library", 2, 0.0),
        ("Community > School > University", 1, 0.0),
        ("Outdoor Recreation > Park", 2, 0.0),
        ("Retail > Grocery Store", 2, 0.80),
        ("Retail > Pharmacy", 2, 0.75),
    ],
}

# Chain names to draw from when a POI is flagged as a chain
CHAIN_BRAND_POOL = {
    "coffee":  ["Starbucks", "Gloria Jean's", "Michel's Patisserie", "Boost Juice"],
    "fast":    ["McDonald's", "KFC", "Hungry Jack's", "Subway", "Guzman y Gomez",
                "Oporto", "Nando's", "Grill'd", "Red Rooster", "Zambrero", "Mad Mex"],
    "pizza":   ["Domino's", "Pizza Hut", "Crust Pizza"],
    "grocery": ["Woolworths", "Coles", "IGA", "Aldi", "Harris Farm Markets"],
    "pharma":  ["Priceline Pharmacy", "Chemist Warehouse", "Terry White Chemmart", "Amcal"],
    "retail":  ["Kmart", "Target", "Big W", "JB Hi-Fi", "Myer", "David Jones",
                "Cotton On", "Uniqlo", "H&M"],
    "dept":    ["Myer", "David Jones", "Westfield"],
    "convenience": ["7-Eleven", "Shell", "BP", "Caltex"],
}

# Indie-style name fragments for non-chain POIs
INDIE_ADJECTIVES = ["Little", "Old", "Wild", "Blue", "Green", "Red", "The", "Golden",
                    "Black", "White", "Silver", "Single O", "Two Birds", "Five Points",
                    "North", "South", "East", "West", "Daily", "Humble", "Common"]
INDIE_NOUNS = ["Bean", "Bird", "Fox", "Wolf", "Crow", "Canteen", "Kitchen",
               "Table", "Plate", "Spoon", "Fork", "Grain", "Leaf", "Root",
               "Grounds", "Roasters", "Bakery", "Larder", "Counter", "Deli",
               "Workshop", "Collective", "Social"]


def pick_chain_brand(category: str) -> str:
    """Choose a plausible chain brand given the POI category."""
    cat = category.lower()
    if "coffee" in cat or "cafe" in cat:   return random.choice(CHAIN_BRAND_POOL["coffee"])
    if "grocery" in cat:                   return random.choice(CHAIN_BRAND_POOL["grocery"])
    if "pharmacy" in cat:                  return random.choice(CHAIN_BRAND_POOL["pharma"])
    if "fast food" in cat:                 return random.choice(CHAIN_BRAND_POOL["fast"])
    if "burger" in cat:                    return random.choice(["McDonald's", "Hungry Jack's", "Grill'd"])
    if "pizza" in cat:                     return random.choice(CHAIN_BRAND_POOL["pizza"])
    if "department store" in cat:          return random.choice(CHAIN_BRAND_POOL["dept"])
    if "convenience" in cat or "gas" in cat: return random.choice(CHAIN_BRAND_POOL["convenience"])
    if "fashion" in cat or "retail" in cat: return random.choice(CHAIN_BRAND_POOL["retail"])
    if "cinema" in cat:                    return random.choice(["Event Cinemas", "Hoyts", "Village"])
    return random.choice(CHAIN_BRAND_POOL["retail"])


def make_indie_name(category: str) -> str:
    """Generate a plausible independent business name."""
    return f"{random.choice(INDIE_ADJECTIVES)} {random.choice(INDIE_NOUNS)}"


def generate_pois_for_suburb(suburb, density_multiplier=1.0):
    """Generate POIs for a single suburb based on its character."""
    character = suburb["character"]
    templates = CATEGORY_TEMPLATES.get(character, CATEGORY_TEMPLATES["inner_west_creative"])
    pois = []
    pid = 0

    # POI count scales with suburb area (more area → more places, up to a cap)
    # and with character (CBD is denser than outer areas)
    base_density = {
        "cbd_buzz":           220,
        "inner_west_creative": 160,
        "inner_east_bougie":  180,
        "beach_outdoor":      130,
        "north_shore_family":  95,
        "western_commuter":   120,
        "cultural_hub":       150,
        "student_buzz":       170,
    }.get(character, 130)

    poi_count = int(base_density * min(3.5, math.sqrt(suburb["area_km2"])) * density_multiplier)

    # Weighted random sample of templates
    weights = [t[1] for t in templates]

    for _ in range(poi_count):
        template = random.choices(templates, weights=weights)[0]
        category_label, _, chain_bias = template

        is_chain = random.random() < chain_bias
        if is_chain:
            name = pick_chain_brand(category_label)
            chain_info = [{"chain_id": f"chain_{hash(name) & 0xFFFF}", "chain_name": name}]
        else:
            name = make_indie_name(category_label)
            chain_info = []

        # Scatter coordinates within a plausible suburb radius (~0.6-1.2 km)
        # Use gaussian with sigma proportional to sqrt(area)
        sigma = 0.004 * math.sqrt(suburb["area_km2"])
        lat_offset = random.gauss(0, sigma)
        lng_offset = random.gauss(0, sigma)

        pid += 1
        pois.append({
            "fsq_place_id":       f"mock_{suburb['name'].replace(' ','_')}_{pid:04d}",
            "name":               name,
            "latitude":           suburb["lat"] + lat_offset,
            "longitude":          suburb["lng"] + lng_offset,
            "address":            f"{random.randint(1, 400)} {random.choice(['King', 'Queen', 'High', 'Main', 'Oxford', 'George'])} St",
            "locality":           suburb["name"],
            "region":             "NSW",
            "country":            "AU",
            "postcode":           str(2000 + random.randint(0, 700)),
            "fsq_category_labels": [category_label],
            "fsq_category_ids":   [f"cat_{hash(category_label) & 0xFFFF}"],
            "chains":             chain_info,
            "date_created":       "2024-01-01",
            "date_closed":        None,
        })

    return pois


def build_mock_dataset(output_path: Path, seed: int = 42):
    """Generate the full mock FSQ dataset and save as Parquet."""
    random.seed(seed)

    all_pois = []
    for suburb in SYDNEY_SUBURBS:
        pois = generate_pois_for_suburb(suburb)
        all_pois.extend(pois)

    df = pl.DataFrame(all_pois)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    print(f"  Generated {len(all_pois):,} POIs across {len(SYDNEY_SUBURBS)} suburbs")
    print(f"  Saved to {output_path}")
    print(f"  POIs per suburb (top 5): {dict(sorted(df.group_by('locality').len().iter_rows(), key=lambda x: -x[1])[:5])}")
    return df


if __name__ == "__main__":
    out = Path(__file__).parent.parent / "data" / "raw" / "fsq_sydney.parquet"
    build_mock_dataset(out)
