"""
Category keyword sets that define each of the six personality dimensions.

These lists are matched against Foursquare's fsq_category_labels using substring
checking (case-insensitive). A POI counts toward a dimension if any of its
category labels contains any of the dimension's keywords.

Every dimension in suburbs.parquet traces back to this file — when a judge asks
"how did you decide a suburb is high social energy?", the answer is:
    1) it has many POIs whose category labels match entries in SOCIAL_CATEGORIES
    2) normalized by suburb area (log-scale density)
    3) percentile-ranked across all NSW suburbs

This is the single source of truth for the product's taxonomy.
"""

# ---------------------------------------------------------------------------
# Social energy: bars, pubs, venues, nightlife, event spaces
# ---------------------------------------------------------------------------
SOCIAL_CATEGORIES = {
    "Bar", "Cocktail Bar", "Pub", "Beer Garden", "Wine Bar", "Beer Bar",
    "Whisky Bar", "Sports Bar", "Gastropub", "Hotel Bar", "Dive Bar",
    "Nightclub", "Dance Club", "Live Music Venue", "Karaoke Bar",
    "Comedy Club", "Music Venue", "Jazz Club", "Rock Club",
    "Event Space", "Concert Hall", "Performing Arts Venue",
    "Speakeasy", "Brewery", "Tasting Room",
}

# ---------------------------------------------------------------------------
# Aesthetic: creative / indie / cultural venues. "Aesthetic" also reads from
# the independent-to-chain ratio (computed separately from FSQ's `chains`).
# ---------------------------------------------------------------------------
AESTHETIC_CATEGORIES = {
    "Art Gallery", "Art Museum", "Museum", "Creative Space", "Design Studio",
    "Tattoo Parlor", "Piercing Parlor", "Barbershop",
    "Vintage Store", "Thrift Store", "Antique Store", "Record Store",
    "Bookstore", "Independent Bookstore", "Comic Shop",
    "Arts and Crafts Store", "Stationery Store",
    "Concert Hall", "Indie Theater", "Cinema",
    "Pottery Studio", "Arts and Entertainment",
    "Photography Lab", "Art Studio", "Art School",
}

# ---------------------------------------------------------------------------
# Outdoor orientation: parks, beaches, trails, waterfront, sports grounds
# ---------------------------------------------------------------------------
OUTDOOR_CATEGORIES = {
    "Park", "Beach", "Trail", "Surf Spot", "Playground", "Dog Park",
    "Botanical Garden", "Garden", "Nature Preserve", "Nature Reserve",
    "National Park", "State Park", "Public Garden",
    "Bike Trail", "Hiking Trail", "Mountain",
    "Bay", "River", "Lake", "Marina", "Harbor", "Pier",
    "Sports Field", "Athletic Field", "Tennis Court", "Basketball Court",
    "Soccer Field", "Skate Park", "Golf Course", "Rock Climbing",
    "Scenic Lookout", "Scenic Point", "Pool", "Swimming Pool",
    "Outdoor Recreation", "Campground",
}

# ---------------------------------------------------------------------------
# Culinary: all things food. Entropy across sub-cuisines drives the score.
# ---------------------------------------------------------------------------
CULINARY_CATEGORIES = {
    "Restaurant", "Café", "Cafe", "Coffee Shop", "Bakery",
    "Food Truck", "Specialty Food Store", "Gourmet Shop",
    "Farmers Market", "Butcher", "Seafood Market", "Fish Market",
    "Cheese Shop", "Wine Shop", "Deli", "Delicatessen",
    "Dessert Shop", "Ice Cream Shop", "Frozen Yogurt Shop",
    "Chocolatier", "Tea Room", "Juice Bar",
}

# Cuisine sub-categories used for culinary entropy (diversity of food)
CUISINE_KEYWORDS = [
    "Italian", "Chinese", "Japanese", "Korean", "Thai", "Vietnamese",
    "Indian", "Mexican", "French", "Spanish", "Greek", "Turkish",
    "Lebanese", "Middle Eastern", "Mediterranean", "American",
    "Australian", "Modern Australian", "Ethiopian", "Moroccan",
    "African", "Caribbean", "Brazilian", "Peruvian", "Argentine",
    "Sushi", "Ramen", "Dumpling", "Dim Sum", "BBQ", "Burger",
    "Pizza", "Sandwich", "Seafood", "Steakhouse", "Vegetarian",
    "Vegan", "Fusion", "Tapas", "Breakfast", "Brunch", "Bakery",
    "Malaysian", "Indonesian", "Filipino", "Singaporean",
    "Cantonese", "Sichuan", "Taiwanese",
]

# ---------------------------------------------------------------------------
# Community tightness: libraries, schools, worship, civic spaces
# ---------------------------------------------------------------------------
COMMUNITY_CATEGORIES = {
    "Library", "Community Center", "Senior Center", "Youth Center",
    "Place of Worship", "Church", "Mosque", "Synagogue", "Temple",
    "Buddhist Temple", "Religious Institution", "Spiritual Center",
    "School", "Elementary School", "Middle School", "High School",
    "Preschool", "Daycare", "Nursery School",
    "Post Office", "Town Hall", "City Hall", "Government Building",
    "Community Garden", "Scout Hut",
    "Medical Center", "Doctor's Office", "Dentist",
    "Family Service", "Charity", "Non-Profit",
}

# ---------------------------------------------------------------------------
# Pace: late-night venues that indicate "buzzy" suburbs. Combined with
# raw POI density for the full pace score.
# ---------------------------------------------------------------------------
LATE_NIGHT_CATEGORIES = {
    "Nightclub", "Dance Club", "Cocktail Bar", "Bar", "Pub",
    "Beer Garden", "Speakeasy", "Karaoke Bar",
    "Late Night Restaurant", "Diner", "24-Hour Food",
    "Live Music Venue", "Music Venue",
}

# ---------------------------------------------------------------------------
# Chains fallback list — used when FSQ's `chains` field is empty.
# Covers the 30 most common AU chain brands so the indie/chain ratio stays
# meaningful even where FSQ's chain tagging has gaps.
# ---------------------------------------------------------------------------
KNOWN_CHAIN_NAMES = {
    # Grocery & major retail
    "woolworths", "coles", "aldi", "iga", "harris farm",
    "kmart", "target", "big w", "bunnings", "chemist warehouse",
    # Fast food
    "mcdonald's", "mcdonalds", "kfc", "hungry jack's", "hungry jacks",
    "subway", "guzman y gomez", "oporto", "domino's", "dominos",
    "pizza hut", "red rooster", "nando's", "nandos", "grill'd",
    "schnitz", "mad mex", "zambrero", "chatime", "gong cha",
    # Cafes & restaurants
    "starbucks", "gloria jean's", "gloria jeans", "boost juice",
    "michel's patisserie", "muffin break", "san churro",
    # Convenience & fuel
    "7-eleven", "7 eleven", "shell", "bp", "caltex", "united",
    # Pharmacy & health
    "priceline", "chemist warehouse", "terry white", "amcal",
}


def normalize_label(label: str) -> str:
    """Strip and lowercase so matches are robust to formatting inconsistencies."""
    return label.strip().lower() if label else ""


def label_matches_any(label: str, keyword_set) -> bool:
    """Case-insensitive substring match — 'Independent Coffee Shop' matches 'Coffee Shop'."""
    lbl = normalize_label(label)
    return any(normalize_label(kw) in lbl for kw in keyword_set)


def extract_cuisine(label: str) -> str | None:
    """Given a category label, return the cuisine keyword it contains (if any)."""
    lbl = normalize_label(label)
    for cuisine in CUISINE_KEYWORDS:
        if normalize_label(cuisine) in lbl:
            return cuisine
    return None


# ---------------------------------------------------------------------------
# Breakdowns — finer-grained keyword sets so explanations can reference what's
# *actually* in a suburb. The old "X parks, beaches or outdoor venues" string
# misled users (e.g. Chippendale, which has parks but no beaches). Each
# breakdown is a subset of its parent dimension's categories.
#
# Order matters in the resulting dict — the frontend/LLM can cite the top
# non-zero entries as a list of real features.
# ---------------------------------------------------------------------------
OUTDOOR_BREAKDOWNS = {
    "beaches":       {"Beach", "Surf Spot"},
    "parks":         {"Park", "Garden", "Botanical Garden", "Public Garden",
                      "Dog Park", "Nature Preserve", "Nature Reserve"},
    "trails":        {"Trail", "Bike Trail", "Hiking Trail"},
    "waterfront":    {"Marina", "Harbor", "Pier", "Bay", "River", "Lake"},
    "sports_grounds":{"Sports Field", "Athletic Field", "Tennis Court",
                      "Basketball Court", "Soccer Field", "Golf Course",
                      "Skate Park", "Rock Climbing"},
    "pools":         {"Pool", "Swimming Pool"},
    "lookouts":      {"Scenic Lookout", "Scenic Point"},
    "playgrounds":   {"Playground"},
}

SOCIAL_BREAKDOWNS = {
    "bars":          {"Bar", "Cocktail Bar", "Wine Bar", "Beer Bar",
                      "Whisky Bar", "Dive Bar", "Speakeasy", "Hotel Bar"},
    "pubs":          {"Pub", "Beer Garden", "Gastropub", "Sports Bar"},
    "nightclubs":    {"Nightclub", "Dance Club"},
    "live_music":    {"Live Music Venue", "Music Venue", "Jazz Club",
                      "Rock Club", "Concert Hall", "Performing Arts Venue"},
    "breweries":     {"Brewery", "Tasting Room"},
    "karaoke":       {"Karaoke Bar"},
}

AESTHETIC_BREAKDOWNS = {
    "galleries":     {"Art Gallery", "Art Studio", "Creative Space"},
    "museums":       {"Museum", "Art Museum"},
    "vintage":       {"Vintage Store", "Thrift Store", "Antique Store"},
    "indie_retail":  {"Record Store", "Independent Bookstore", "Comic Shop",
                      "Arts and Crafts Store", "Stationery Store",
                      "Design Studio"},
    "cinemas":       {"Cinema", "Indie Theater"},
    "bookstores":    {"Bookstore"},
    "tattoo_piercing": {"Tattoo Parlor", "Piercing Parlor"},
}

CULINARY_BREAKDOWNS = {
    "restaurants":   {"Restaurant"},
    "cafes":         {"Café", "Cafe", "Coffee Shop"},
    "bakeries":      {"Bakery"},
    "specialty_food":{"Specialty Food Store", "Gourmet Shop", "Deli",
                      "Delicatessen", "Cheese Shop", "Wine Shop", "Butcher"},
    "markets":       {"Farmers Market", "Food Truck"},
    "desserts":      {"Dessert Shop", "Ice Cream Shop", "Frozen Yogurt Shop",
                      "Chocolatier"},
}

COMMUNITY_BREAKDOWNS = {
    "libraries":     {"Library"},
    "schools":       {"School", "Elementary School", "Middle School",
                      "High School", "Preschool", "Daycare", "Nursery School"},
    "worship":       {"Place of Worship", "Church", "Mosque", "Synagogue",
                      "Temple", "Buddhist Temple", "Religious Institution",
                      "Spiritual Center"},
    "community_centers": {"Community Center", "Senior Center", "Youth Center",
                          "Community Garden"},
    "civic":         {"Town Hall", "City Hall", "Government Building",
                      "Post Office"},
    "medical":       {"Medical Center", "Doctor's Office", "Dentist"},
}

# Master breakdown map so downstream code can iterate uniformly.
DIMENSION_BREAKDOWNS = {
    "outdoor":       OUTDOOR_BREAKDOWNS,
    "social_energy": SOCIAL_BREAKDOWNS,
    "aesthetic":     AESTHETIC_BREAKDOWNS,
    "culinary":      CULINARY_BREAKDOWNS,
    "community":     COMMUNITY_BREAKDOWNS,
}


def count_breakdowns(
    labels: list[str],
    breakdowns: dict,
) -> dict[str, int]:
    """Count how many labels match each sub-category in a breakdown map.
    Labels can match multiple buckets (substring-based matching by design).
    Returns a dict with every breakdown key, even zeros, so downstream
    consumers don't need to worry about missing keys.
    """
    out: dict[str, int] = {k: 0 for k in breakdowns}
    for lbl in labels:
        for key, kws in breakdowns.items():
            if label_matches_any(lbl, kws):
                out[key] += 1
    return out
