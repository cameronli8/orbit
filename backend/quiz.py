"""
quiz.py  —  the seven-question personality quiz.

The quiz is the UX bridge between a human and the six-dimensional taste
vector that drives the suburb matching. Every answer carries pre-computed
weights that push the user's scores up or down on each dimension.

Design constraints:
 - Questions sound like lifestyle scenarios, not a property filter form.
 - Each question touches 1-3 dimensions so the final vector reflects all six.
 - All weights are between -30 and +30 on the 0-100 dimension scale, so no
   single question can dominate the overall profile.
 - The baseline user starts at 50 on every dimension (neutral). Each answer
   nudges from there.

After all 7 answers are processed, the user vector is clipped to [0, 100].

Exports:
    QUESTIONS       list of seven question dicts, each with multiple answers
    score_user(answers) -> dict of six 0-100 scores
"""

from typing import Dict, List


# ---------------------------------------------------------------------------
# The quiz
# ---------------------------------------------------------------------------
# Schema:
#  - id:        stable identifier so the frontend can submit by id
#  - text:      question as shown to the user
#  - answers:   each answer is {id, label, weights}
#                weights is a {dimension: delta} dict — deltas are integer
#                nudges from the neutral 50 baseline, range roughly ±30.

QUESTIONS: List[dict] = [
    # -------- 1. lifestyle tempo (pace + social) --------
    {
        "id": "weekend_morning",
        "text": "Your ideal Saturday morning looks like…",
        "answers": [
            {
                "id": "coastal_walk",
                "label": "A coastal walk, coffee from a local café, then back home for a slow start.",
                "weights": {"outdoor": +18, "pace": -8, "aesthetic": +5, "social_energy": -5},
            },
            {
                "id": "brunch_crowd",
                "label": "Brunch with friends somewhere busy — bonus points for a queue.",
                "weights": {"social_energy": +18, "pace": +12, "culinary": +10},
            },
            {
                "id": "market_crawl",
                "label": "Wandering a farmers' market, record store, or a gallery opening.",
                "weights": {"aesthetic": +20, "culinary": +8, "community": +5},
            },
            {
                "id": "home_chill",
                "label": "At home — book, garden, quiet streets.",
                "weights": {"pace": -20, "community": +10, "social_energy": -15},
            },
        ],
    },

    # -------- 2. nights out (social + pace + late-night) --------
    {
        "id": "nights_out",
        "text": "In a typical month, how often are you out past 10pm?",
        "answers": [
            {"id": "almost_never", "label": "Almost never — early nights suit me.",
             "weights": {"social_energy": -22, "pace": -18, "community": +8}},
            {"id": "once_twice",   "label": "Once or twice — a wine bar, a dinner.",
             "weights": {"social_energy": 0,   "pace": 0,  "culinary": +5}},
            {"id": "few_times",    "label": "A few times — gigs, bars, long dinners.",
             "weights": {"social_energy": +15, "pace": +10}},
            {"id": "weekly_plus",  "label": "Weekly or more — I need a neighbourhood that's alive at night.",
             "weights": {"social_energy": +25, "pace": +22}},
        ],
    },

    # -------- 3. the café test (aesthetic + culinary) --------
    {
        "id": "coffee_preference",
        "text": "Pick three words for your ideal coffee spot.",
        "answers": [
            {"id": "specialty_indie",
             "label": "Single-origin, mismatched chairs, a vinyl playing.",
             "weights": {"aesthetic": +22, "culinary": +12, "social_energy": +3}},
            {"id": "brunch_spot",
             "label": "Bright, busy, good eggs on the menu.",
             "weights": {"culinary": +15, "social_energy": +8, "pace": +5}},
            {"id": "reliable_chain",
             "label": "Reliable, consistent, I know what I'm getting.",
             "weights": {"aesthetic": -15, "pace": +3, "community": +3}},
            {"id": "park_kiosk",
             "label": "A park kiosk — I'd rather be outside.",
             "weights": {"outdoor": +18, "aesthetic": +3, "social_energy": -5}},
        ],
    },

    # -------- 4. movement (outdoor + community) --------
    {
        "id": "how_you_move",
        "text": "Which of these feels most like you on a weekday?",
        "answers": [
            {"id": "surf_run",        "label": "Surf, run, or outdoor sport before or after work.",
             "weights": {"outdoor": +25, "pace": +5}},
            {"id": "cycle_walk",      "label": "Cycle or walk where I can — I hate sitting in a car.",
             "weights": {"outdoor": +15, "community": +8}},
            {"id": "gym_or_studio",   "label": "A gym or studio class, indoors.",
             "weights": {"outdoor": -10, "pace": +5}},
            {"id": "mostly_sedentary","label": "Movement isn't really a priority most weeks.",
             "weights": {"outdoor": -18}},
        ],
    },

    # -------- 5. neighbours + rootedness (community) --------
    {
        "id": "neighbours",
        "text": "How much of a neighbourhood do you want around you?",
        "answers": [
            {"id": "know_everyone",
             "label": "I want to know my neighbours, the barista, the dog down the street.",
             "weights": {"community": +25, "pace": -5, "aesthetic": +3}},
            {"id": "some_community",
             "label": "A couple of regular spots, a friendly local feel.",
             "weights": {"community": +12}},
            {"id": "transient_ok",
             "label": "I don't mind transient — I like anonymous city energy.",
             "weights": {"community": -15, "pace": +10, "social_energy": +8}},
            {"id": "indifferent",
             "label": "Honestly I don't think about it much.",
             "weights": {}},
        ],
    },

    # -------- 6. the food scene (culinary) --------
    {
        "id": "food_scene",
        "text": "When you think about eating out, what matters most?",
        "answers": [
            {"id": "adventurous",
             "label": "I want a different cuisine every week.",
             "weights": {"culinary": +25, "aesthetic": +5}},
            {"id": "standout_local",
             "label": "A couple of great standout locals I return to often.",
             "weights": {"culinary": +12, "community": +8}},
            {"id": "quick_convenient",
             "label": "Quick, convenient, affordable — food is fuel.",
             "weights": {"culinary": -10, "pace": +5}},
            {"id": "mostly_home",
             "label": "I mostly cook — groceries and markets matter more.",
             "weights": {"culinary": +5, "community": +5, "social_energy": -8}},
        ],
    },

    # -------- 7. the "vibe" question (aesthetic + pace + social) --------
    {
        "id": "suburb_vibe",
        "text": "Picture the streetscape you'd want to come home to.",
        "answers": [
            {"id": "buzzy_streets",
             "label": "Buzzy streets — bars, foot traffic, things happening every night.",
             "weights": {"social_energy": +20, "pace": +18, "aesthetic": +5}},
            {"id": "leafy_family",
             "label": "Leafy, calm, kids on bikes, a primary school on the corner.",
             "weights": {"community": +20, "outdoor": +10, "pace": -15, "social_energy": -10}},
            {"id": "indie_main_st",
             "label": "An indie main street — cafés, vintage shops, a small theatre.",
             "weights": {"aesthetic": +22, "culinary": +8, "community": +5}},
            {"id": "coastal",
             "label": "A walk to the water, a café on the way home.",
             "weights": {"outdoor": +20, "aesthetic": +5, "pace": -3}},
        ],
    },
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
DIMENSIONS = ("social_energy", "aesthetic", "pace", "outdoor", "culinary", "community")

QUESTION_BY_ID = {q["id"]: q for q in QUESTIONS}


def score_user(answers: Dict[str, str]) -> Dict[str, float]:
    """Turn a set of quiz answers into a six-dimensional user vector.

    Args:
        answers: mapping of question_id -> answer_id

    Returns:
        {dimension: float in [0, 100]} for all six canonical dimensions.
    """
    vec = {d: 50.0 for d in DIMENSIONS}

    for q_id, a_id in answers.items():
        q = QUESTION_BY_ID.get(q_id)
        if q is None:
            continue
        answer = next((a for a in q["answers"] if a["id"] == a_id), None)
        if answer is None:
            continue
        for dim, delta in answer.get("weights", {}).items():
            if dim in vec:
                vec[dim] += delta

    # Clip to [0, 100]
    for d in DIMENSIONS:
        vec[d] = max(0.0, min(100.0, vec[d]))
    return vec


def quiz_payload() -> List[dict]:
    """Frontend-safe version of the quiz — drops the scoring weights so the
    answer distribution stays hidden (small UX win; prevents users
    'gaming' which answer scores highest on which dimension)."""
    return [
        {
            "id": q["id"],
            "text": q["text"],
            "answers": [{"id": a["id"], "label": a["label"]} for a in q["answers"]],
        }
        for q in QUESTIONS
    ]


if __name__ == "__main__":
    # A quick manual check — verify scoring for a few personas.
    personas = {
        "Indie inner-west": {
            "weekend_morning":   "market_crawl",
            "nights_out":        "few_times",
            "coffee_preference": "specialty_indie",
            "how_you_move":      "cycle_walk",
            "neighbours":        "some_community",
            "food_scene":        "standout_local",
            "suburb_vibe":       "indie_main_st",
        },
        "CBD night owl": {
            "weekend_morning":   "brunch_crowd",
            "nights_out":        "weekly_plus",
            "coffee_preference": "brunch_spot",
            "how_you_move":      "gym_or_studio",
            "neighbours":        "transient_ok",
            "food_scene":        "adventurous",
            "suburb_vibe":       "buzzy_streets",
        },
        "Beach outdoor": {
            "weekend_morning":   "coastal_walk",
            "nights_out":        "once_twice",
            "coffee_preference": "park_kiosk",
            "how_you_move":      "surf_run",
            "neighbours":        "some_community",
            "food_scene":        "standout_local",
            "suburb_vibe":       "coastal",
        },
        "North-shore family": {
            "weekend_morning":   "home_chill",
            "nights_out":        "almost_never",
            "coffee_preference": "reliable_chain",
            "how_you_move":      "cycle_walk",
            "neighbours":        "know_everyone",
            "food_scene":        "mostly_home",
            "suburb_vibe":       "leafy_family",
        },
    }
    for name, answers in personas.items():
        vec = score_user(answers)
        line = "  ".join(f"{d[:3]}={int(v):3d}" for d, v in vec.items())
        print(f"{name:<22} {line}")
