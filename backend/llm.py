"""
llm.py  —  OpenAI-powered personality profiling and suburb explanations.

Two public entry points:

    profile_user(user_vec, answers) -> dict
        Produces a short, punchy personality profile for the renter based on
        their six-dimensional taste vector plus their raw quiz answers. This
        is what the frontend shows in the "Your Orbit profile" card right
        after the quiz submits.

    explain_suburb(evidence, profile_text) -> dict
        Given the strictly-factual evidence blob from matcher.evidence_for(),
        writes tailored positive and negative explanations. The model is
        told in no uncertain terms to only cite facts present in the
        evidence — so no beaches in Chippendale.

Design rules:
    - Graceful fallback. If the API key is missing, the network is down, or
      the response parses badly, every function returns a dict marked
      `"source": "fallback"` so the caller can swap in the template text. The
      server never raises for LLM failure.
    - In-memory LRU cache. Same inputs -> same outputs within a process.
      Keeps hackathon demo costs negligible and makes drawer opens instant
      on repeat click.
    - Strict prompts. The suburb prompt gets the evidence as JSON and an
      explicit "do not invent any feature not in this JSON" instruction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("orbit.llm")


# ---------------------------------------------------------------------------
# .env loader — no hard dependency on python-dotenv. If it's installed we use
# it; otherwise we parse the .env file ourselves so the backend still runs.
# ---------------------------------------------------------------------------
def _load_env_from_dotenv() -> None:
    """Populate os.environ from the project-root .env if present. Idempotent."""
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
        return
    except ImportError:
        pass

    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception as e:
        log.warning("Failed to parse .env: %s", e)


_load_env_from_dotenv()


API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
MODEL = os.environ.get("ORBIT_LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
LLM_ENABLED = bool(API_KEY) and not API_KEY.startswith("sk-...")


# ---------------------------------------------------------------------------
# OpenAI client — lazy so importing this module never fails at startup.
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not LLM_ENABLED:
        return None
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=API_KEY)
        return _client
    except Exception as e:  # pragma: no cover
        log.warning("OpenAI client init failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _hash_payload(obj) -> str:
    """Stable hash of any JSON-serialisable object, used as a cache key."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@lru_cache(maxsize=512)
def _chat_cached(cache_key: str, system: str, user: str, json_mode: bool) -> Optional[str]:
    """Uncached underlying call. Cache key is what we actually key on — the
    system/user args are here only so the LRU retains them.

    Returns raw string content, or None on any failure.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        kwargs = dict(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.5,
            max_tokens=700,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
    except Exception as e:
        log.warning("OpenAI call failed (model=%s): %s", MODEL, e)
        return None


def _chat(system: str, user: str, json_mode: bool = False) -> Optional[str]:
    key = _hash_payload({"s": system, "u": user, "j": json_mode, "m": MODEL})
    return _chat_cached(key, system, user, json_mode)


# ---------------------------------------------------------------------------
# User profile — one call, runs once per quiz submission
# ---------------------------------------------------------------------------
_PROFILE_SYSTEM = (
    "You are Orbit, a Sydney renter-matching assistant. "
    "Given a person's six-dimensional lifestyle profile and their raw quiz "
    "answers, describe who this person is as a renter in warm, specific, "
    "second-person prose. No marketing fluff, no emojis, no headings, no "
    "lists. Avoid the phrase 'you're someone who'. Avoid naming specific "
    "Sydney suburbs — that's for another step."
)

_PROFILE_USER_TEMPLATE = """\
This user's lifestyle profile, scored 0–100 per dimension (50 = neutral):

  social_energy: {social_energy}
  aesthetic:     {aesthetic}
  pace:          {pace}
  outdoor:       {outdoor}
  culinary:      {culinary}
  community:     {community}

Their raw quiz answers (question_id -> chosen_option_id):
{answers_block}

Return a JSON object with three keys:
  "headline"  — 3-6 word persona title (title case, no quotes)
  "summary"   — 2-3 sentences in second person ("You...") capturing the
                person's core lifestyle shape. Reference 2-3 of the
                strongest dimensions by name in natural prose.
  "wants"     — an array of 3 short second-person bullet strings (each
                under 12 words) naming concrete things they'd want from
                a suburb. No generic platitudes — be specific.
"""


def _fallback_profile(user_vec: Dict[str, float]) -> Dict:
    """Template-only profile for when the LLM isn't available."""
    pairs = sorted(user_vec.items(), key=lambda kv: -kv[1])
    top = [d for d, v in pairs[:3] if v >= 55]
    low = [d for d, v in pairs[-2:] if v <= 45]
    pretty = {
        "social_energy": "social energy", "aesthetic": "indie aesthetic",
        "pace": "pace", "outdoor": "outdoor life", "culinary": "food scene",
        "community": "neighbourly community",
    }
    top_phrase = ", ".join(pretty[d] for d in top) if top else "a balanced lifestyle"
    low_phrase = ", ".join(pretty[d] for d in low) if low else ""
    summary = f"You lean into {top_phrase}."
    if low_phrase:
        summary += f" You're less drawn to {low_phrase}."
    return {
        "headline": "Your Orbit profile",
        "summary":  summary,
        "wants":    [
            f"Suburbs strong on {pretty[d]}" for d in top[:3]
        ] or ["A suburb that balances your priorities"],
        "source":   "fallback",
    }


def profile_user(user_vec: Dict[str, float], answers: Dict[str, str]) -> Dict:
    """Return {headline, summary, wants, source} for the current user."""
    if not LLM_ENABLED:
        return _fallback_profile(user_vec)

    answers_block = "\n".join(f"  {q} -> {a}" for q, a in sorted(answers.items()))
    prompt = _PROFILE_USER_TEMPLATE.format(
        answers_block=answers_block or "  (no answers)",
        **{k: round(v, 1) for k, v in user_vec.items()},
    )
    raw = _chat(_PROFILE_SYSTEM, prompt, json_mode=True)
    if not raw:
        return _fallback_profile(user_vec)

    try:
        data = json.loads(raw)
        wants = data.get("wants") or []
        if not isinstance(wants, list):
            wants = []
        return {
            "headline": str(data.get("headline") or "Your Orbit profile").strip(),
            "summary":  str(data.get("summary") or "").strip(),
            "wants":    [str(w).strip() for w in wants if str(w).strip()][:4],
            "source":   "llm",
        }
    except (json.JSONDecodeError, TypeError) as e:
        log.warning("profile_user: failed to parse LLM JSON: %s", e)
        return _fallback_profile(user_vec)


# ---------------------------------------------------------------------------
# Suburb explanations — one call per suburb-drawer open (cached by key)
# ---------------------------------------------------------------------------
_SUBURB_SYSTEM = (
    "You are Orbit, a Sydney renter-matching assistant writing drawer copy "
    "for ONE suburb at a time. You are given a JSON evidence blob and an "
    "optional persona summary for the user. RULES — non-negotiable:\n"
    "  1. You may ONLY reference specific features (beaches, parks, bars, "
    "cuisines, etc.) that appear as non-zero counts inside evidence.breakdowns "
    "or evidence.top_cuisines. If a feature is not in those fields, it does "
    "NOT exist for this suburb. Never invent beaches, parks, or venues.\n"
    "  2. Use the evidence.gaps field to decide tone: a large positive gap "
    "(suburb stronger than user wants) on social_energy/pace means 'louder "
    "than you'd probably like'; a large negative gap means 'weaker than you "
    "want'.\n"
    "  3. Address the user as 'you'. No marketing fluff. No emojis. No "
    "headings. No lists inside prose.\n"
    "  4. Every sentence must be grounded in at least one number or category "
    "name from the evidence."
)

_SUBURB_USER_TEMPLATE = """\
User persona:
{persona}

Suburb evidence (THE ONLY FACTS YOU MAY USE):
```json
{evidence_json}
```

Return a JSON object:
  "headline"  — 4-8 word summary of the fit (e.g. "A food-dense fit with a
                quieter edge"). No quotes inside the string.
  "summary"   — 2-3 sentences in second person explaining how this suburb
                relates to the user's profile. Mention at least two concrete
                breakdown categories (e.g. "12 cafés", "3 galleries") by name.
  "positive"  — array of 2-3 short second-person sentences, each grounded in
                a non-zero breakdown value the user likely cares about.
  "negative"  — array of 1-2 short second-person sentences naming the real
                trade-offs (use evidence.gaps). Empty array if nothing
                meaningful is off.
"""


def _fallback_suburb(evidence: Dict, template_positive: List[str],
                     template_negative: List[str]) -> Dict:
    """Fallback when the LLM can't be reached — surface the existing template
    strings so the drawer still has content."""
    name = evidence.get("suburb", "this suburb")
    return {
        "headline": f"{name} at a glance",
        "summary":  (template_positive[0] if template_positive
                     else f"Here's what {name} looks like based on the data."),
        "positive": template_positive,
        "negative": template_negative,
        "source":   "fallback",
    }


def explain_suburb(
    evidence: Dict,
    profile_summary: str = "",
    template_positive: Optional[List[str]] = None,
    template_negative: Optional[List[str]] = None,
) -> Dict:
    """Return {headline, summary, positive, negative, source} for one suburb.

    `template_positive` / `template_negative` come from matcher.build_explanations
    — we pass them in as the safety net if the LLM fails.
    """
    template_positive = template_positive or []
    template_negative = template_negative or []

    if not LLM_ENABLED:
        return _fallback_suburb(evidence, template_positive, template_negative)

    prompt = _SUBURB_USER_TEMPLATE.format(
        persona=profile_summary.strip() or "(no persona summary supplied — write in neutral second person)",
        evidence_json=json.dumps(evidence, indent=2, ensure_ascii=False),
    )
    raw = _chat(_SUBURB_SYSTEM, prompt, json_mode=True)
    if not raw:
        return _fallback_suburb(evidence, template_positive, template_negative)

    try:
        data = json.loads(raw)
        positive = data.get("positive") or []
        negative = data.get("negative") or []
        if not isinstance(positive, list):
            positive = []
        if not isinstance(negative, list):
            negative = []
        return {
            "headline": str(data.get("headline") or "").strip(),
            "summary":  str(data.get("summary") or "").strip(),
            "positive": [str(s).strip() for s in positive if str(s).strip()][:3],
            "negative": [str(s).strip() for s in negative if str(s).strip()][:2],
            "source":   "llm",
        }
    except (json.JSONDecodeError, TypeError) as e:
        log.warning("explain_suburb: failed to parse LLM JSON: %s", e)
        return _fallback_suburb(evidence, template_positive, template_negative)


# ---------------------------------------------------------------------------
# Introspection — used by /health so the frontend can show whether the AI
# layer is live without leaking the key.
# ---------------------------------------------------------------------------
def llm_status() -> Dict:
    return {
        "enabled": LLM_ENABLED,
        "model":   MODEL if LLM_ENABLED else None,
    }
