"""
real_listings.py  —  Domain API client for live Sydney rentals.

Thin, resilient wrapper around Domain's public Listings API. The rest of
the backend never calls Domain directly — it calls `fetch_listings()`,
which either returns live data or raises `DomainUnavailable`. `matcher.py`
catches that and falls back to the mock generator, so the demo never dies
because of a network hiccup.

What's here:

    get_access_token()
        OAuth2 client-credentials exchange with
        https://auth.domain.com.au/v1/connect/token.
        Cached in-process with its expiry so we don't burn requests on
        tokens.

    fetch_listings(suburb, beds=2, max_price=None, n=3)
        POSTs to /v1/listings/residential/_search with a `Rent` filter on
        the requested suburb, maps the response to the exact schema the
        frontend already consumes (address / rent_pw / beds / baths /
        parking / property_type / headline / description / image / lat /
        lng / url), and writes the mapped listings to a 24h disk cache.

    DOMAIN_ENABLED
        `True` iff `DOMAIN_CLIENT_ID` and `DOMAIN_CLIENT_SECRET` are set
        in the env (loaded from the project-root .env via llm._load_env_from_dotenv,
        which has already run by the time this module is imported the
        same way as llm.py).

Environment:

    DOMAIN_CLIENT_ID        OAuth client id from developer.domain.com.au
    DOMAIN_CLIENT_SECRET    OAuth client secret
    DOMAIN_SCOPE            Override the scope list (default:
                            api_listings_read). Rarely needed.
    ORBIT_LISTINGS_CACHE_H  Override cache TTL in hours (default 24).

Design rules:
    - Never raise out of this module unless the caller explicitly asks
      for uncached data and Domain is unreachable. The normal path returns
      a (possibly empty) list of listings.
    - Cache on disk, keyed by `(suburb_lowered, beds, max_price_bucket)`.
      A stale-cache-is-better-than-nothing policy keeps demo day sane.
    - The on-disk cache survives restarts so judges hitting /match for
      the same suburbs never cost us a call.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Re-use llm.py's .env loader so every env var we need is already in os.environ
# by the time this module's top-level runs. llm is imported before us
# (api.py imports llm first), but we still call the loader defensively.
try:
    from llm import _load_env_from_dotenv  # type: ignore
    _load_env_from_dotenv()
except Exception:
    pass


log = logging.getLogger("orbit.real_listings")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLIENT_ID     = os.environ.get("DOMAIN_CLIENT_ID", "").strip()
CLIENT_SECRET = os.environ.get("DOMAIN_CLIENT_SECRET", "").strip()
SCOPE         = os.environ.get("DOMAIN_SCOPE", "api_listings_read").strip()
CACHE_H       = float(os.environ.get("ORBIT_LISTINGS_CACHE_H", "24"))

DOMAIN_ENABLED = bool(CLIENT_ID and CLIENT_SECRET)

AUTH_URL = "https://auth.domain.com.au/v1/connect/token"
API_BASE = "https://api.domain.com.au"
SEARCH_PATH = "/v1/listings/residential/_search"

# Disk cache: backend/../data/listings_cache.json — lives next to suburbs.parquet.
CACHE_PATH = Path(__file__).parent.parent / "data" / "listings_cache.json"


class DomainUnavailable(Exception):
    """Raised when the Domain path is not usable and the caller explicitly
    asked for a real fetch (e.g. a CLI smoke test). The normal code path
    catches any exception and falls back silently."""


# ---------------------------------------------------------------------------
# Disk cache — simple {cache_key: {ts, listings}} JSON blob
# ---------------------------------------------------------------------------
_cache_mem: Optional[Dict[str, dict]] = None


def _cache_load() -> Dict[str, dict]:
    global _cache_mem
    if _cache_mem is not None:
        return _cache_mem
    if not CACHE_PATH.exists():
        _cache_mem = {}
        return _cache_mem
    try:
        _cache_mem = json.loads(CACHE_PATH.read_text())
    except Exception as e:
        log.warning("listings_cache.json unreadable (%s) — starting fresh", e)
        _cache_mem = {}
    return _cache_mem


def _cache_save() -> None:
    if _cache_mem is None:
        return
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(_cache_mem, indent=2))
    except Exception as e:
        log.warning("Could not persist listings cache: %s", e)


def _cache_key(suburb: str, beds: int, max_price: Optional[int]) -> str:
    # Bucket max_price so +/- $50 doesn't balloon the cache.
    bucket = None if max_price is None else (int(max_price) // 50) * 50
    return f"{suburb.lower().strip()}|{beds}|{bucket}"


def _cache_get(suburb: str, beds: int, max_price: Optional[int]) -> Optional[List[dict]]:
    c = _cache_load()
    entry = c.get(_cache_key(suburb, beds, max_price))
    if not entry:
        return None
    age_h = (time.time() - float(entry.get("ts", 0))) / 3600.0
    if age_h > CACHE_H:
        return None
    return list(entry.get("listings") or [])


def _cache_put(suburb: str, beds: int, max_price: Optional[int], listings: List[dict]) -> None:
    c = _cache_load()
    c[_cache_key(suburb, beds, max_price)] = {
        "ts": time.time(),
        "listings": listings,
    }
    _cache_save()


# ---------------------------------------------------------------------------
# OAuth — token cache in-process
# ---------------------------------------------------------------------------
_token_cache: Dict[str, object] = {"token": None, "expires_at": 0.0}


def get_access_token(force: bool = False) -> str:
    """Fetch (and cache) a Domain access token. Raises DomainUnavailable
    if credentials are missing or the auth call fails."""
    if not DOMAIN_ENABLED:
        raise DomainUnavailable("DOMAIN_CLIENT_ID / DOMAIN_CLIENT_SECRET not set")

    now = time.time()
    if (not force) and _token_cache["token"] and _token_cache["expires_at"] > now + 30:
        return str(_token_cache["token"])

    try:
        r = requests.post(
            AUTH_URL,
            data={
                "client_id":     CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "grant_type":    "client_credentials",
                "scope":         SCOPE,
            },
            timeout=6,
        )
    except requests.RequestException as e:
        raise DomainUnavailable(f"Auth network error: {e}") from e

    if r.status_code != 200:
        raise DomainUnavailable(f"Auth status {r.status_code}: {r.text[:200]}")

    payload = r.json()
    token = payload.get("access_token")
    if not token:
        raise DomainUnavailable(f"Auth returned no access_token: {payload}")
    _token_cache["token"] = token
    _token_cache["expires_at"] = now + float(payload.get("expires_in", 3600))
    return str(token)


# ---------------------------------------------------------------------------
# Response mapping — Domain listing -> Orbit listing schema
# ---------------------------------------------------------------------------
def _pick_image(media: List[dict]) -> Optional[str]:
    """Pick the best photo url from a Domain media array. Prefers 'Photo'
    types with the largest available image."""
    if not media:
        return None
    photos = [m for m in media if (m.get("type") or "").lower() == "photo" and m.get("url")]
    if not photos:
        photos = [m for m in media if m.get("url")]
    if not photos:
        return None
    # Domain typically exposes sized variants in the url or via ?w=... — we
    # just take the first. Some payloads include `media[i].images[j].url`
    # with explicit sizes; handle both shapes.
    first = photos[0]
    if "images" in first and isinstance(first["images"], list) and first["images"]:
        # Largest by assumed ordering — Domain returns descending sizes.
        url = first["images"][0].get("url") or first.get("url")
    else:
        url = first.get("url")
    return url


def _map_listing(item: dict, suburb_name: str) -> Optional[dict]:
    """Map a single Domain search-result item to Orbit's listing dict.
    Returns None if the item is missing fields we consider essential
    (price, address, coords)."""
    # The _search response wraps each hit as
    #   {"type": "PropertyListing", "listing": {...}}
    # but some tiers flatten it. Handle both.
    listing = item.get("listing") if "listing" in item else item
    if not isinstance(listing, dict):
        return None

    pd  = listing.get("propertyDetails") or {}
    prc = listing.get("priceDetails")    or {}

    address = pd.get("displayableAddress") or pd.get("address")
    if not address:
        return None

    # Prefer the structured `rentPerWeek` if it's there; fall back to the
    # price-from bound or parse $ from displayPrice.
    rent_pw: Optional[float] = None
    for key in ("rentPerWeek", "priceFrom", "price"):
        v = prc.get(key)
        if isinstance(v, (int, float)) and v > 0:
            rent_pw = float(v)
            break
    if rent_pw is None:
        dp = prc.get("displayPrice") or ""
        import re
        m = re.search(r"\$?\s*([0-9,]{2,6})", dp)
        if m:
            try:
                rent_pw = float(m.group(1).replace(",", ""))
            except Exception:
                rent_pw = None
    if not rent_pw:
        return None

    lat = pd.get("latitude")
    lng = pd.get("longitude")
    try:
        lat = float(lat) if lat is not None else None
        lng = float(lng) if lng is not None else None
    except (TypeError, ValueError):
        lat = lng = None

    beds  = int(pd.get("bedrooms")   or 0)
    baths = int(pd.get("bathrooms")  or 0)
    parking = pd.get("carspaces")
    parking = int(parking) if parking is not None else None
    # Domain uses CamelCase codes like "ApartmentUnitFlat" — normalise to
    # human-readable labels. Unknown codes fall back to a title-cased split.
    _PTYPE_MAP = {
        "ApartmentUnitFlat": "Apartment",
        "SemiDetached":      "Semi-Detached",
        "DuplexSemiDetached": "Duplex",
        "Townhouse":         "Townhouse",
        "House":             "House",
        "Studio":            "Studio",
        "Terrace":           "Terrace",
        "BlockOfUnits":      "Block of Units",
    }
    raw_pt = pd.get("propertyType") or (pd.get("propertyTypes") or [None])[0]
    if raw_pt and str(raw_pt) in _PTYPE_MAP:
        ptype = _PTYPE_MAP[str(raw_pt)]
    elif raw_pt:
        # Split CamelCase → "Apartment Unit Flat"
        import re as _re
        ptype = _re.sub(r"(?<!^)(?=[A-Z])", " ", str(raw_pt)).strip()
    else:
        ptype = "Property"

    headline = listing.get("headline") or f"{beds}BR {ptype} in {suburb_name}"
    summary = listing.get("summaryDescription") or ""

    # Detail page URL. Prefer the slug, fall back to /rent/<id>.
    slug = listing.get("listingSlug")
    lid  = listing.get("id")
    if slug:
        url = f"https://www.domain.com.au/{slug}"
    elif lid:
        url = f"https://www.domain.com.au/rent/{lid}/"
    else:
        url = "https://www.domain.com.au/rent/"

    return {
        "id":            str(lid or f"domain-{suburb_name.lower()}-{hash(address) & 0xffff}"),
        "address":       address,
        "rent_pw":       int(round(rent_pw)),
        "beds":          beds,
        "baths":         baths,
        "parking":       parking,
        "property_type": ptype,
        "headline":      headline,
        "description":   summary,
        "image":         _pick_image(listing.get("media") or []),
        "lat":           lat,
        "lng":           lng,
        "url":           url,
        "source":        "domain",
    }


# ---------------------------------------------------------------------------
# Public: fetch listings
# ---------------------------------------------------------------------------
def fetch_listings(
    suburb: str,
    beds: int = 2,
    max_price: Optional[float] = None,
    n: int = 3,
    use_cache: bool = True,
) -> List[dict]:
    """Return up to `n` live Domain rental listings for a suburb.

    Fails *open*: if the Domain path isn't configured, the call fails, or
    no listings come back, returns an empty list. Caller (matcher.py) is
    expected to fall back to mock listings in that case.
    """
    max_price_int = int(max_price) if max_price else None

    # Cache hit?
    if use_cache:
        hit = _cache_get(suburb, beds, max_price_int)
        if hit is not None:
            return hit[:n]

    if not DOMAIN_ENABLED:
        return []

    try:
        token = get_access_token()
    except DomainUnavailable as e:
        log.warning("Domain auth unavailable — %s", e)
        return []

    # Search body. Domain accepts multiple locations, but we scope to one
    # suburb-at-a-time so cache keys stay simple.
    body = {
        "listingType":  "Rent",
        "propertyTypes": [
            "House", "ApartmentUnitFlat", "Townhouse", "SemiDetached",
            "Studio", "Terrace", "DuplexSemiDetached",
        ],
        "minBedrooms":  max(1, int(beds) - 1),
        "maxBedrooms":  int(beds) + 1,
        "pageSize":     max(6, int(n) * 2),
        "pageNumber":   1,
        "locations": [{
            "state":   "NSW",
            "region":  "",
            "area":    "",
            "suburb":  suburb,
            "postCode": "",
            "includeSurroundingSuburbs": False,
        }],
    }
    if max_price_int:
        body["maxPrice"] = int(max_price_int)

    try:
        r = requests.post(
            API_BASE + SEARCH_PATH,
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=8,
        )
    except requests.RequestException as e:
        log.warning("Domain network error for %s: %s", suburb, e)
        return []

    if r.status_code == 401:
        # Token expired between cache hit and call — one retry.
        try:
            token = get_access_token(force=True)
            r = requests.post(
                API_BASE + SEARCH_PATH,
                json=body,
                headers={"Authorization": f"Bearer {token}"},
                timeout=8,
            )
        except (requests.RequestException, DomainUnavailable) as e:
            log.warning("Domain retry failed for %s: %s", suburb, e)
            return []

    if r.status_code != 200:
        log.warning("Domain /_search %d for %s: %s", r.status_code, suburb, r.text[:300])
        return []

    try:
        payload = r.json()
    except ValueError:
        log.warning("Domain returned non-JSON for %s", suburb)
        return []

    items = payload if isinstance(payload, list) else payload.get("results") or []
    mapped: List[dict] = []
    for item in items:
        m = _map_listing(item, suburb)
        if m:
            mapped.append(m)
        if len(mapped) >= n:
            break

    # Cache even an empty result — saves us from hammering Domain for
    # suburbs that never return anything (Cabramatta with an unrealistic
    # max_price for example).
    _cache_put(suburb, beds, max_price_int, mapped)
    return mapped


# ---------------------------------------------------------------------------
# CLI smoke: `python3 real_listings.py Bondi`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sub = sys.argv[1] if len(sys.argv) > 1 else "Bondi"
    beds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    print(f"Domain enabled: {DOMAIN_ENABLED}")
    out = fetch_listings(sub, beds=beds, n=3, use_cache=False)
    if not out:
        print(f"No live listings for {sub} — will fall back to mocks in matcher.py")
    else:
        print(f"Got {len(out)} listings:")
        for l in out:
            print(f"  - ${l['rent_pw']}/wk  {l['beds']}BR {l['property_type']}  {l['address']}")
            print(f"    {l['url']}")
