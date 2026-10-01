import logging
from dataclasses import asdict, replace

import requests

from common.cache import JsonCache
from common.http import build_session
from common.record import normalize_address

from .config import Settings
from .record import GeocodeResult, now_iso

GOOGLE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"

logger = logging.getLogger(__name__)


class GeocodeError(Exception):
    def __init__(self, status: str, message: str = "") -> None:
        super().__init__(f"{status}: {message}" if message else status)
        self.status = status


class Geocoder:
    def __init__(self, settings: Settings, cache: JsonCache, session: requests.Session | None = None) -> None:
        self._settings = settings
        self._cache = cache
        self._session = session or build_session(settings.max_retries)
        self._use_google = bool(settings.google_api_key)
        if not self._use_google:
            logger.info("No Google API key set, geocoding with the US Census geocoder")

    def geocode(self, address: str, refresh: bool = False) -> GeocodeResult:
        cached = None if refresh else self._cache.get(normalize_address(address))
        if cached:
            logger.info("Cache hit for %r", address)
            return replace(GeocodeResult(**cached), input_address=address, timestamp=now_iso())

        result = self._lookup(address)
        if result.status == "ok":
            self._cache.put(normalize_address(address), asdict(result))
        return result

    def _lookup(self, address: str) -> GeocodeResult:
        if self._use_google:
            try:
                result = self._google(address)
                if result:
                    return result
                logger.info("Google found no result for %r, falling back to Census", address)
            except GeocodeError as exc:
                if exc.status == "REQUEST_DENIED":
                    # A key/billing problem affects every address, so report it once and stop asking Google.
                    logger.warning("Google Geocoding rejected the API key (%s); using Census for the rest of this run", exc)
                    self._use_google = False
                else:
                    logger.warning("Google failed for %r (%s), falling back to Census", address, exc)
            except requests.RequestException as exc:
                logger.warning("Google failed for %r (%s), falling back to Census", address, exc)

        try:
            result = self._census(address)
        except requests.RequestException as exc:
            return GeocodeResult.failure(address, f"Census request failed: {exc}", provider="census")
        return result or GeocodeResult.failure(address, "No result found", provider="census")

    def _google(self, address: str) -> GeocodeResult | None:
        response = self._session.get(
            GOOGLE_URL,
            params={"address": address, "key": self._settings.google_api_key},
            timeout=self._settings.timeout,
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] == "ZERO_RESULTS":
            return None
        if data["status"] != "OK":
            raise GeocodeError(data["status"], data.get("error_message", ""))

        top = data["results"][0]
        countries = [c["short_name"] for c in top["address_components"] if "country" in c["types"]]
        if countries != ["US"]:
            return GeocodeResult.failure(
                address, f"Result is outside the United States: {top['formatted_address']}", provider="google"
            )

        geometry = top["geometry"]
        return GeocodeResult(
            input_address=address,
            formatted_address=top["formatted_address"],
            lat=geometry["location"]["lat"],
            lng=geometry["location"]["lng"],
            provider="google",
            match_quality=geometry["location_type"],
            partial_match=top.get("partial_match", False),
        )

    def _census(self, address: str) -> GeocodeResult | None:
        matches = self._census_matches(address)
        if not matches:
            return None

        top = matches[0]
        try:
            segment = self._census_segment(top)
        except requests.RequestException as exc:
            logger.warning("Could not resolve street segment for %r: %s", address, exc)
            segment = None

        # Census is US-only and always interpolates along TIGER address ranges.
        return GeocodeResult(
            input_address=address,
            formatted_address=top["matchedAddress"],
            lat=top["coordinates"]["y"],
            lng=top["coordinates"]["x"],
            provider="census",
            match_quality="RANGE_INTERPOLATED",
            street_segment=segment,
        )

    def _census_matches(self, address: str) -> list[dict]:
        response = self._session.get(
            CENSUS_URL,
            params={"address": address, "benchmark": "Public_AR_Current", "format": "json"},
            timeout=self._settings.timeout,
        )
        response.raise_for_status()
        return response.json()["result"]["addressMatches"]

    def _census_segment(self, match: dict) -> dict | None:
        # The range's from/to numbers sit at the TIGER edge's from/to nodes, so geocoding them
        # gives the edge direction that the L/R side of street is relative to.
        line_id = match["tigerLine"]["tigerLineId"]
        street = match["matchedAddress"].split(" ", 1)[1]
        ends = {}
        for end in ("from", "to"):
            number = match["addressComponents"][f"{end}Address"]
            same_edge = [m for m in self._census_matches(f"{number} {street}") if m["tigerLine"]["tigerLineId"] == line_id]
            if not same_edge:
                return None
            coords = same_edge[0]["coordinates"]
            ends[end] = {"address_number": number, "lat": coords["y"], "lng": coords["x"]}
        return {"tiger_line_id": line_id, "side": match["tigerLine"]["side"], **ends}
