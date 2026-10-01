import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


class StreetViewError(Exception):
    pass


@dataclass
class Pano:
    pano_id: str
    lat: float
    lng: float
    capture_date: str | None
    status: str = "OK"


class StreetViewProvider(ABC):
    name: str

    def __init__(self) -> None:
        self.image_requests = 0

    @abstractmethod
    def find_pano(self, lat: float, lng: float, radius: float) -> Pano | None:
        """Nearest outdoor panorama within radius metres, or None if there is none."""

    @abstractmethod
    def fetch_image(self, pano_id: str, heading: float, pitch: float, fov: float, width: int, height: int) -> bytes:
        """JPEG bytes for one view of a panorama."""


class GoogleStreetViewProvider(StreetViewProvider):
    name = "google"
    METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
    IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"

    def __init__(self, api_key: str, session: requests.Session, timeout: float) -> None:
        super().__init__()
        self._api_key = api_key
        self._session = session
        self._timeout = timeout

    def find_pano(self, lat: float, lng: float, radius: float) -> Pano | None:
        params = {"location": f"{lat},{lng}", "radius": int(radius), "source": "outdoor", "key": self._api_key}
        response = self._session.get(self.METADATA_URL, params=params, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()

        if data["status"] in ("ZERO_RESULTS", "NOT_FOUND"):
            return None
        if data["status"] != "OK":
            raise StreetViewError(f"Metadata {data['status']}: {data.get('error_message', '')}")
        return Pano(data["pano_id"], data["location"]["lat"], data["location"]["lng"], data.get("date"), data["status"])

    def fetch_image(self, pano_id: str, heading: float, pitch: float, fov: float, width: int, height: int) -> bytes:
        params = {
            "pano": pano_id,
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "size": f"{width}x{height}",
            "return_error_code": "true",
            "key": self._api_key,
        }
        self.image_requests += 1
        logger.debug("Image request #%d: pano %s heading %s pitch %s", self.image_requests, pano_id, heading, pitch)
        response = self._session.get(self.IMAGE_URL, params=params, timeout=self._timeout)
        response.raise_for_status()
        return response.content
