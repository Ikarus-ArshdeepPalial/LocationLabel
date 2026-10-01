import logging
from collections import Counter

from common.cache import JsonCache
from common.http import build_session
from common.progress import track_addresses

from .config import load_settings
from .provider import GoogleStreetViewProvider
from .streetview import StreetViewStage

logger = logging.getLogger(__name__)


def run(addresses: list[str], dry_run: bool = False, force: bool = False, refine_with_google: bool = False) -> Counter[str]:
    settings = load_settings()
    if not settings.google_api_key:
        logger.error("GOOGLE_MAPS_API_KEY is not set (see .env.example)")
        return Counter({"error": len(addresses)})

    session = build_session(settings.max_retries)
    provider = GoogleStreetViewProvider(settings.google_api_key, session, settings.timeout)
    cache = JsonCache(settings.cache_path)
    try:
        stage = StreetViewStage(settings, provider, cache, session, dry_run, force, refine_with_google)
        statuses = Counter(stage.run(address) for address in track_addresses(addresses, "[2/4] streetview"))
    finally:
        cache.close()
    logger.info("Street View: %d image requests this run", provider.image_requests)
    return statuses
