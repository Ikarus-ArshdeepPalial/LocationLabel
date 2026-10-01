import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_session(max_retries: int) -> requests.Session:
    retry = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session
