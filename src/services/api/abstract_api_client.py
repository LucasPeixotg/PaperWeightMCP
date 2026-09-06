
from abc import ABC, abstractmethod

import httpx
from common import PaperData

# from pydantic import BaseModel, ConfigDict
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# class UserProfile(BaseModel):
#     model_config = ConfigDict(extra="ignore")
#     id: int
#     username: str
#     email: str

class APIError(Exception):
    """Base exception for consumer errors."""

class ResearchApiClient(ABC):
    def __init__(self, base_url: str, api_token: str = "", timeout: float = 10.0):
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
        } if api_token else {}

        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._client.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),  # Fixed
        reraise=True,
    )
    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        try:
            response = self._client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as err:
            raise APIError(f"API status {err.response.status_code}: {err.response.text}") from err
        except httpx.RequestError as err:
            raise APIError(f"Network failure: {err}") from err

    @abstractmethod
    def search_papers(self, query: str) -> list[PaperData]:
        pass