from config import settings
from common import PaperData

from .abstract_api_client import ResearchApiClient


class OpenAlexClient(ResearchApiClient):
    def __init__(self, timeout = 10):
        base_url = settings.SEMANTIC_SCHOLAR_API_BASE_URL
        bearer_token = settings.SEMANTIC_SCHOLAR_API_TOKEN

        super().__init__(base_url, timeout, bearer_token=bearer_token)
        
    def search_papers(self, query: str, limit: int = 10) -> list[PaperData]:
        return []