from common import PaperData

from config import settings

from .abstract_api_client import ResearchApiClient


class SemanticScholarClient(ResearchApiClient):
    def __init__(self, timeout = 10):
        base_url = settings.SEMANTIC_SCHOLAR_API_BASE_URL
        api_token = settings.SEMANTIC_SCHOLAR_API_TOKEN

        super().__init__(base_url, api_token, timeout)
        
    def search_papers(self, query: str) -> list[PaperData]:
        return []