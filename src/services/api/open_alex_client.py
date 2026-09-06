import logging

from common import PaperData
from config import settings

from .abstract_api_client import APIError, ResearchApiClient

logger = logging.getLogger(__name__)

class OpenAlexClient(ResearchApiClient):
    # Fields requested via `select` — the OpenAlex counterpart of Semantic Scholar's
    # `fields`. Anything not listed here is left out of the response.
    SEARCH_FIELDS = (
        "id,doi,display_name,publication_year,"
        "abstract_inverted_index,best_oa_location,primary_location"
    )

    # The API rejects a per-page outside 1..200 with a "Pagination error". Note that
    # OpenAlex's own LLM quick reference claims a maximum of 100; that is stale — 200
    # is what the endpoint actually serves, and what its rejection message names.
    MAX_LIMIT = 200

    def __init__(self, timeout = 10):
        base_url = settings.OPENALEX_API_BASE_URL
        bearer_token = settings.OPENALEX_API_TOKEN

        super().__init__(base_url, timeout=timeout, bearer_token=bearer_token)

    def search_papers(self, query: str, limit: int = 10) -> list[PaperData]:
        # Unlike Semantic Scholar, a blank search is a 200 here — it just matches every
        # work in OpenAlex. That makes the request worse than useless, so skip it.
        if not query.strip():
            return []

        try:
            payload = self._request(
                "GET",
                "/works",
                params={
                    "search": query,
                    "per-page": max(1, min(limit, self.MAX_LIMIT)),
                    "select": self.SEARCH_FIELDS,
                },
            )
        except APIError as error:
            logger.error(error)
            return []

        return [self._to_paper_data(item) for item in payload.get("results") or []]

    @staticmethod
    def _to_paper_data(item: dict) -> PaperData:
        # Every field is nullable while PaperData requires all six, so each value is
        # coerced to its empty equivalent.
        best = item.get("best_oa_location") or {}
        primary = item.get("primary_location") or {}

        # A PDF the caller can actually read beats a landing page, and the open-access
        # location beats the publisher's. Whichever wins also supplies the license.
        url, license_ = "", ""
        for location in (best, primary):
            url = location.get("pdf_url") or location.get("landing_page_url") or ""
            if url:
                license_ = location.get("license") or ""
                break
        else:
            url = item.get("doi") or ""

        # `id` is a URL ("https://openalex.org/W2626778328"); callers want the bare id.
        openalex_id = item.get("id") or ""

        return PaperData(
            paper_id=openalex_id.rsplit("/", 1)[-1],
            title=item.get("display_name") or "",
            year=item.get("publication_year") or 0,
            abstract=OpenAlexClient._abstract_from_inverted_index(
                item.get("abstract_inverted_index")
            ),
            url=url,
            license=license_,
        )

    @staticmethod
    def _abstract_from_inverted_index(index: dict | None) -> str:
        """Rebuild the plain-text abstract from OpenAlex's inverted index.

        Abstracts are only ever published as ``{"word": [positions...]}`` — there is no
        plain `abstract` field, and `select=abstract` is rejected outright — so this
        reconstruction is mandatory rather than an optimisation.
        """
        if not index:
            return ""

        positions = [position for slots in index.values() for position in slots]
        if not positions:
            return ""

        # A word can occupy several positions, and the index may skip some entirely;
        # the gaps are dropped rather than left as blanks.
        words = [""] * (max(positions) + 1)
        for word, slots in index.items():
            for position in slots:
                words[position] = word

        return " ".join(word for word in words if word)
