import logging

from common import PaperData
from config import settings

from .api_client import ApiClient, APIError

logger = logging.getLogger(__name__)

class OpenAlexClient(ApiClient):
    # Fields requested via `select` — the OpenAlex counterpart of Semantic Scholar's
    # `fields`. Anything not listed here is left out of the response, `relevance_score`
    # included: unselected it is simply absent, with no error, which would leave every
    # paper at the default 0.0 and silently disable the reranker's tiebreak.
    SEARCH_FIELDS = (
        "id,doi,display_name,publication_year,"
        "abstract_inverted_index,best_oa_location,primary_location,relevance_score"
    )

    # The API rejects a per-page outside 1..200 with a "Pagination error". Note that
    # OpenAlex's own LLM quick reference claims a maximum of 100; that is stale — 200
    # is what the endpoint actually serves, and what its rejection message names.
    MAX_LIMIT = 200

    # Semantic search runs against a vector index rather than the inverted one, and caps
    # far lower than the keyword endpoint's 200.
    SEMANTIC_MAX_LIMIT = 50

    # OpenAlex publishes DOIs in resolver-URL form; the bare identifier is what the
    # other APIs take back, so the prefix comes off in `_bare_doi`. Note the `url`
    # fallback in `_to_paper_data` deliberately keeps the URL form — there the DOI is a
    # link, not an identifier.
    DOI_URL_PREFIX = "https://doi.org/"

    # Only the first 2000 characters are embedded; the rest is dropped server-side, so
    # there is no reason to put it on the wire.
    SEMANTIC_MAX_QUERY_CHARS = 2000

    def __init__(self, timeout = 10):
        base_url = settings.OPENALEX_API_BASE_URL
        bearer_token = settings.OPENALEX_API_TOKEN

        super().__init__(base_url, timeout=timeout, bearer_token=bearer_token)

    def search_papers(self, query: str, limit: int = 10) -> list[PaperData]:
        # Unlike Semantic Scholar, a blank search is a 200 here — it just matches every
        # work in OpenAlex. That makes the request worse than useless, so skip it.
        if not query.strip():
            return []

        return self._search_works({"search": query}, limit, self.MAX_LIMIT)

    def semantic_search_papers(self, query: str, limit: int = 10) -> list[PaperData]:
        """Rank works by embedding similarity rather than by keyword overlap.

        OpenAlex embeds every work's title and abstract with GTE Large EN and compares
        the query against them by cosine similarity, so this reaches papers that answer
        the query in vocabulary it never used — "predicting drug toxicity from molecular
        structure" finds work that only ever says "QSAR". It rewards long, prose-like
        input: an abstract or a paragraph beats a handful of keywords.
        """
        # Nothing to embed, and OpenAlex would answer with every work in the index —
        # the same reasoning as the keyword path.
        if not query.strip():
            return []

        return self._search_works(
            {"search.semantic": query[: self.SEMANTIC_MAX_QUERY_CHARS]},
            limit,
            self.SEMANTIC_MAX_LIMIT,
        )

    def _search_works(
        self, search_param: dict, limit: int, max_limit: int
    ) -> list[PaperData]:
        """Issue one /works search and map the payload.

        The keyword and semantic paths differ only in which search parameter carries the
        query and how high the per-page ceiling goes; everything else — the endpoint, the
        selected fields, the error handling and the mapping — is shared. `search_param`
        holds exactly one entry because the API rejects a request naming more than one of
        `search`, `search.exact` and `search.semantic`.
        """
        try:
            payload = self._request(
                "GET",
                "/works",
                params={
                    **search_param,
                    "per-page": max(1, min(limit, max_limit)),
                    "select": self.SEARCH_FIELDS,
                },
            )
        except APIError as error:
            logger.error(error)
            return []

        return [self._to_paper_data(item) for item in payload.get("results") or []]

    @staticmethod
    def _to_paper_data(item: dict) -> PaperData:
        # Every field is nullable while PaperData requires all seven, so each value is
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
            doi=OpenAlexClient._bare_doi(item.get("doi")),
            title=item.get("display_name") or "",
            year=item.get("publication_year") or 0,
            abstract=OpenAlexClient._abstract_from_inverted_index(
                item.get("abstract_inverted_index")
            ),
            url=url,
            license=license_,
            # Absent on any request carrying no search parameter, and null-able even
            # when present; both mean "this source did not rank this record".
            relevance_score=float(item.get("relevance_score") or 0.0),
        )

    @staticmethod
    def _bare_doi(value: str | None) -> str:
        if not value:
            return ""

        return value.removeprefix(OpenAlexClient.DOI_URL_PREFIX)

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
