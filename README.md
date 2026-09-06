# PaperWeightMCP

An MCP server that finds the academic papers most relevant to a natural-language
research question.

It holds no local corpus and no database. Every query is answered by searching
public research APIs live, then merging and reranking the candidates locally with
a cross-encoder before handing back a short, ordered list.

## How it works

A call to `find_relevant_papers` runs one pipeline
(`src/tools/find_relevant_papers/tool.py`):

1. **Clean the query** — `*` and `?` are stripped, since search APIs read them as
   wildcard operators rather than as language.
2. **Search** — OpenAlex's semantic (embedding) endpoint, deliberately over-fetching
   `max(top_k * 2, RERANK_CANDIDATE_POOL)` candidates. A reranker can only improve on
   the order it is given, so it needs a pool wider than the slots it fills.
3. **Deduplicate** — two records are the same work if their DOIs match, or if title
   and abstract both clear a fuzzy-match threshold. A published copy beats a preprint
   of the same paper.
4. **Rerank** — a cross-encoder scores each candidate against the *raw* query, using
   the source's own relevance score to break ties.
5. **Split** — the top `top_k` slots are filled only with papers that actually carry
   an abstract; higher-ranked papers whose source publishes no abstract are returned
   separately rather than dropped.

## Requirements

- Python 3.13
- ~2 GB of disk for dependencies (`sentence-transformers` pulls in PyTorch), plus a
  one-time download of the cross-encoder weights on first run.

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
python src/server.py
```

The server speaks MCP over stdio. To register it with an MCP client:

```json
{
  "mcpServers": {
    "paperweight": {
      "command": "/absolute/path/to/PaperWeightMCP/venv/bin/python",
      "args": ["/absolute/path/to/PaperWeightMCP/src/server.py"]
    }
  }
}
```

## Tools

| Tool | Signature | Returns |
| --- | --- | --- |
| `find_relevant_papers` | `(query: str, top_k: int = 5) -> str` | JSON with `papers` and `relevant_without_abstract` |


Each paper in either list carries the same seven fields:

| Field | Meaning |
| --- | --- |
| `paper_id` | The source's identifier for the work |
| `doi` | DOI, when the source publishes one |
| `title` | Paper title |
| `year` | Publication year |
| `abstract` | Abstract text (empty in `relevant_without_abstract`) |
| `url` | Best available link: open-access PDF, else landing page, else DOI |
| `license` | License of the location `url` points at, when known |

`papers` holds up to `top_k` matches that come with an abstract — the ones worth
reading now. `relevant_without_abstract` holds matches that ranked among the best but
whose abstract the source does not publish. That often means the work is not openly
available, though not always; some publishers restrict the abstract alone. They are
still worth following by DOI or URL.

## Configuration

Settings live in `src/config.py` and can be overridden through the environment or
the `.env` file at the repo root. Both API tokens are optional — they only raise
rate limits and daily budgets.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENALEX_API_BASE_URL` | `https://api.openalex.org` | OpenAlex endpoint |
| `OPENALEX_API_TOKEN` | *(empty)* | Optional; sent as a bearer token |
| `SEMANTIC_SCHOLAR_API_BASE_URL` | `https://api.semanticscholar.org/graph/v1` | Semantic Scholar Graph API endpoint |
| `SEMANTIC_SCHOLAR_API_TOKEN` | *(empty)* | Optional; sent as `x-api-key` |
| `RERANK_MODEL_ID` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used for reranking |
| `RERANK_CANDIDATE_POOL` | `40` | Candidates to rerank; the cost knob — one forward pass each. Past 50 buys nothing, since OpenAlex caps its semantic endpoint there. |

## Layout

```
src/
  server.py                        FastMCP instance and tool registration
  config.py                        Settings, loaded from the environment or .env
  common/paper_data.py             PaperData — the record every source maps into
  services/
    api/api_client.py              Shared HTTP client: headers, retries, error wrapping
    api/open_alex_client.py        OpenAlex keyword and semantic search
    api/semantic_scholar_client.py Semantic Scholar Graph API search
    models/reranker.py             Cross-encoder reranking
  tools/
    find_relevant_papers/          The one registered tool, plus its query cleaner
                                   and deduplicator
tests/
```

## Development

```bash
pytest
```

There is no `pyproject.toml`; imports are rooted at `src/`, and `tests/conftest.py`
puts that directory on `sys.path` to reproduce what running `python src/server.py`
does. No test touches the network — the API client suites drive `httpx.MockTransport`,
and the reranker suite stubs the cross-encoder out.

## Status

Only `find_relevant_papers` is registered today. Also in the plans, but not yet wired up:

- `fetch_full_paper`
- `fetch_paper_artifacts`
- `get_author_top_works`

and more ...

## License

MIT — see [LICENSE](LICENSE).
