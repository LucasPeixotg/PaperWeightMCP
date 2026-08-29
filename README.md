# PaperWeightMCP

A lightweight academic research agent built as a [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server.

PaperWeightMCP uses a two-stage hybrid retrieval strategy—combining fast semantic search over pre-indexed paper abstracts with an internal HyDE translation layer, Text-to-SQL metadata filtering, and on-the-fly full-text fetching. The goal is to maximize research depth without requiring massive local disk space.

## Features
 
> 🚧 **Status:** PaperWeightMCP is under active development. The features below are currently being built.
 
- [ ] **Hybrid retrieval** — semantic search over abstracts plus exact metadata filtering
- [ ] **Low storage footprint** — full texts are fetched on demand rather than stored locally
- [ ] **HyDE translation layer** — improves recall by reformulating queries before search
- [ ] **Text-to-SQL filtering** — natural language turned into precise metadata queries
- [ ] **Citation graph traversal** — walk forward and backward through the literature
- [ ] **Artifact discovery** — surface code, models, and datasets tied to a paper


## Tools

### Core Retrieval & Search

| Tool | Description |
|------|-------------|
| `search_paper_abstracts(query, top_k)` | Semantic vector search over pre-indexed abstracts to retrieve the most relevant publications. |
| `fetch_full_paper(paper_id_or_url)` | Downloads and extracts full-text content on the fly for granular details, equations, or code implementations. |
| `query_paper_metadata(sql, limit)` | Executes a read-only SQL SELECT query against the papers metadata table for exact filtering by date, citation count, author, category, or other structured fields. |

### Citation & Impact Tracking

| Tool | Description |
|------|-------------|
| `get_citations(paper_id, limit)` | Forward citation search to find newer studies and benchmarks that cited a paper. |
| `get_references(paper_id, limit)` | Backward citation search to extract a paper's bibliography and foundational prior work. |
| `get_author_top_works(author_id, limit)` | Retrieves the most influential or highly cited works by a specific researcher or lab. |

### Code & Artifact Discovery

| Tool | Description |
|------|-------------|
| `get_paper_artifacts(paper_id)` | Extracts links to execution assets—GitHub repos, Hugging Face weights, datasets, and project pages. |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/PaperWeightMCP.git
cd PaperWeightMCP

# Install dependencies
# TODO: add install steps (e.g. pip install -r requirements.txt)
```

## Usage

Add PaperWeightMCP to your MCP client configuration:

```json
{
  "mcpServers": {
    "paperweight": {
      "command": "TODO",
      "args": ["TODO"]
    }
  }
}
```

<!-- TODO: describe how to run the server and a quick example query -->

## Configuration

<!-- TODO: document environment variables, index paths, and any API keys -->

## License

<!-- TODO: choose a license -->
