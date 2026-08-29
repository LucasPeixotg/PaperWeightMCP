"""HyDE (Hypothetical Document Embeddings) query translation.

Instead of embedding the user's question directly, the model first writes a
plausible abstract answering it. That hypothetical document sits much closer in
embedding space to real abstracts than a question does, which lifts recall.
"""

import numpy as np

from config import settings

from .rag_model import RagModel

HYDE_PROMPT = "Write a plausible academic abstract answering: {query}"


def hyde_embed(model: RagModel, query: str) -> np.ndarray:
    """Embed a query via a generated hypothetical abstract."""
    hypothetical_doc = model.generate(
        HYDE_PROMPT.format(query=query),
        max_new_tokens=settings.hyde_max_new_tokens,
        temperature=settings.hyde_temperature,
    )
    return model.embedder.encode(hypothetical_doc)