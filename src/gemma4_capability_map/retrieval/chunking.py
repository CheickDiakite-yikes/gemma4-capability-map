from __future__ import annotations

from gemma4_capability_map.schemas import Document


def chunk_document(document: Document, chunk_size: int = 120) -> list[Document]:
    words = document.content.split()
    if len(words) <= chunk_size:
        return [document]
    chunks: list[Document] = []
    for index in range(0, len(words), chunk_size):
        chunk_words = words[index:index + chunk_size]
        chunks.append(
            Document(
                doc_id=f"{document.doc_id}::chunk-{index // chunk_size}",
                content=" ".join(chunk_words),
                metadata={**document.metadata, "source_doc_id": document.doc_id},
            )
        )
    return chunks

