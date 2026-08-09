import tiktoken

from app.config import get_settings

settings = get_settings()


def chunk_text(
    text: str,
    page_number: int,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if overlap is None:
        overlap = settings.CHUNK_OVERLAP

    encoding = tiktoken.encoding_for_model(settings.EMBEDDING_MODEL)
    tokens = encoding.encode(text)
    chunks = []

    start = 0
    chunk_index = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_decoded = encoding.decode(chunk_tokens)

        if chunk_text_decoded.strip():
            chunks.append(
                {
                    "content": chunk_text_decoded.strip(),
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

        if end >= len(tokens):
            break
        start = end - overlap

    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    all_chunks = []
    global_index = 0

    for page in pages:
        page_chunks = chunk_text(
            text=page["content"],
            page_number=page["page_number"],
        )
        for chunk in page_chunks:
            chunk["chunk_index"] = global_index
            all_chunks.append(chunk)
            global_index += 1

    return all_chunks
