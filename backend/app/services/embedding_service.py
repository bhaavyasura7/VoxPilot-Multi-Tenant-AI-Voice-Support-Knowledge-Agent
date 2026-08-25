from openai import AsyncOpenAI

from app.config import get_settings

settings = get_settings()

openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def generate_embedding(text: str) -> list[float]:
    from app.services.cache_service import get_cached_embedding, set_cached_embedding

    cached = await get_cached_embedding(text)
    if cached is not None:
        return cached

    response = await openai_client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=text,
    )
    embedding = response.data[0].embedding
    await set_cached_embedding(text, embedding)
    return embedding


async def generate_embeddings(texts: list[str]) -> list[list[float]]:
    from app.services.cache_service import get_cached_embedding, set_cached_embedding

    embeddings = []
    uncached_indices = []
    uncached_texts = []

    for i, text in enumerate(texts):
        cached = await get_cached_embedding(text)
        if cached is not None:
            embeddings.append(cached)
        else:
            embeddings.append(None)
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        response = await openai_client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=uncached_texts,
        )
        for j, item in enumerate(response.data):
            idx = uncached_indices[j]
            embeddings[idx] = item.embedding
            await set_cached_embedding(uncached_texts[j], item.embedding)

    return embeddings


async def count_tokens(text: str) -> int:
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
