from openai import AsyncOpenAI

from app.config import get_settings

settings = get_settings()

openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def generate_embedding(text: str) -> list[float]:
    response = await openai_client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


async def generate_embeddings(texts: list[str]) -> list[list[float]]:
    response = await openai_client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


async def count_tokens(text: str) -> int:
    import tiktoken

    encoding = tiktoken.encoding_for_model(settings.EMBEDDING_MODEL)
    return len(encoding.encode(text))
