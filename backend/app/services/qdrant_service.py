import uuid
from typing import Any
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.config import get_settings

settings = get_settings()

parsed_url = urlparse(settings.QDRANT_URL)
qdrant_client = QdrantClient(
    host=parsed_url.hostname,
    port=parsed_url.port or (443 if parsed_url.scheme == "https" else 80),
    https=parsed_url.scheme == "https",
    api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
    prefer_grpc=False,
)

COLLECTION_NAME = settings.QDRANT_COLLECTION_NAME
VECTOR_SIZE = 1536  # text-embedding-3-small dimension


def get_collection() -> bool:
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        return True
    except Exception:
        return False


def create_collection() -> None:
    existing = get_collection()
    if existing:
        return
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qdrant_models.VectorParams(
            size=VECTOR_SIZE,
            distance=qdrant_models.Distance.COSINE,
        ),
    )
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="tenant_id",
        field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
    )
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="document_id",
        field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
    )


def delete_collection() -> None:
    if get_collection():
        qdrant_client.delete_collection(COLLECTION_NAME)


def upsert_points(points: list[dict[str, Any]]) -> None:
    qdrant_points = []
    for point in points:
        qdrant_points.append(
            qdrant_models.PointStruct(
                id=point["id"],
                vector=point["vector"],
                payload=point["payload"],
            )
        )
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=qdrant_points,
    )


def search(
    query_vector: list[float],
    tenant_id: int,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    if top_k is None:
        top_k = settings.RAG_TOP_K
    if score_threshold is None:
        score_threshold = settings.RAG_SCORE_THRESHOLD

    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="tenant_id",
                    match=qdrant_models.MatchValue(value=str(tenant_id)),
                )
            ]
        ),
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )

    return [
        {
            "id": hit.id,
            "score": hit.score,
            "content": hit.payload.get("content", ""),
            "document_name": hit.payload.get("document_name", ""),
            "page_number": hit.payload.get("page_number", 0),
            "document_id": hit.payload.get("document_id", ""),
            "chunk_id": hit.payload.get("chunk_id", 0),
        }
        for hit in results
    ]


def delete_by_document_id(tenant_id: int, document_id: int) -> None:
    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=qdrant_models.FilterSelector(
            filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="tenant_id",
                        match=qdrant_models.MatchValue(value=str(tenant_id)),
                    ),
                    qdrant_models.FieldCondition(
                        key="document_id",
                        match=qdrant_models.MatchValue(value=str(document_id)),
                    ),
                ]
            )
        ),
    )
