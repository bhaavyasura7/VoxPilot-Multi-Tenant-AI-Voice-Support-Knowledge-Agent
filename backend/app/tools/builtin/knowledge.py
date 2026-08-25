from app.tools.registry import registry


async def search_knowledge_base(
    tenant_id: int,
    query: str,
    top_k: int = 4,
    **kwargs,
) -> str:
    from app.services.pipeline import retrieve_relevant_chunks
    import json

    results = await retrieve_relevant_chunks(
        query=query,
        tenant_id=tenant_id,
        top_k=top_k,
    )

    if not results:
        return json.dumps({"message": "No relevant documents found in the knowledge base."})

    items = []
    for r in results:
        items.append({
            "document": r["document_name"],
            "page": r["page_number"],
            "score": round(r["score"], 3),
            "content": r["content"][:600],
        })

    return json.dumps({"results": items, "total": len(items)})


registry.register(
    name="search_knowledge_base",
    description="Search the organization's knowledge base for relevant information. Use this to find answers from uploaded documents like PDFs, policies, FAQs, etc.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query or question to look up in the knowledge base",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top results to return. Default 4. Increase for broader searches.",
                "default": 4,
            },
        },
        "required": ["query"],
    },
    handler=search_knowledge_base,
)
