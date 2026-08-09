from openai import AsyncOpenAI

from app.config import get_settings

settings = get_settings()

openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def generate_rag_answer(
    query: str,
    context_chunks: list[dict],
    conversation_history: list[dict] | None = None,
) -> str:
    context_text = "\n\n---\n\n".join(
        [
            f"[Source: {chunk['document_name']}, Page {chunk['page_number']}]\n{chunk['content']}"
            for chunk in context_chunks
        ]
    )

    system_prompt = """You are a helpful AI support agent for an organization. 
You answer questions based on the organization's provided documents and knowledge base.

RULES:
1. Base your answer on the provided context chunks. Cite sources when relevant.
2. If the context does not contain enough information to answer the question, 
   say: "I couldn't find that information in the organization's knowledge base."
3. Never invent policies, rules, or information not present in the context.
4. Be concise and direct. For voice-compatible responses, keep answers brief.
5. If the question is a follow-up to previous conversation, use that context too."""

    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append(
        {
            "role": "user",
            "content": f"Context from organization documents:\n\n{context_text}\n\nQuestion: {query}",
        }
    )

    response = await openai_client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1000,
    )

    return response.choices[0].message.content or "I couldn't generate a response."
