import json
import logging

from openai import AsyncOpenAI

from app.config import get_settings
from app.tools.registry import registry
from app.tools.executor import execute_tool

# Import builtin tools to register them
import app.tools.builtin  # noqa: F401

logger = logging.getLogger(__name__)
settings = get_settings()

openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

MAX_TOOL_ITERATIONS = 5

AGENT_SYSTEM_PROMPT = """You are a helpful, knowledgeable AI support agent for {org_name}.
You answer customer questions using the organization's documents and knowledge base.

YOUR CAPABILITIES:
- You have access to tools that let you search the knowledge base, list available documents, and retrieve conversation history.
- Use tools when needed to find accurate, sourced answers.
- If a tool returns no results or an error, tell the user honestly rather than guessing.

RULES:
1. Always use search_knowledge_base first when answering questions about the organization.
2. Use list_documents when asked what information you can help with.
3. Use get_conversation_history when asked about past conversations.
4. If multiple searches are needed, search again with different queries.
5. Base answers ONLY on tool results. Never invent policies, rules, or information.
6. Be concise and warm. Cite document names when relevant.
7. If you cannot find an answer after searching, say so clearly."""

VOICE_SYSTEM_PROMPT = """You are a helpful, friendly voice AI support agent for {org_name}.
Answer customer questions using ONLY the provided knowledge base tools.

RULES:
- Keep spoken answers concise and natural — under 2-3 sentences.
- Always search the knowledge base before answering factual questions.
- Use list_documents when asked what you can help with.
- Base your answer ONLY on search results. Never invent policies or information.
- If search returns nothing, say: "I couldn't find that in our knowledge base."
- Be warm and conversational."""


class AgentSession:
    def __init__(
        self,
        tenant_id: int,
        user_id: int,
        org_name: str = "this organization",
        voice_mode: bool = False,
    ):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.org_name = org_name
        self.voice_mode = voice_mode
        self._tools = registry.get_definitions()
        self._system_prompt = (
            VOICE_SYSTEM_PROMPT if voice_mode else AGENT_SYSTEM_PROMPT
        ).format(org_name=org_name)

    async def run(self, user_message: str, conversation_history: list[dict] | None = None) -> tuple[str, list[str]]:
        messages: list[dict] = [{"role": "system", "content": self._system_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_message})

        tools_used: list[str] = []

        for iteration in range(MAX_TOOL_ITERATIONS):
            kwargs = {
                "model": settings.LLM_MODEL,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1000 if not self.voice_mode else 300,
            }

            if self._tools:
                kwargs["tools"] = self._tools
                if iteration == 0:
                    kwargs["tool_choice"] = "auto"

            response = await openai_client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                assistant_tool_msg = {
                    "role": "assistant",
                    "content": choice.message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.message.tool_calls
                    ],
                }
                messages.append(assistant_tool_msg)

                for tc in choice.message.tool_calls:
                    tool_name = tc.function.name
                    tool_args = tc.function.arguments
                    tools_used.append(tool_name)

                    logger.info(
                        f"Agent calling tool: {tool_name} (tenant={self.tenant_id}, user={self.user_id})"
                    )

                    result = await execute_tool(
                        name=tool_name,
                        arguments=tool_args,
                        tenant_id=self.tenant_id,
                        user_id=self.user_id,
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                continue

            answer = choice.message.content or "I couldn't generate a response."
            return answer, tools_used

        return "I'm having trouble finding the right information. Could you rephrase your question?", tools_used
