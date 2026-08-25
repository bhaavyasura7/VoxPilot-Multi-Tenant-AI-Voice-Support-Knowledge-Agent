import json
import logging

from app.tools.registry import registry

logger = logging.getLogger(__name__)


async def execute_tool(name: str, arguments: str, **ctx) -> str:
    handler = registry.get_handler(name)
    if not handler:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        parsed_args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return json.dumps({"error": f"Invalid arguments JSON for tool: {name}"})

    try:
        result = await handler(**ctx, **parsed_args)
        return result
    except Exception as e:
        logger.error(f"Tool '{name}' execution failed: {e}")
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})
