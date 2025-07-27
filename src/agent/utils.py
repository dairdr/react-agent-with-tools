"""Utility functions for the agent graph."""

from copy import copy
from typing import Any, Dict, List, Sequence

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage


def has_too_many_consecutive_tool_calls(
    messages: Sequence[AnyMessage],
    max_calls: int = 3,
    look_back: int = 10
) -> bool:
    """Check if there are too many consecutive tool calls to prevent infinite loops."""
    recent_messages = messages[-look_back:] if len(
        messages) > look_back else messages
    consecutive_tool_calls = 0

    for msg in reversed(recent_messages):
        if message_has_tool_calls(msg):
            consecutive_tool_calls += 1
        elif is_tool_response(msg):
            continue  # Skip tool responses, they don't break the chain
        else:
            break  # Stop at first non-tool message

    return consecutive_tool_calls >= max_calls


def message_has_tool_calls(message: AnyMessage) -> bool:
    """Check if a message contains tool calls."""
    # Check for tool_calls format (modern standard)
    tool_calls = getattr(message, "tool_calls", None)
    return bool(tool_calls)


def is_tool_response(message: AnyMessage) -> bool:
    """Check if a message is a tool response."""
    return isinstance(message, ToolMessage)


def execute_tool(tool, tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Execute a single tool and return the result content."""
    try:
        result = tool.run(tool_input)
        # Ensure result is not empty or None
        result_content = str(
            result) if result is not None else "Tool execution completed with no output"
        result_content = result_content.strip()
        if not result_content:
            result_content = "Tool execution completed but returned empty result"
        return result_content
    except Exception as e:
        return f"Error executing tool {tool_name}: {str(e)}"


def find_tool_by_name(tools: List, tool_name: str):
    """Find a tool by its name from the tools list."""
    return next((t for t in tools if t.name == tool_name), None)


def process_tool_calls(tool_calls: List[Dict], tools: List) -> List[ToolMessage]:
    """Process tool calls and return ToolMessage list."""
    messages_to_add = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]

        tool = find_tool_by_name(tools, tool_name)
        if tool:
            result_content = execute_tool(tool, tool_name, tool_input)
        else:
            result_content = f"Tool '{tool_name}' not found"

        messages_to_add.append(
            ToolMessage(content=result_content, tool_call_id=tool_call["id"])
        )

    return messages_to_add


def filter_empty_content_messages(messages: List[AnyMessage]) -> List[AnyMessage]:
    """Filter messages to ensure no message has empty content (prevents Gemini errors)."""
    processed_messages = []
    for message in messages:
        content = getattr(message, "content", "")

        # If content is empty or just whitespace, provide minimal content
        if not content or (isinstance(content, str) and not content.strip()):
            # Create a copy with minimal non-empty content
            new_message = copy(message)
            if isinstance(message, ToolMessage):
                new_message.content = "Completed"
            elif isinstance(message, SystemMessage):
                # Keep system message as is (it should have content)
                processed_messages.append(message)
                continue
            else:
                new_message.content = "..."
            processed_messages.append(new_message)
        else:
            processed_messages.append(message)

    return processed_messages
