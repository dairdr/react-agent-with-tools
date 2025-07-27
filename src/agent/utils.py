"""Utility functions for the agent graph."""

import json
from typing import List, Dict, Any, Optional
from langchain_core.messages import ToolMessage, FunctionMessage


def has_too_many_consecutive_tool_calls(messages: list, max_calls: int = 3, look_back: int = 10) -> bool:
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


def message_has_tool_calls(message) -> bool:
    """Check if a message contains tool calls in either format."""
    # Check new tool_calls format
    if hasattr(message, "tool_calls") and getattr(message, "tool_calls", None):
        return True

    # Check old function_call format
    if hasattr(message, "additional_kwargs"):
        additional_kwargs = getattr(message, "additional_kwargs", {})
        return "function_call" in additional_kwargs

    return False


def is_tool_response(message) -> bool:
    """Check if a message is a tool or function response."""
    return hasattr(message, "__class__") and message.__class__.__name__ in ["ToolMessage", "FunctionMessage"]


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


def process_new_format_tool_calls(tool_calls: List[Dict], tools: List) -> List[ToolMessage]:
    """Process tool calls in the new format and return ToolMessage list."""
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


def process_old_format_function_call(function_call: Dict, tools: List) -> FunctionMessage:
    """Process function call in the old format and return FunctionMessage."""
    tool_name = function_call["name"]
    tool_input = json.loads(function_call["arguments"])

    tool = find_tool_by_name(tools, tool_name)
    if tool:
        result_content = execute_tool(tool, tool_name, tool_input)
    else:
        result_content = f"Tool '{tool_name}' not found"

    return FunctionMessage(content=result_content, name=tool_name)
