"""This module defines the state graph for the agent, including the main agent node and tool handling."""

from typing import Sequence

from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph

from agent.config import Configuration
from agent.tools import tools
from agent.utils import (
    filter_empty_content_messages,
    get_agent_config,
    get_llm,
    get_system_message,
    has_too_many_consecutive_tool_calls,
    message_has_tool_calls,
    process_tool_calls,
)


class AgentState(MessagesState):
    """Global agent state that includes messages."""
    pass


def agent_node(state: AgentState, config: RunnableConfig) -> dict[str, Sequence[BaseMessage]]:
    """Process user messages and generate a response."""
    agent_config: Configuration = get_agent_config(config)

    # Get LLM and system message based on configuration
    llm = get_llm(agent_config)
    llm_with_tools = llm.bind_tools(tools)
    system_message = get_system_message(agent_config)

    messages = state["messages"]

    # Add system message if it's not already the first message
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [system_message] + messages

    # Handle empty conversation case - Gemini requires at least one user message
    if len(messages) == 1 and isinstance(messages[0], SystemMessage):
        # Add a minimal user message to satisfy Gemini's requirements
        from langchain_core.messages import HumanMessage
        messages.append(HumanMessage(content="Hello"))

    # Filter messages to ensure no empty content (prevents Gemini errors)
    processed_messages = filter_empty_content_messages(messages)

    response = llm_with_tools.invoke(processed_messages)
    return {"messages": [response]}


def should_continue(state: AgentState, config: RunnableConfig) -> str:
    """Determine if we should continue or end."""
    agent_config: Configuration = get_agent_config(config)

    messages = state["messages"]
    last_message: AnyMessage = messages[-1]

    # Prevent infinite loops by checking recent tool usage with configured max_tool_calls
    if has_too_many_consecutive_tool_calls(messages, max_calls=agent_config.max_tool_calls):
        return END

    # Check if the last message contains tool calls
    if message_has_tool_calls(last_message):
        return "tools_edge"

    return END


def tools_node(state: AgentState, config: RunnableConfig) -> dict[str, Sequence[BaseMessage]]:
    """Handle tool execution using the modern tool calling format."""
    last_message: AnyMessage = state["messages"][-1]

    # Handle tool_calls format
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        messages_to_add = process_tool_calls(tool_calls, tools)
        return {"messages": messages_to_add}

    # log a warning for observation

    # No tool calls found - this shouldn't happen with modern LLMs
    return {"messages": [ToolMessage(content="No tool calls found", tool_call_id="error")]}


workflow = StateGraph(AgentState, config_schema=Configuration)

workflow.add_node("agent_node", agent_node)
workflow.add_node("tools_node", tools_node)

workflow.add_edge(START, "agent_node")
workflow.add_edge("tools_node", "agent_node")
workflow.add_conditional_edges(
    "agent_node",
    should_continue,
    {
        "tools_edge": "tools_node",
        END: END
    }
)

graph = workflow.compile()
