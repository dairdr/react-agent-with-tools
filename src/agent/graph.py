"""This module defines the state graph for the agent, including the main agent node and tool handling."""

from typing import Sequence

from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph

from agent.tools import tools
from agent.utils import (
    filter_empty_content_messages,
    has_too_many_consecutive_tool_calls,
    message_has_tool_calls,
    process_tool_calls,
)


class Configuration:
    """Basic configuration for agent."""
    pass


SYSTEM_MESSAGE = SystemMessage(
    content="""You are a helpful assistant that can search the web and academic papers to answer questions. You can use tools like Brave Search and ArXiv Search to find information. If you need to use a tool, you will call it with the appropriate arguments. If you cannot find an answer, you will say 'I don't know'.""",
)

llm_name = "gemini-2.5-flash"
llm = ChatGoogleGenerativeAI(model=llm_name)
llm_with_tools = llm.bind_tools(tools)


class AgentState(MessagesState):
    """Global agent state that includes messages."""
    pass


def agent_node(state: AgentState, config: RunnableConfig) -> dict[str, Sequence[BaseMessage]]:
    """Process user messages and generate a response."""
    messages = state["messages"]

    # Add system message if it's not already the first message
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_MESSAGE] + messages

    # Handle empty conversation case - Gemini requires at least one user message
    if len(messages) == 1 and isinstance(messages[0], SystemMessage):
        # Add a minimal user message to satisfy Gemini's requirements
        from langchain_core.messages import HumanMessage
        messages.append(HumanMessage(content="Hello"))

    # Filter messages to ensure no empty content (prevents Gemini errors)
    processed_messages = filter_empty_content_messages(messages)

    response = llm_with_tools.invoke(processed_messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end."""
    messages = state["messages"]
    last_message: AnyMessage = messages[-1]

    # Prevent infinite loops by checking recent tool usage
    if has_too_many_consecutive_tool_calls(messages):
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


workflow = StateGraph(AgentState)

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
