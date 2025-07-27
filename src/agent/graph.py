from copy import copy
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import FunctionMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
# from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.tools import tools
from agent.utils import (
    has_too_many_consecutive_tool_calls,
    message_has_tool_calls,
    process_new_format_tool_calls,
    process_old_format_function_call
)


class Configuration:
    pass


SYSTEM_MESSAGE = SystemMessage(
    content="""You are a helpful assistant that can search the web and academic papers to answer questions. You can use tools like Brave Search and ArXiv Search to find information. If you need to use a tool, you will call it with the appropriate arguments. If you cannot find an answer, you will say 'I don't know'.""",

)

# llm_name = "llama3.2:3b-instruct-fp16"
# llm = ChatOllama(
#     model=llm_name,
#     temperature=0.1,
# )
llm_name = "gemini-2.5-flash"
llm = ChatGoogleGenerativeAI(model=llm_name)
llm_with_tools = llm.bind_tools(tools)


class AgentState(MessagesState):
    pass


def agent_node(state: AgentState, config: RunnableConfig):
    """Main agent node that processes the input."""
    messages = state["messages"]

    # Add system message if it's not already the first message
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_MESSAGE] + messages

    # Ensure no message has completely empty content (which causes Gemini errors)
    processed_messages = []
    for message in messages:
        content = getattr(message, "content", "")

        # If content is empty or just whitespace, provide minimal content
        if not content or (isinstance(content, str) and not content.strip()):
            # Create a copy with minimal non-empty content
            new_message = copy(message)
            if isinstance(message, (ToolMessage, FunctionMessage)):
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

    response = llm_with_tools.invoke(processed_messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end."""
    messages = state["messages"]
    last_message = messages[-1]

    # Prevent infinite loops by checking recent tool usage
    if has_too_many_consecutive_tool_calls(messages):
        return END

    # Check if the last message contains tool calls
    if message_has_tool_calls(last_message):
        return "tools_edge"

    return END


def tools_node(state: AgentState, config: RunnableConfig):
    """Handle tool execution for both new and old tool calling formats."""
    last_message = state["messages"][-1]

    # Handle new tool_calls format (preferred)
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        messages_to_add = process_new_format_tool_calls(tool_calls, tools)
        return {"messages": messages_to_add}

    # Handle old function_call format (fallback)
    additional_kwargs = getattr(last_message, "additional_kwargs", {})
    function_call = additional_kwargs.get("function_call")
    if function_call:
        message = process_old_format_function_call(function_call, tools)
        return {"messages": [message]}

    # No tool calls found
    return {"messages": [FunctionMessage(content="No tool calls found", name="error")]}


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
