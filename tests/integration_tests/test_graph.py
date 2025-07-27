"""Integration tests for the agent graph."""

import pytest
from unittest.mock import patch, Mock
from langchain_core.messages import HumanMessage

from agent import graph
from agent.graph import AgentState

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_agent_simple_message() -> None:
    """Test agent can process a simple message."""
    inputs: AgentState = {"messages": [
        HumanMessage(content="Hello, how are you?")]}
    res = await graph.ainvoke(inputs)
    assert res is not None
    assert "messages" in res
    assert len(res["messages"]) > 0


@pytest.mark.langsmith
async def test_agent_responds_to_greeting() -> None:
    """Test that agent responds appropriately to greetings."""
    inputs: AgentState = {"messages": [HumanMessage(content="Hello")]}
    res = await graph.ainvoke(inputs)

    assert res is not None
    assert "messages" in res
    # Should have at least the user message and agent response
    assert len(res["messages"]) >= 2


@pytest.mark.langsmith
@patch('agent.brave_search_tool.BraveSearch.from_api_key')
async def test_agent_with_search_tool_request(mock_brave_search) -> None:
    """Test agent can handle requests that might use search tools."""
    # Mock the Brave Search tool
    mock_search_instance = Mock()
    mock_search_instance.run.return_value = "Weather information for today"
    mock_brave_search.return_value = mock_search_instance

    inputs: AgentState = {"messages": [HumanMessage(
        content="What's the weather like today?")]}
    res = await graph.ainvoke(inputs)

    assert res is not None
    assert "messages" in res
    assert len(res["messages"]) >= 2


@pytest.mark.langsmith
@patch('agent.arxiv_search_tool.arxiv.Client')
async def test_agent_with_arxiv_search_request(mock_arxiv_client) -> None:
    """Test agent can handle requests for academic papers."""
    # Mock the ArXiv search
    mock_client = Mock()
    mock_paper = Mock()
    mock_paper.title = "Test Machine Learning Paper"
    mock_paper.authors = [Mock(name="John Doe")]
    mock_paper.summary = "This is a test paper about machine learning."
    mock_paper.published = Mock()
    mock_paper.published.strftime.return_value = "2023-01-15"
    mock_paper.entry_id = "http://arxiv.org/abs/2301.12345v1"
    mock_paper.categories = ["cs.AI"]

    mock_client.results.return_value = [mock_paper]
    mock_arxiv_client.return_value = mock_client

    inputs: AgentState = {"messages": [HumanMessage(
        content="Find papers about machine learning")]}
    res = await graph.ainvoke(inputs)

    assert res is not None
    assert "messages" in res
    assert len(res["messages"]) >= 2


@pytest.mark.langsmith
async def test_agent_state_structure() -> None:
    """Test that the agent maintains proper state structure."""
    inputs: AgentState = {"messages": [HumanMessage(content="Test message")]}
    res = await graph.ainvoke(inputs)

    # Check that response has the expected structure
    assert isinstance(res, dict)
    assert "messages" in res
    assert isinstance(res["messages"], list)

    # Each message should be a proper message type
    for message in res["messages"]:
        assert hasattr(message, 'content') or hasattr(message, 'tool_calls')


@pytest.mark.langsmith
async def test_agent_handles_empty_input() -> None:
    """Test agent behavior with minimal input."""
    inputs: AgentState = {"messages": []}
    res = await graph.ainvoke(inputs)

    assert res is not None
    assert "messages" in res
    # Should handle empty messages gracefully
