"""Unit tests for agent configuration and graph setup."""

import os
import sys
from unittest.mock import Mock, patch

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Mock the Google credentials to avoid authentication issues during import
with patch('agent.graph.ChatGoogleGenerativeAI') as mock_llm:
    mock_llm.return_value = Mock()
    from agent.graph import SYSTEM_MESSAGE, graph, llm_with_tools
    from agent.tools import tools


def test_graph_is_pregel_instance() -> None:
    """Test that the graph is a properly configured Pregel instance."""
    assert isinstance(graph, Pregel)


def test_system_message_configuration() -> None:
    """Test that the system message is properly configured."""
    assert isinstance(SYSTEM_MESSAGE, SystemMessage)
    assert SYSTEM_MESSAGE.content is not None

    # Handle both string and list content types
    content_str = str(SYSTEM_MESSAGE.content)
    assert len(content_str) > 0
    assert "helpful assistant" in content_str.lower()
    assert "search" in content_str.lower()


def test_llm_with_tools_configuration() -> None:
    """Test that the LLM is properly configured with tools."""
    # Check that llm_with_tools is a runnable (has been bound with tools)
    assert hasattr(llm_with_tools, 'invoke')
    assert hasattr(llm_with_tools, 'ainvoke')

    # The tools should be available in the tools list
    assert len(tools) == 2


def test_tools_are_base_tool_instances() -> None:
    """Test that all tools are proper BaseTool instances."""
    for tool in tools:
        assert isinstance(tool, BaseTool)
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert callable(tool._run)


def test_graph_has_required_nodes() -> None:
    """Test that the graph contains the expected nodes."""
    # Get the graph's compiled structure
    compiled_graph = graph.get_graph()
    node_names = list(compiled_graph.nodes.keys())

    # Check that we have some nodes
    assert len(node_names) > 0

    # The graph should have some essential structure
    # This is a basic structural test to ensure the graph is built
