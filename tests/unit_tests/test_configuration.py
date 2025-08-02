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
with patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_llm:
    mock_llm.return_value = Mock()
    from agent.config import Configuration
    from agent.graph import graph
    from agent.tools import tools
    from agent.utils import get_system_message, get_llm, get_agent_config


def test_graph_is_pregel_instance() -> None:
    """Test that the graph is a properly configured Pregel instance."""
    assert isinstance(graph, Pregel)


def test_system_message_configuration() -> None:
    """Test that the system message is properly configured."""
    config = Configuration()
    system_message = get_system_message(config)

    assert isinstance(system_message, SystemMessage)
    assert system_message.content is not None

    # Handle both string and list content types
    content_str = str(system_message.content)
    assert len(content_str) > 0
    assert "helpful assistant" in content_str.lower()
    assert "search" in content_str.lower()


def test_llm_configuration() -> None:
    """Test that the LLM is properly configured."""
    config = Configuration()

    with patch('agent.utils.ChatGoogleGenerativeAI') as mock_llm_class:
        mock_llm_instance = Mock()
        mock_llm_class.return_value = mock_llm_instance

        llm = get_llm(config)

        # Check that the LLM was created with correct parameters
        mock_llm_class.assert_called_once_with(
            model=config.model_name,
            temperature=config.temperature
        )
        assert llm == mock_llm_instance


def test_configuration_defaults() -> None:
    """Test that Configuration has the expected default values."""
    config = Configuration()

    assert config.model_name == "gemini-2.5-flash"
    assert config.temperature == 0.0
    assert config.max_tool_calls == 5
    assert len(config.system_message) > 0


def test_tools_are_base_tool_instances() -> None:
    """Test that all tools are proper BaseTool instances."""
    # The tools should be available in the tools list
    assert len(tools) == 2

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

    # Check that we have the expected nodes
    expected_nodes = ['agent_node', 'tools_node']
    for node in expected_nodes:
        assert node in node_names

    # The graph should have some essential structure
    assert len(node_names) >= 2


def test_get_agent_config() -> None:
    """Test the get_agent_config function."""
    from langchain_core.runnables import RunnableConfig

    # Test with empty config
    empty_config: RunnableConfig = {}
    result = get_agent_config(empty_config)
    assert isinstance(result, Configuration)
    assert result.model_name == "gemini-2.5-flash"  # Should use defaults

    # Test with valid configuration fields
    valid_config: RunnableConfig = {
        "configurable": {
            "model_name": "gemini-pro",
            "temperature": 0.5,
            "max_tool_calls": 10,
            "extra_field": "should_be_ignored"  # Extra fields should be filtered out
        }
    }
    result = get_agent_config(valid_config)
    assert isinstance(result, Configuration)
    assert result.model_name == "gemini-pro"
    assert result.temperature == 0.5
    assert result.max_tool_calls == 10

    # Test with invalid config (should fall back to defaults)
    invalid_config: RunnableConfig = {
        "configurable": {
            "temperature": "invalid_type"  # Wrong type
        }
    }
    result = get_agent_config(invalid_config)
    assert isinstance(result, Configuration)
    assert result.temperature == 0.0  # Should use default
