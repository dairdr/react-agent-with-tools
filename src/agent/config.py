"""Configuration module for the agent.

This module defines the Configuration class that contains all configurable
parameters for the agent, including LLM settings, behavior limits, and 
system message customization.
"""

from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Configuration for the agent."""

    model_name: str = Field(
        default="gemini-2.5-flash",
        description="LLM model to use"
    )
    temperature: float = Field(
        default=0.0, description="Temperature for LLM responses"
    )
    max_tool_calls: int = Field(
        default=5, description="Maximum consecutive tool calls before stopping"
    )
    system_message: str = Field(
        default="""You are a helpful assistant that can search the web and academic papers to answer questions. 
                You can use tools like Brave Search and ArXiv Search to find information. 
                If you need to use a tool, you will call it with the appropriate arguments. 
                If you cannot find an answer, you will say 'I don't know'.""",
        description="System message for the agent"
    )
