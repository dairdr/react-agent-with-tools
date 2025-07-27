"""This module define all tools available for the agent."""

import os
from .brave_search_tool import BraveSearchTool
from .arxiv_search_tool import ArxivSearchTool


tools = [
    BraveSearchTool(api_key=os.getenv("BRAVE_SEARCH_API_KEY")),
    ArxivSearchTool(max_results=5),
]
