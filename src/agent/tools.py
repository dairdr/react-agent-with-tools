"""This module define all tools available for the agent."""

import os

from .arxiv_search_tool import ArxivSearchTool
from .brave_search_tool import BraveSearchTool

tools = [
    BraveSearchTool(api_key=os.getenv("BRAVE_SEARCH_API_KEY")),
    ArxivSearchTool(max_results=5),
]
