"""Brave Search tool for web searching."""

from typing import Optional
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.tools import BraveSearch


class BraveSearchTool(BaseTool):
    """Tool for searching the web using Brave Search."""

    name: str = "brave_search"
    description: str = "Search the web using Brave Search. Provide a search query as input."
    api_key: Optional[str] = None

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the BraveSearchTool with an API key."""
        super().__init__(api_key=api_key, **kwargs)
        self._brave_search_tool = None
        if self.api_key:
            self._brave_search_tool = BraveSearch.from_api_key(
                api_key=self.api_key)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the search with the given query about any topic you can find in internet."""
        if self._brave_search_tool:
            return self._brave_search_tool.run(query)
        else:
            raise ValueError(
                "API key for Brave Search is required to run this tool."
            )
