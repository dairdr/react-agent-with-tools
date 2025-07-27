"""ArXiv search tool for academic paper searching."""


import arxiv
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool


class ArxivSearchTool(BaseTool):
    """Tool for searching academic papers on arXiv."""

    name: str = "arxiv_search"
    description: str = "Search for academic papers on arXiv. Provide a search query, max_results (default 5), and start position (default 0) for pagination."
    max_results: int = 5
    start: int = 0

    def __init__(self, max_results: int = 5, start: int = 0, **kwargs):
        """Initialize the ArxivSearchTool with maximum number of results to return and start position for pagination."""
        super().__init__(max_results=max_results, start=start, **kwargs)

    def _run(
        self,
        query: str,
        max_results: int | None = None,
        start: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        """Execute the search with the given query to find academic papers on arXiv.

        Args:
            query: The search query string
            max_results: Maximum number of results to return (overrides default)
            start: Starting position for pagination (overrides default)
            run_manager: Optional callback manager
        """
        # Use provided parameters or fall back to instance defaults
        max_results = max_results if max_results is not None else self.max_results
        start = start if start is not None else self.start

        try:
            # Create a search client
            client = arxiv.Client()

            # For pagination, we need to request enough results to cover our offset + max_results
            # We'll request a reasonable upper bound to ensure we have enough results
            total_needed = start + max_results
            # Request at least 50 results
            search_max_results = max(total_needed, 50)

            # Perform the search using built-in pagination
            search = arxiv.Search(
                query=query,
                max_results=search_max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            results = []
            # Use the built-in offset parameter for pagination and limit results manually
            for paper in client.results(search, offset=start):
                if len(results) >= max_results:
                    break

                paper_info = {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "summary": paper.summary,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "url": paper.entry_id,
                    "categories": paper.categories
                }
                results.append(paper_info)

            if not results:
                return f"No papers found for query: {query}"

            # Format the results as a readable string
            formatted_results = f"Found {len(results)} papers for query: '{query}' (showing results {start + 1}-{start + len(results)})\n\n"

            for i, paper in enumerate(results, 1):
                # Show first 3 authors
                authors_str = ", ".join(paper["authors"][:3])
                if len(paper["authors"]) > 3:
                    authors_str += " et al."

                formatted_results += f"{start + i}. **{paper['title']}**\n"
                formatted_results += f"   Authors: {authors_str}\n"
                formatted_results += f"   Published: {paper['published']}\n"
                formatted_results += f"   Categories: {', '.join(paper['categories'])}\n"
                formatted_results += f"   URL: {paper['url']}\n"
                formatted_results += f"   Summary: {paper['summary'][:300]}...\n\n"

            return formatted_results

        except Exception as e:
            return f"Error searching arXiv: {str(e)}"
