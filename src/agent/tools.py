import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any
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


class ArXivSearchTool(BaseTool):
    """Tool for searching academic papers using ArXiv API."""

    name: str = "arxiv_search"
    description: str = "Search for academic papers on ArXiv. Provide a search query to find relevant research papers."

    def _run(
        self,
        query: str,
        max_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the search with the given query for academic papers."""
        try:
            # Encode the query for URL
            encoded_query = urllib.parse.quote(query)

            # Construct the API URL
            url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}"

            # Make the request
            with urllib.request.urlopen(url) as response:
                data = response.read().decode('utf-8')

            # Parse the XML response
            root = ET.fromstring(data)

            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }

            # Extract paper information
            papers = []
            entries = root.findall('atom:entry', namespaces)

            for entry in entries:
                paper = {}

                # Title
                title_elem = entry.find('atom:title', namespaces)
                paper['title'] = title_elem.text.strip(
                ) if title_elem is not None and title_elem.text else "No title"

                # Authors
                authors = []
                for author in entry.findall('atom:author', namespaces):
                    name_elem = author.find('atom:name', namespaces)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())
                paper['authors'] = ', '.join(
                    authors) if authors else "No authors listed"

                # Abstract/Summary
                summary_elem = entry.find('atom:summary', namespaces)
                paper['abstract'] = summary_elem.text.strip(
                ) if summary_elem is not None and summary_elem.text else "No abstract"

                # Published date
                published_elem = entry.find('atom:published', namespaces)
                paper['published'] = published_elem.text.strip(
                ) if published_elem is not None and published_elem.text else "No date"

                # ArXiv ID (from the link)
                id_elem = entry.find('atom:id', namespaces)
                if id_elem is not None and id_elem.text:
                    paper['arxiv_id'] = id_elem.text.strip().split('/')[-1]
                else:
                    paper['arxiv_id'] = "Unknown"

                # PDF link
                for link in entry.findall('atom:link', namespaces):
                    if link.get('title') == 'pdf':
                        paper['pdf_url'] = link.get('href')
                        break
                else:
                    paper['pdf_url'] = None

                papers.append(paper)

            # Format the results
            if not papers:
                return f"No papers found for query: {query}"

            result = f"Found {len(papers)} papers for query: {query}\n\n"

            for i, paper in enumerate(papers, 1):
                # Clean and format the title
                title = paper['title'].replace(
                    '\n', ' ').replace('\r', ' ').strip()
                title = ' '.join(title.split())  # Remove extra whitespace

                # Clean abstract
                abstract = paper['abstract'].replace(
                    '\n', ' ').replace('\r', ' ').strip()
                # Remove extra whitespace
                abstract = ' '.join(abstract.split())

                result += f"{i}. {title}\n"
                result += f"   Authors: {paper['authors']}\n"
                # Just the date part
                result += f"   Published: {paper['published'][:10]}\n"
                result += f"   ArXiv ID: {paper['arxiv_id']}\n"
                if paper['pdf_url']:
                    result += f"   PDF: {paper['pdf_url']}\n"

                # Truncate abstract and ensure it's not empty
                abstract_snippet = abstract[:200] if abstract else "No abstract available"
                if len(abstract) > 200:
                    abstract_snippet += "..."
                result += f"   Abstract: {abstract_snippet}\n\n"

            return result.strip()  # Remove any trailing whitespace

        except Exception as e:
            error_msg = f"Error searching ArXiv: {str(e)}"
            return error_msg if error_msg.strip() else "Error occurred while searching ArXiv"


tools = [
    BraveSearchTool(api_key=os.getenv("BRAVE_SEARCH_API_KEY")),
    ArXivSearchTool(),
]
