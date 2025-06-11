from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os
import requests


class TavilySearchInput(BaseModel):
    """Input schema for TavilySearchTool."""
    query: str = Field(..., description="The search query to find relevant information on the web.")
    max_results: int = Field(default=5, description="Maximum number of search results to return (default: 5).")


class TavilySearchTool(BaseTool):
    name: str = "Tavily Search"
    description: str = (
        "A web search tool powered by Tavily API. Use this to search for current information, "
        "news, facts, or any topic on the internet. Provide a clear search query to get relevant results."
    )
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str, max_results: int = 10) -> str:
        """
        Execute a web search using Tavily API.
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return
            
        Returns:
            Formatted search results as a string
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY environment variable not set. Please set your Tavily API key."
        
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "include_images": False,
            "include_raw_content": False,
            "max_results": max_results
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the results
            results = []
            
            # Add the answer if available
            if data.get("answer"):
                results.append(f"**Answer:** {data['answer']}\n")
            
            # Add search results
            if data.get("results"):
                results.append("**Search Results:**")
                for i, result in enumerate(data["results"][:max_results], 1):
                    title = result.get("title", "No title")
                    url = result.get("url", "No URL")
                    content = result.get("content", "No content available")
                    
                    results.append(f"\n{i}. **{title}**")
                    results.append(f"   URL: {url}")
                    results.append(f"   Content: {content[:300]}{'...' if len(content) > 300 else ''}")
            
            return "\n".join(results) if results else "No results found for the given query."
            
        except requests.exceptions.RequestException as e:
            return f"Error making request to Tavily API: {str(e)}"
        except Exception as e:
            return f"Error processing Tavily search: {str(e)}"
