"""
Web Search Tool

This module provides a mock web search tool that agents can use.
In a real implementation, this would connect to a search API like Google,
Bing, or DuckDuckGo.

For learning purposes, this is a mock implementation that returns
simulated search results.
"""

from typing import List, Dict, Any, Optional
import random
from utils.agent_logging import logger
from tools.registry import tool_registry, ToolSafetyLevel, ToolParameter


# Mock search results database
MOCK_SEARCH_RESULTS = {
    "python": [
        {
            "title": "Python.org - Official Python Website",
            "url": "https://www.python.org",
            "snippet": "Python is a programming language that lets you work quickly and integrate systems more effectively."
        },
        {
            "title": "Python Tutorial - W3Schools",
            "url": "https://www.w3schools.com/python",
            "snippet": "Learn Python programming with our comprehensive tutorial covering basics to advanced topics."
        },
        {
            "title": "Python Documentation",
            "url": "https://docs.python.org",
            "snippet": "Official Python documentation with tutorials, library references, and guides."
        }
    ],
    "ai": [
        {
            "title": "Artificial Intelligence - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "snippet": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence."
        },
        {
            "title": "AI Research Papers",
            "url": "https://arxiv.org/list/cs.AI/recent",
            "snippet": "Latest research papers on artificial intelligence and machine learning."
        }
    ],
    "weather": [
        {
            "title": "Weather.com - Weather Forecast",
            "url": "https://weather.com",
            "snippet": "Get the latest weather forecasts and conditions for your location."
        },
        {
            "title": "National Weather Service",
            "url": "https://www.weather.gov",
            "snippet": "Official weather forecasts and warnings from the National Weather Service."
        }
    ],
    "default": [
        {
            "title": "Search Result 1",
            "url": "https://example.com/result1",
            "snippet": "This is a mock search result. In a real implementation, this would be actual web search results."
        },
        {
            "title": "Search Result 2",
            "url": "https://example.com/result2",
            "snippet": "Another mock search result demonstrating the search tool functionality."
        },
        {
            "title": "Search Result 3",
            "url": "https://example.com/result3",
            "snippet": "Additional mock search result for demonstration purposes."
        }
    ]
}


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web for information.
    
    This is a mock implementation that returns simulated search results.
    In a real system, this would call an actual search API.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        List of search results, each containing:
        - title: Result title
        - url: Result URL
        - snippet: Brief description/snippet
        
    Example:
        >>> results = search_web("python programming")
        >>> print(results[0]['title'])
        'Python.org - Official Python Website'
    """
    logger.info(f"Searching web for: '{query}'")
    
    # Normalize query for lookup
    query_lower = query.lower().strip()
    
    # Check if we have mock results for this query
    results = None
    for key in MOCK_SEARCH_RESULTS:
        if key in query_lower:
            results = MOCK_SEARCH_RESULTS[key].copy()
            break
    
    # Use default results if no match
    if results is None:
        results = MOCK_SEARCH_RESULTS["default"].copy()
        # Add query to snippets to make it more realistic
        for result in results:
            result["snippet"] = f"Results for '{query}': {result['snippet']}"
    
    # Limit results
    results = results[:max_results]
    
    # Add some randomness to make it feel more realistic
    if len(results) > 1:
        random.shuffle(results)
    
    logger.observation(f"Found {len(results)} search results")
    
    return results


def search_web_advanced(
    query: str,
    max_results: int = 5,
    language: str = "en",
    date_range: Optional[str] = None
) -> Dict[str, Any]:
    """
    Advanced web search with additional options.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        language: Language code for results (default: "en")
        date_range: Optional date range filter (e.g., "past_year")
        
    Returns:
        Dictionary containing:
        - results: List of search results
        - total_results: Estimated total results
        - query_info: Query metadata
    """
    logger.info(f"Advanced web search: '{query}' (language: {language})")
    
    # Get basic results
    results = search_web(query, max_results)
    
    # Simulate total results count
    total_results = random.randint(1000, 100000)
    
    return {
        "results": results,
        "total_results": total_results,
        "query_info": {
            "query": query,
            "language": language,
            "date_range": date_range,
            "results_returned": len(results)
        }
    }


# Register tools with the registry
def register_search_tools():
    """Register search tools with the tool registry."""
    
    # Register basic search tool
    tool_registry.register(
        name="search_web",
        description="Search the web for information. Returns a list of search results with titles, URLs, and snippets.",
        function=search_web,
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="The search query string",
                required=True
            ),
            ToolParameter(
                name="max_results",
                type="number",
                description="Maximum number of results to return (default: 5)",
                required=False
            )
        ],
        safety_level=ToolSafetyLevel.SAFE,
        category="search"
    )
    
    # Register advanced search tool
    tool_registry.register(
        name="search_web_advanced",
        description="Advanced web search with language and date filtering options.",
        function=search_web_advanced,
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="The search query string",
                required=True
            ),
            ToolParameter(
                name="max_results",
                type="number",
                description="Maximum number of results to return (default: 5)",
                required=False
            ),
            ToolParameter(
                name="language",
                type="string",
                description="Language code for results (default: 'en')",
                required=False
            ),
            ToolParameter(
                name="date_range",
                type="string",
                description="Optional date range filter (e.g., 'past_year')",
                required=False
            )
        ],
        safety_level=ToolSafetyLevel.SAFE,
        category="search"
    )
    
    logger.info("Search tools registered")


# Auto-register tools when module is imported
register_search_tools()


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Web Search Tool Demo")
    print("=" * 60)
    
    # Test basic search
    print("\n[Test 1] Basic Search")
    results = search_web("python programming", max_results=3)
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. {result['title']}")
        print(f"     URL: {result['url']}")
        print(f"     {result['snippet']}")
    
    # Test advanced search
    print("\n[Test 2] Advanced Search")
    advanced_results = search_web_advanced(
        query="artificial intelligence",
        max_results=2,
        language="en"
    )
    print(f"Total results: {advanced_results['total_results']}")
    print(f"Returned: {len(advanced_results['results'])} results")
    
    # Test tool registry integration
    print("\n[Test 3] Tool Registry Integration")
    from tools.registry import tool_registry
    from utils.schemas import ToolCall
    
    tool_call = ToolCall(
        id="test_1",
        name="search_web",
        arguments={"query": "weather forecast", "max_results": 2}
    )
    
    result = tool_registry.execute_tool(tool_call)
    if result.is_success:
        print(f"Tool executed successfully!")
        print(f"Results: {len(result.result)} items")
    else:
        print(f"Tool execution failed: {result.error}")
    
    print("\nSearch tool demo complete!")

