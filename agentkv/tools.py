"""
AgentKV Tools — Web search and knowledge acquisition for autonomous agents.

Tools the LLM can invoke to gather information and store it in AgentKV.
"""
import time
from typing import List, Dict, Optional
from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
    
    Returns:
        List of {"title": ..., "url": ..., "snippet": ...} dicts.
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        results.append({
            "title": "Search Error",
            "url": "",
            "snippet": f"Web search failed: {e}",
        })
    return results


def format_search_results(results: List[Dict[str, str]]) -> str:
    """Format search results into a readable string for the LLM."""
    if not results:
        return "No search results found."
    
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"{i}. {r['title']}\n   {r['snippet']}\n   Source: {r['url']}")
    return "\n\n".join(parts)


# Tool definitions for the LLM (OpenAI-style function calling format)
TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": "Search the web for current information. Use this when you need up-to-date facts, news, or information you don't have in memory.",
        "parameters": {
            "query": "The search query string"
        }
    }
]

TOOL_PROMPT = """You have access to the following tools:

1. **web_search(query)** — Search the web for current information. Use when you need facts you don't know or need up-to-date info.

To use a tool, respond with EXACTLY this format (no other text before it):
TOOL_CALL: web_search("your search query here")

Rules:
- Only use a tool if you genuinely need information you don't have.
- After receiving tool results, synthesize a natural answer.
- You can only call ONE tool per turn.
- If you don't need a tool, just respond normally."""

# Regex to detect tool calls in LLM output
import re
TOOL_CALL_PATTERN = re.compile(r'TOOL_CALL:\s*web_search\("([^"]+)"\)', re.IGNORECASE)
