# web_tools.py
import os

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from src.researchit_ai.utils.config_loader import get_config


class WebSearchInput(BaseModel):
    query: str = Field(description="A specific, self-contained web search query.")


class WebResearcherTools:
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        self.tavily = TavilyClient(api_key=api_key)

        cfg = get_config().tool["tavily"]
        self.search_depth = cfg["search_depth"]
        self.max_results = cfg["max_results"]
        self.include_answer = cfg["include_answer"]
        self.include_raw_content = cfg["include_raw_content"]
        self.include_domains = cfg.get("include_domains") or None
        self.exclude_domains = cfg.get("exclude_domains") or None

    def web_search_tavily(self, query: str) -> list[dict]:
        """Search the internet for up-to-date information."""
        response = self.tavily.search(
            query=query,
            search_depth=self.search_depth,
            max_results=self.max_results,
            include_answer=self.include_answer,
            include_raw_content=self.include_raw_content,
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
        )
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
            }
            for r in response.get("results", [])
        ]

    def get_tools(self):
        return [
            StructuredTool.from_function(
                func=self.web_search_tavily,
                name="web_search",
                description="Search the web for current, external information.",
                args_schema=WebSearchInput,
            )
        ]