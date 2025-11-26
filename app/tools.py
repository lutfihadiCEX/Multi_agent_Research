"""
Tools for research agents
Agents can use these tools to search and gather information
"""

from duckduckgo_search import DDGS
import wikipedia
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchTools:
    """Collection of tools for research agents"""
    
    @staticmethod
    def search_web(query: str, max_results: int = 5) -> List[Dict]:
        """
        Search the web using DuckDuckGo
        
        Args:
            query: Search query
            max_results: Number of results to return
            
        Returns:
            List of search results with title, body, and link
        """
        logger.info(f"Looking up '{query}' on the web... fingers crossed!")
        try:
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=max_results))

            if not results:
                logger.warning("No web results found")
                return []
            
            formatted_results = []
            for result in results:
                title = result.get("title", "")
                body = result.get("body", "")
                link = result.get("href", "")

                if len(body) < 50:  # Skip very short results
                    continue


                if "windows" in title.lower() and "windows" not in query.lower():
                    logger.warning(f"Filtering potentially irrelevant result: {title}")
                    continue


                formatted_results.append({
                    "title": title,
                    "summary": body[:500],  # Limit to 500 chars
                    "link": link
                })

            
            
            logger.info(f"Found {len(formatted_results)} relevant web results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return []
    
    @staticmethod
    def search_wikipedia(query: str, max_results: int = 3) -> List[Dict]:
        """
        Search Wikipedia for information
        
        Args:
            query: Search query
            max_results: Number of results to return
            
        Returns:
            List of Wikipedia summaries
        """
        logger.info(f"Checking Wikipedia for '{query}'... let's see what we can dig up!")
        try:
            results = wikipedia.search(query, results=max_results)
            
            formatted_results = []
            for page_title in results:
                try:
                    page = wikipedia.page(page_title)
                    formatted_results.append({
                        "title": page.title,
                        "summary": page.summary[:500],  # Limit to 500 chars
                        "url": page.url
                    })
                except wikipedia.exceptions.DisambiguationError:
                    logger.warning(f"Oops! '{page_title}' is a disambiguation page, skipping it...")
                    continue
            
            logger.info(f"Wikipedia search done. Got {len(formatted_results)} pages to look at!")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Wikipedia search error: {str(e)}")
            return []
    
    @staticmethod
    def summarize_text(text: str, max_length: int = 300) -> str:
        """
        Summarize long text (simple extraction-based)
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            
        Returns:
            Summarized text
        """
        logger.info("Summarizing text... let's make it concise and readable ✨")
        
        # Quick summary: taking the first few sentences to keep it snap!
        sentences = text.split('.')
        summary = ""
        
        for sentence in sentences:
            if len(summary) < max_length:
                summary += sentence.strip() + ". "
            else:
                break
        
        return summary.strip()
    
    @staticmethod
    def validate_sources(sources: List[Dict]) -> List[Dict]:
        """
        Validate and score sources for reliability
        
        Args:
            sources: List of sources to validate
            
        Returns:
            Validated sources with reliability score
        """
        logger.info(f"Validating {len(sources)} sources... hoping they are reliable!") 
        # Quick check: see which sources look trustworthy based on known good domains
        
        validated = []
        for source in sources:
            # Simple validation: check for common reliable domains
            reliable_domains = [
                "wikipedia.org",
                "github.com",
                "medium.com",
                "arxiv.org",
                ".edu",
                ".gov",
                "research",
                "journal",
                "conference"
            ]
            
            url = source.get("link", "") or source.get("url", "")
            reliability_score = 0.5  # Default medium score
            
            for domain in reliable_domains:
                if domain in url.lower():
                    reliability_score = 0.9
                    break
            
            source["reliability_score"] = reliability_score
            validated.append(source)
        
        # Sort by reliability
        validated.sort(key=lambda x: x.get("reliability_score", 0), reverse=True)
        
        return validated


# Tool definitions for LangGraph
SEARCH_TOOLS = {
    "web_search": ResearchTools.search_web,
    "wikipedia_search": ResearchTools.search_wikipedia,
    "summarize": ResearchTools.summarize_text,
    "validate_sources": ResearchTools.validate_sources
}


if __name__ == "__main__":
    # Test tools
    tools = ResearchTools()
    
    print("Testing Web Search...")
    web_results = tools.search_web("quantum computing 2024")
    print(f"Found {len(web_results)} results")
    
    print("\nTesting Wikipedia Search...")
    wiki_results = tools.search_wikipedia("artificial intelligence")
    print(f"Found {len(wiki_results)} results")
    
    print("\nTesting Source Validation...")
    all_sources = web_results + wiki_results
    validated = tools.validate_sources(all_sources)
    print(f"Validated {len(validated)} sources")