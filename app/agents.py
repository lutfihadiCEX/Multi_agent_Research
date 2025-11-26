# Multi agents to research and summarize findings into keypoints.
# Some prompt temps might be simple and will be tuned later.
# For exploring LangChain frameworks

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from app.tools import ResearchTools
from app.state import ResearchState, ResearchSource, ResearchFinding
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchAgents:
    """ResearchAgents: A set of agents to collect, analyze, and summarize research. This experiment includes multi agent workflows using LangChain/Ollama. For multi agents learning, embedding and exploring modular llm pipeline"""
    
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.7):
        """
        Initialize agents with local LLM
        
        Args:
            model_name: Ollama model to use
            temperature: LLM temperature (0=deterministic, 1=creative)
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize local LLM
        logger.info(f"Initializing Ollama with model: {model_name}")
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434"
        )
        
        self.tools = ResearchTools()
        logger.info("Research agents initialized")
    
    def researcher_agent(self, state: ResearchState) -> ResearchState:
        """
        RESEARCHER AGENT: Searches for information
        
        Responsibilities:
        - Conduct web and Wikipedia searches
        - Gather multiple sources
        - Store findings in state
        """
        logger.info(f"Researcher Agent: Processing '{state.research_query}'")
        
        state.current_agent = "researcher"
        state.execution_status = "running"
        
        try:
            # Search web
            logger.info("Searching web...")
            web_hits = self.tools.search_web(state.research_query, max_results=10)
            
            # Search Wikipedia
            logger.info("Searching Wikipedia...")
            wiki_hits = self.tools.search_wikipedia(state.research_query, max_results=3)

            if not web_hits and not wiki_hits:
                logger.error("No search results found")
                state.error_message = "Search failed - no results found"
                state.execution_status = "error"
                return state
            
            # Convert to ResearchSource objects
            sources = []
            
            for result in wiki_hits:
                source = ResearchSource(
                    title=result.get("title", ""),
                    content=result.get("summary", ""),
                    url=result.get("url", ""),
                    source_type="wikipedia",
                    reliability_score=0.9
                )
                sources.append(source)
            
            for result in web_hits[:7]:  
                source = ResearchSource(
                    title=result.get("title", ""),
                    content=result.get("summary", ""),
                    url=result.get("link", ""),
                    source_type="web",
                    reliability_score=0.6
                )
                sources.append(source)


            if not sources:
                logger.error("No valid sources after filtering")
                state.error_message = "All search results were filtered as irrelevant"
                state.execution_status = "error"
                return state
            
            state.raw_research = sources
            state.add_to_history(
                "researcher",
                "search",
                f"Found {len(sources)} sources ({len(wiki_hits)} Wikipedia, {len(web_hits)} web)"
            )
            
            logger.info(f"Researcher found {len(sources)} sources")
            
        except Exception as e:
            logger.error(f"[Researcher] Hit an error while researching: {e}")    # Might add retry or fallback search
            state.error_message = f"Error in researcher agent: {e}"
            state.execution_status = "error"
        
        return state
    
    def analyzer_agent(self, state: ResearchState) -> ResearchState:
        """
        ANALYZER AGENT: Processes and summarizes findings
        
        Responsibilities:
        - Summarize each source
        - Extract key findings
        - Organize information
        """
        logger.info("Analyzer Agent: Processing findings")
        
        state.current_agent = "analyzer"
        state.execution_status = "running"
        
        try:
            if not state.raw_research:
                logger.warning("No raw research to analyze")
                state.add_to_history("analyzer", "analyze", "No sources to analyze")
                return state
            
            findings = []

            verification_prompt = PromptTemplate(
                input_variables=["query", "title", "content"],
                template="""Is this source relevant to the query? Answer ONLY 'YES' or 'NO'.

Query: {query}
Source Title: {title}
Content Preview: {content}

Relevant (YES/NO)?"""
            )
            
            for source in state.raw_research:
                verify = verification_prompt.format(
                    query=state.research_query,
                    title=source.title,
                    content=source.content[:300]
                )
                
                relevance = self.llm.invoke(verify).strip().upper()
                
                if "NO" in relevance:
                    logger.warning(f"Filtering irrelevant source: {source.title}")
                    continue
                
                
                summary_prompt = PromptTemplate(
                    input_variables=["content", "topic"],
                    template="""Analyze this research content and extract 2-3 key findings.

Topic: {topic}
Content: {content}

Extract key findings as a concise bullet-point summary:"""
                )
                
                
                prompt = summary_prompt.format(
                    topic=state.research_query,
                    content=source.content[:500]  # Limit content
                )
                
                analysis = self.llm.invoke(prompt)
                
                
                finding = ResearchFinding(
                    topic=state.research_query,
                    finding=analysis,
                    sources=[source],
                    verified=False
                )
                
                findings.append(finding)
            
            if not findings:
                logger.error("No relevant findings after filtering")
                state.error_message = "All sources were deemed irrelevant to the query"
                state.execution_status = "error"
                return state
            
            state.analyzed_findings = findings
            state.add_to_history(
                "analyzer",
                "analyze",
                f"Analyzed {len(findings)} relevant sources and extracted findings"
            )
            
            logger.info(f"Analyzer extracted {len(findings)} findings from relevant sources")
            
        except Exception as e:
            logger.error(f"Analyzer error: {str(e)}")
            state.error_message = f"Analyzer error: {str(e)}"
            state.execution_status = "error"
        
        return state
    
    def critic_agent(self, state: ResearchState) -> ResearchState:
        """
        CRITIC AGENT: Validates and checks accuracy
        
        Responsibilities:
        - Verify findings against sources
        - Check for contradictions
        - Rate source reliability
        """
        logger.info("Critic Agent: Validating findings")
        
        state.current_agent = "critic"
        state.verification_status = "in_progress"
        
        try:
            if not state.analyzed_findings:
                logger.warning("No findings to critique")
                return state
            
            # Validate sources
            all_sources = []
            for finding in state.analyzed_findings:
                all_sources.extend(finding.sources)
            
            validated_sources = self.tools.validate_sources(
                [{"link": s.url, "url": s.url} for s in all_sources]
            )
            
            # Create critique prompt
            findings_text = "\n".join([
                f"- {f.finding[:200]}" for f in state.analyzed_findings[:5]
            ])
            
            critique_prompt = PromptTemplate(
                input_variables=["findings", "topic"],
                template="""As a skeptical research critic, evaluate these findings for accuracy and consistency.
Topic: {topic}

Findings:
{findings}

Provide:
1. Overall reliability assessment
2. Any contradictions or conflicts
3. Quality of evidence
4. Confidence level (0-100%)"""   # Might change template for improving critic response
            )
            
            prompt = critique_prompt.format(
                topic=state.research_query,
                findings=findings_text
            )
            
            criticism = self.llm.invoke(prompt)
            state.criticism = criticism
            
            # Update verification status
            state.verification_status = "completed"
            
            # Mark findings as verified
            for finding in state.analyzed_findings:
                finding.verified = True
            
            state.add_to_history(
                "critic",
                "validate",
                "Validated findings and sources"
            )
            
            logger.info("Critic validated findings")
            
        except Exception as e:
            logger.error(f"[Critic] Hit an error while evaluating source: {e}")
            state.error_message = f"Error in critic agent: {e}"
            state.execution_status = "error"    # Consider fallback or partial processing
        
        return state
    
    def writer_agent(self, state: ResearchState) -> ResearchState:
        """
        WRITER AGENT: Compiles final research report
        
        Responsibilities:
        - Organize findings into report
        - Add sources and citations
        - Write professional summary
        """
        logger.info("Writer Agent: Compiling report")
        
        state.current_agent = "writer"
        
        try:
            if not state.analyzed_findings:
                state.final_report = "No findings to report"
                return state
            
            # Prepare findings text
            findings_text = "\n\n".join([
                f"**Finding {i+1}:**\n{f.finding}\n*Source: {f.sources[0].title if f.sources else 'Unknown'}*"
                for i, f in enumerate(state.analyzed_findings[:10])
            ])
            
            # Create report prompt
            report_prompt = PromptTemplate(
                input_variables=["topic", "findings", "criticism"],
                template="""Write a professional research report based on these findings.

Topic: {topic}

Findings:
{findings}

Critic's Assessment:
{criticism}

Write a comprehensive report with:
1. Executive Summary
2. Key Findings
3. Analysis
4. Conclusion
5. Recommendation for Further Research

Make it concise and professional.""" # Currently using standard verbose prompt template
            )
            
            prompt = report_prompt.format(
                topic=state.research_query,
                findings=findings_text,
                criticism=state.criticism
            )
            
            report = self.llm.invoke(prompt)
            state.final_report = report
            
            # Add metadata
            state.report_metadata = {
                "query": state.research_query,
                "sources_used": len(state.raw_research),
                "findings_extracted": len(state.analyzed_findings),
                "verification_completed": state.verification_status == "completed"
            }
            
            state.execution_status = "completed"
            state.add_to_history(
                "writer",
                "report",
                "Compiled final research report"
            )
            
            logger.info("Writer compiled final report")
            
        except Exception as e:
            logger.error(f"[Writer] Hit an error while compiling and reporting source: {e}")
            state.error_message = f"[Writer] Error in writer reporting agent: {e}"
            state.execution_status = "error"  # Consider fallback or partial processing
        
        return state
    
    def execute_workflow(self, query: str) -> ResearchState:
        """
        Execute the complete research workflow
        All agents work in sequence
        
        Args:
            query: Research query
            
        Returns:
            Final research state with report
        """
        logger.info(f"Starting research workflow for: {query}")
        
        # Initialize state
        state = ResearchState(research_query=query)
        
        # Execute agents in sequence
        logger.info("Step 1/4: Researcher Agent")
        state = self.researcher_agent(state)
        
        logger.info("Step 2/4: Analyzer Agent")
        state = self.analyzer_agent(state)
        
        logger.info("Step 3/4: Critic Agent")
        state = self.critic_agent(state)
        
        logger.info("Step 4/4: Writer Agent")
        state = self.writer_agent(state)
        
        logger.info("Workflow completed")
        return state


if __name__ == "__main__":
    # Test agents, Ollama must be running via 'ollama serve' before testing.
    agents = ResearchAgents(model_name="llama3.2")
    
    # Run research
    query = "What are the latest developments in quantum computing?"
    print(f"Researching: {query}\n")
    
    final_state = agents.execute_workflow(query)
    
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    print(final_state.final_report)