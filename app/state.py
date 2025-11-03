"""
State management for multi agent research system.
Tracks research queries, raw sources, findings, and agent actions.

Built for experimenting with multi agent LLM workflows.
Includes notes for exploring and debugging
"""

from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime
import json


class ResearchSource(BaseModel):
    """Represents one research source (From Researcher)"""
    title: str
    content: str
    url: str
    reliability_score: float = 0.5
    source_type: str = "web"  # web, wikipedia, paper, etc.
    
    class Config:
        arbitrary_types_allowed = True


class ResearchFinding(BaseModel):
    """Represents one research finding (From analyzer)"""
    topic: str
    finding: str
    sources: List[ResearchSource]
    verified: bool = False
    
    class Config:
        arbitrary_types_allowed = True


class ResearchState(BaseModel):
    """
    Shared state for all multi agents
    Each agent reads and modifies this state (Acts as shared memory)
    """
    # Initial query
    research_query: str = ""
    
    # Raw research from researcher agent
    raw_research: List[ResearchSource] = []
    
    # Analyzed findings from analyzer agent
    analyzed_findings: List[ResearchFinding] = []
    
    # Critic agent feedback
    criticism: str = ""
    contradictions_found: List[str] = []
    verification_status: str = "not_started"  # not_started, in_progress, completed
    
    # Final report from writer agent
    final_report: str = ""
    report_metadata: Dict = {}
    
    # Processing metadata
    conversation_history: List[Dict] = []
    current_agent: str = ""
    execution_status: str = "idle"  # idle, running, completed, error
    error_message: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.created_at is None:
            # Help track workflow experiments, for easier debugging in multi runs
            self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_to_history(self, agent: str, action: str, result: str):
        """Log agent action to conversation history. Added timestamps for incase I need to debug workflow order."""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "result": result
        })
        self.updated_at = datetime.now()    # Log every action, so easier to trace bugs in agent outputs for me
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary"""
        return {
            "research_query": self.research_query,
            "raw_research": [source.dict() for source in self.raw_research],
            "analyzed_findings": [finding.dict() for finding in self.analyzed_findings],
            "criticism": self.criticism,
            "contradictions_found": self.contradictions_found,
            "verification_status": self.verification_status,
            "final_report": self.final_report,
            "report_metadata": self.report_metadata,
            "conversation_history": self.conversation_history,
            "execution_status": self.execution_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    def to_json(self) -> str:
        """Convert state to JSON. Help to debug intermediate outputs during workflow testing."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save_to_file(self, filepath: str):
        """Save state to JSON file"""
        with open(filepath, 'w') as f:
            f.write(self.to_json()) # TODO: Add better error handling when loading corrupted JSON
    
    @staticmethod
    def load_from_file(filepath: str) -> 'ResearchState':
        """Load state from JSON file. Assume file structure matches the current ResearchState model."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return ResearchState(**data)   # If you change the model, old files might fail to load properly


# Example usage and testing
if __name__ == "__main__":
    # Create initial state
    state = ResearchState(
        research_query="What are the latest developments in AI safety?"
    )
    
    print("Initial State:")
    print(state.to_json())
    
    # Add sample source manually for quick testing
    source = ResearchSource(
        title="AI Safety Research",
        content="Latest developments in AI safety...",
        url="https://example.com",
        reliability_score=0.9
    )
    
    state.raw_research.append(source)
    state.add_to_history("researcher", "search", "Found 5 sources")
    
    print("\n\nUpdated State:")
    print(state.to_json())