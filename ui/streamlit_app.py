"""
Streamlit UI for Multi-Agent Research System
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents import ResearchAgents
from app.state import ResearchState

# Page configu
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .agent-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .finding-box {
        background-color: #e8f4f8;
        padding: 12px;
        border-radius: 5px;
        margin: 8px 0;
    }
    .source-box {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🔬 Multi-Agent Research Assistant</p>', unsafe_allow_html=True)
st.markdown("*Powered by collaborative AI agents working together to research topics*")

# Sidebar config
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model = st.selectbox(
        "Select LLM Model",
        ["llama3.2", "mistral", "gemma2:9b"],
        help="Choose which Ollama model to use"
    )
    
    # Temperature control
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower = deterministic, Higher = creative"
    )
    
    st.info(f"Model: **{model}** | Temp: **{temperature}**")
    
    st.divider()
    
    st.header("📊 Instructions")
    st.markdown("""
    1. Enter a research topic
    2. Click "Start Research"
    3. Watch agents collaborate:
       - 🔍 Researcher: Finds sources
       - 📊 Analyzer: Processes findings
       - ✅ Critic: Validates accuracy
       - 📝 Writer: Compiles report
    """)

# Initialize session
if 'research_state' not in st.session_state:
    st.session_state.research_state = None
if 'agents' not in st.session_state:
    st.session_state.agents = None
if 'research_history' not in st.session_state:
    st.session_state.research_history = []

# Main content
st.header("🔎 Research Query")

# Research input
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "What would you like to research?",
        placeholder="e.g., Latest developments in AI safety, Quantum computing breakthroughs, etc.",
        label_visibility="collapsed"
    )

with col2:
    research_button = st.button("🚀 Start Research", type="primary", use_container_width=True)

# Execute research
if research_button and query:
    with st.spinner("🤖 Initializing agents..."):
        try:
            # Initialize agents
            agents = ResearchAgents(model_name=model, temperature=temperature)
            st.session_state.agents = agents
            
            # Execute research workflow
            progress_placeholder = st.empty()
            
            with st.spinner("🔬 Research in progress..."):
                # Show agent workflow
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="agent-box"><b>🔍 Researcher</b><br/>Searching sources...</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="agent-box"><b>📊 Analyzer</b><br/>Processing...</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="agent-box"><b>✅ Critic</b><br/>Validating...</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="agent-box"><b>📝 Writer</b><br/>Compiling...</div>', unsafe_allow_html=True)
                
                # Run the workflow
                final_state = agents.execute_workflow(query)
                st.session_state.research_state = final_state
                
                # Add to history
                st.session_state.research_history.append({
                    "timestamp": datetime.now(),
                    "query": query,
                    "state": final_state
                })
            
            st.success("✅ Research completed!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure Ollama is running: `ollama serve`")

# Display results
if st.session_state.research_state:
    state = st.session_state.research_state
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Final Report",
        "🔍 Raw Research",
        "📊 Analysis",
        "✅ Validation",
        "📋 History"
    ])
    
    with tab1:
        st.header("Research Report")
        if state.final_report:
            st.markdown(state.final_report)
        else:
            st.info("No report generated yet")
        
        # Download report
        if state.final_report:
            st.download_button(
                label="📥 Download Report",
                data=state.final_report,
                file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with tab2:
        st.header("Raw Research Sources")

        # For debugging
        if state.error_message:
            st.error(f"⚠️ Error: {state.error_message}")

        if state.raw_research:
            st.success(f"✅ Found {len(state.raw_research)} sources")


        # Listing source for checking
            col1, col2 = st.columns(2)
            with col1:
                wiki_count = len([s for s in state.raw_research if s.source_type == "wikipedia"])
                st.metric("Wikipedia Sources", wiki_count)
            with col2:
                web_count = len([s for s in state.raw_research if s.source_type == "web"])
                st.metric("Web Sources", web_count)
            
            st.divider()

            for i, source in enumerate(state.raw_research, 1):
                with st.expander(f"Source {i}: {source.title[:80]}..."):
                    st.markdown(f"**Title:** {source.title}")
                    st.markdown(f"**URL:** {source.url}")
                    st.markdown(f"**Type:** {source.source_type}")
                    st.markdown(f"**Reliability:** {source.reliability_score:.0%}")
                    st.markdown("---")
                    st.markdown(source.content)
        else:
            st.warning("⚠️ No sources retrieved - search may have failed")
    
    with tab3:
        st.header("Analyzed Findings")
        if state.analyzed_findings:
            for i, finding in enumerate(state.analyzed_findings, 1):
                with st.container():
                    st.markdown(f"### Finding {i}")
                    st.markdown(finding.finding)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Verified", "✅" if finding.verified else "❌")
                    with col2:
                        st.metric("Sources", len(finding.sources))
                    
                    st.divider()
        else:
            st.info("No findings analyzed yet")
    
    with tab4:
        st.header("Critic Validation")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Status", state.verification_status.upper())
        with col2:
            st.metric("Sources Used", len(state.raw_research))
        with col3:
            st.metric("Findings", len(state.analyzed_findings))
        
        st.divider()
        st.subheader("Critic Assessment")
        if state.criticism:
            st.markdown(state.criticism)
        else:
            st.info("No criticism available")
        
        if state.contradictions_found:
            st.warning(f"⚠️ Contradictions found: {len(state.contradictions_found)}")
            for contradiction in state.contradictions_found:
                st.markdown(f"- {contradiction}")
    
    with tab5:
        st.header("Conversation History")
        if state.conversation_history:
            for entry in state.conversation_history:
                with st.expander(f"{entry['agent']} - {entry['action']}"):
                    st.markdown(f"**Timestamp:** {entry['timestamp']}")
                    st.markdown(f"**Agent:** {entry['agent']}")
                    st.markdown(f"**Action:** {entry['action']}")
                    st.markdown(f"**Result:** {entry['result']}")
        else:
            st.info("No history")
    
    st.divider()
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save as JSON"):
            filename = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            state.save_to_file(filename)
            st.success(f"Saved to {filename}")
    
    with col2:
        if st.button("🗑️ Clear"):
            st.session_state.research_state = None
            st.rerun()

# Research history sidebar
with st.sidebar:
    st.divider()
    st.header("📚 Research History")
    
    if st.session_state.research_history:
        for i, entry in enumerate(reversed(st.session_state.research_history), 1):
            if st.button(f"{i}. {entry['query'][:40]}..."):
                st.session_state.research_state = entry['state']
                st.rerun()
    else:
        st.info("No research history yet")

# Footer
st.divider()
st.caption("🚀 Multi-Agent Research System | Powered by LangGraph, Ollama & Streamlit")