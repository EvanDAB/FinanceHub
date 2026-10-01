import os
import streamlit as st
import sys
from pathlib import Path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Initialize API keys from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY") or OPENAI_API_KEY
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
FRED_API_KEY = os.environ.get("FRED_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# LangChain's ChatOpenAI expects OPENAI_API_KEY specifically.
# Support the legacy OPEN_AI_API_KEY name used elsewhere in this project.
if OPEN_AI_API_KEY and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = OPEN_AI_API_KEY

# Import required classes
# from agents.idea_builder_goal_planner_agent import InvestmentIdea
# from util.idea.idea_builder_ui import render_idea_builder_tab

# Validate required API keys
required_keys = {
    "OPEN_AI_API_KEY": OPEN_AI_API_KEY,
    "FINNHUB_API_KEY": FINNHUB_API_KEY,
}

missing_keys = [key for key, value in required_keys.items() if not value]
if missing_keys:
    st.error(f"Missing required API keys: {', '.join(missing_keys)}")
    st.stop()

class FinanceHub:
    def __init__(self):
        try:
            # Initialize news agent components
            from agents.news_agent import NewsAgent
            self.news_agent = NewsAgent()
            self.news_agent.initialize_components()
            self.news_chain = self.news_agent.create_chain()

            # Initialize market data aggregator
            from util.market.market_aggregator import MarketDataAggregator
            self.market_aggregator = MarketDataAggregator()
            self.market_indicators = self.market_aggregator.fetch_market_data()
            
            # Initialize financial literacy agent
            from agents.fin_literacy_agent import FinancialLiteracyBot
            self.literacy_bot = FinancialLiteracyBot()
            
            # Initialize portfolio agent
            from agents.portfolio_agent import PortfolioAgent
            self.portfolio_agent = PortfolioAgent()
            # Load initial portfolio to ensure everything is initialized
            self.portfolio_agent.load_portfolio()
            # Note: Beta calculation is deferred until dashboard tab is accessed
            
            # # Initialize idea builder agent
            from agents.idea_builder_goal_planner_agent import IdeaBuilderAgent, InvestmentIdea
            from util.idea.idea_storage import IdeaStorage
            self.idea_builder = IdeaBuilderAgent()
            self.idea_storage = IdeaStorage()
            
        except Exception as e:
            st.error(f"Failed to initialize agents: {str(e)}")
            raise
            
    def analyze_news(self, query):
        """Analyze market news based on a query."""
        try:
            # First use the retriever to get relevant info
            retrieved_info = self.news_agent.retriever_tool.invoke({"query": query})
            
            # Then pass that info to the analysis chain
            response = self.news_chain.invoke({
                "input": query,
                "retrieved_info": retrieved_info
            })
            return response
        except Exception as e:
            st.error(f"News analysis failed: {str(e)}")
            return None

# Configure page with minimal UI
st.set_page_config(
    page_title="Finance AI Hub",
    layout="wide"
)

# Initialize hub silently if needed
if 'hub' not in st.session_state:
    st.session_state.hub = FinanceHub()
    st.session_state.messages = []

# Display only the title and tabs
st.title("Finance AI Hub")

# Render only one page per rerun. Streamlit tabs execute all tab blocks on
# every rerun, which makes step navigation in Idea Builder feel slow because
# heavy work from other pages still runs in the background.
tab_options = [
    "📊 Dashboard",
    "📊 Market Analysis",
    "📚 Financial/Tax Education",
    "💼 Portfolio Management",
    "💡 Idea Builder",
]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = tab_options[0]

st.session_state.active_tab = st.segmented_control(
    "Navigation",
    options=tab_options,
    selection_mode="single",
    default=st.session_state.active_tab,
    key="active_tab_selector",
    label_visibility="collapsed",
)

from ui.dashboard_tab import render_dashboard_tab
from ui.market_tab import render_market_tab
from ui.education_tab import render_education_tab
from ui.portfolio_tab import render_portfolio_tab
from ui.ideas_tab import render_ideas_tab

# Dashboard Tab
if st.session_state.active_tab == "📊 Dashboard":
    render_dashboard_tab()

# Market Analysis Tab
elif st.session_state.active_tab == "📊 Market Analysis":
    render_market_tab()

# Financial Education Tab
elif st.session_state.active_tab == "📚 Financial/Tax Education":
    render_education_tab()

# Portfolio Management Tab
elif st.session_state.active_tab == "💼 Portfolio Management":
    render_portfolio_tab()

# Idea Builder Tab
else:
    render_ideas_tab()

# Footer
st.markdown("---")
st.caption("Finance AI Hub - Powered by Multiple Specialized Agents")