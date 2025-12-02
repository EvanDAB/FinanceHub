import os
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Initialize API keys from environment
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
FRED_API_KEY = os.environ.get("FRED_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

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

            # Initialize market regime agent
            from agents.market_agent import MarketRegimeRAG
            self.market_agent = MarketRegimeRAG()
            self.market_indicators = self.market_agent.fetch_market_data()
            
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
            # from agents.idea_builder_goal_planner_agent import IdeaBuilderAgent, InvestmentIdea
            # from util.idea.idea_storage import IdeaStorage
            # self.idea_builder = IdeaBuilderAgent()
            # self.idea_storage = IdeaStorage()
            
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

def format_indicator_metric(indicator_name: str, current_value: str, trend: str, impact: str) -> tuple:
    """
    Format indicator metrics with emojis and trend symbols.
    Returns a tuple of (formatted_label, value, delta_color)
    """
    # Define emojis based on impact keywords
    impact_lower = impact.lower()
    if any(word in impact_lower for word in ['positive', 'strong', 'growth', 'improve', 'good', 'bullish']):
        status_emoji = "🟢"  # Green circle for positive
    elif any(word in impact_lower for word in ['caution', 'mixed', 'neutral', 'moderate', 'stable']):
        status_emoji = "🟡"  # Yellow circle for caution
    else:
        status_emoji = "🔴"  # Red circle for negative

    # Define trend symbols
    if trend.lower() == 'growing':
        trend_symbol = "📈"
        delta_color = "normal"  # Green
    elif trend.lower() == 'stable':
        trend_symbol = "➡️"
        delta_color = "off"  # Gray
    else:  # Decreasing
        trend_symbol = "📉"
        delta_color = "inverse"  # Red

    # Format the label with emoji
    formatted_label = f"{status_emoji} {indicator_name}"
    
    return formatted_label, trend_symbol, delta_color

# Initialize hub silently if needed
if 'hub' not in st.session_state:
    st.session_state.hub = FinanceHub()
    st.session_state.messages = []

# Display only the title and tabs
st.title("Finance AI Hub")

# Create tabs for navigation
# tab_dashboard,  tab_market, tab_education, tab_portfolio, tab_ideas = st.tabs([
#     "📊 Dashboard",
#     "📊 Market Analysis",
#     "📚 Financial/Tax Education",
#     "💼 Portfolio Management",
#     "💡 Idea Builder"
# ])
tab_dashboard,  tab_market, tab_education, tab_portfolio = st.tabs([
    "📊 Dashboard",
    "📊 Market Analysis",
    "📚 Financial/Tax Education",
    "💼 Portfolio Management"
])

# Dashboard Tab
with tab_dashboard:
    # Add a debug button to clear cached calculations (hidden in expander)
    with st.expander("🔧 Debug Options"):
        if st.button("Clear Cached Calculations (Beta & YTD)"):
            if 'portfolio_beta' in st.session_state:
                del st.session_state.portfolio_beta
            if 'ytd_performance' in st.session_state:
                del st.session_state.ytd_performance
            st.success("Cached values cleared! Refresh will recalculate.")
            st.rerun()
    
    # Fetch live data for dashboard metrics
    try:
        # Get portfolio metrics
        portfolio_df = st.session_state.hub.portfolio_agent.load_portfolio()
        stock_data = st.session_state.hub.portfolio_agent.fetch_stock_data(portfolio_df['ticker'].unique())
        _, portfolio_metrics = st.session_state.hub.portfolio_agent.calculate_portfolio_metrics(portfolio_df, stock_data)
        
        # Calculate portfolio beta (cached to avoid repeated calculations on every rerun)
        if 'portfolio_beta' not in st.session_state:
            # Calculate beta on first access to dashboard
            with st.spinner("Calculating portfolio beta..."):
                try:
                    st.session_state.portfolio_beta = st.session_state.hub.portfolio_agent.calculate_portfolio_beta(portfolio_df)
                except Exception as e:
                    st.warning(f"Beta calculation encountered an issue: {str(e)}. Using default value.")
                    st.session_state.portfolio_beta = 1.0
        
        portfolio_beta = st.session_state.get('portfolio_beta', 1.0)
        
        # Calculate YTD performance (cached to avoid repeated calculations on every rerun)
        if 'ytd_performance' not in st.session_state:
            with st.spinner("Calculating YTD performance..."):
                try:
                    st.session_state.ytd_performance = st.session_state.hub.portfolio_agent.calculate_ytd_performance(portfolio_df)
                except Exception as e:
                    st.warning(f"YTD calculation encountered an issue: {str(e)}. Using default values.")
                    st.session_state.ytd_performance = {
                        'portfolio_ytd': 0,
                        'spy_ytd': 0,
                        'outperformance': 0
                    }
        
        ytd_performance = st.session_state.ytd_performance
        
        # Get market data
        market_data = st.session_state.hub.market_indicators
         # Extract VIX from market data
        import util.market.retrieve_vix as vix_util
        vix_display, vix_trend = vix_util.retrieve_vix_from_market_data(market_data)
        
    except Exception as e:
        st.error(f"Error loading dashboard metrics: {str(e)}")
        portfolio_metrics = {'total_value': 0, 'daily_change': 0}
        market_data = []
        portfolio_beta = 1.0
    
    # Create columns for key metrics
    with st.container():
        portfolio_val_col, today_pl_col, vix_col, beta_col = st.columns(4)

        with portfolio_val_col:
            total_value = portfolio_metrics.get('total_value', 0)
            st.metric(
                label="Portfolio Value",
                value=f"${total_value:,.2f}",
                delta=None
            )

        with today_pl_col:
            daily_change = portfolio_metrics.get('daily_change', 0)
            daily_change_dollars = total_value * (daily_change / 100)
            st.metric(
                label="Today's P/L",
                value=f"${daily_change_dollars:,.2f}",
                delta=f"{daily_change:.2f}%"
            )

        with vix_col:
            # Display YTD performance (already calculated above)
            st.metric(
                label="YTD Return (vs SPY)",
                value=f"{ytd_performance['portfolio_ytd']:.2f}%",
                delta=f"{ytd_performance['outperformance']:+.2f}% vs SPY",
                help=f"Your YTD: {ytd_performance['portfolio_ytd']:.2f}% | SPY YTD: {ytd_performance['spy_ytd']:.2f}%"
            )
            
        with beta_col:
            st.metric(
                label="Portfolio Beta",
                value=f"{portfolio_beta:.2f}",
                delta=None,
                help="Beta relative to S&P 500. >1 = more volatile, <1 = less volatile"
            )

    with st.container():
        st.subheader("Market Overview")
        spy_col, vix_col, fear_greed_col, regime_col, checklist_col = st.columns(5)

        with spy_col:
            # Fetch S&P 500 (SPY) current price using yfinance
            try:
                import yfinance as yf
                spy = yf.Ticker("SPY")
                spy_info = spy.info
                spy_price = spy_info.get('regularMarketPrice', spy_info.get('currentPrice', 0))
                spy_prev_close = spy_info.get('previousClose', spy_price)
                spy_change_pct = ((spy_price - spy_prev_close) / spy_prev_close * 100) if spy_prev_close else 0
                
                st.metric(
                    label="S&P 500 (SPY)",
                    value=f"${spy_price:.2f}",
                    delta=f"{spy_change_pct:+.2f}%"
                )
            except Exception as e:
                st.metric(label="S&P 500 (SPY)", value="N/A", delta=None)
        
        with vix_col:
            
            st.metric(
                label="VIX",
                value=vix_display,
                delta=vix_trend
            )

        with fear_greed_col:
            import fear_and_greed
            fear_greed = fear_and_greed.get()
            st.metric(
                label="Fear & Greed Index",
                value=f"{fear_greed.value:.2f}"
            )

        with regime_col:
            from util.market.market_regime_sentiment import get_spy_vix_regimes
            spy = yf.download("SPY", start="2015-01-01")["Close"]
            vix = yf.download("^VIX", start="2015-01-01")["Close"]
            regimes_df, current_regime, hmm_model, state_order = get_spy_vix_regimes(spy, vix)

            st.metric(
                label="Market Regime",
                value=current_regime,
                delta="+1"
            )

        with checklist_col:
            st.metric(
                label="Checklist",
                value="All systems go",
                delta="0"
            )

    
    # Market News Analysis
    st.subheader("Market News")
    
    # General Market News
    with st.expander("📰 Market News Analysis"):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("Get the latest analysis of market events and their implications")
        with col2:
            if st.button("🔄 Refresh", key="news_refresh"):
                with st.spinner("Analyzing market news..."):
                    analysis = st.session_state.hub.analyze_news(
                        "what are the most important market events and their implications?"
                    )
                    if analysis:
                        st.session_state.latest_news = analysis

        if 'latest_news' in st.session_state and st.session_state.latest_news:
            st.markdown(st.session_state.latest_news)
            
    # Portfolio-Specific News
    with st.expander("📊 Portfolio News Analysis"):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("Get news and analysis specific to your portfolio holdings")
        with col2:
            if st.button("🔄 Refresh", key="portfolio_news_refresh"):
                with st.spinner("Analyzing portfolio news..."):
                   
                    # Use the A2A protocol
                    news_message = st.session_state.hub.news_agent.get_portfolio_news(
                        st.session_state.hub.portfolio_agent
                    )
                    if news_message and "analysis" in news_message:
                        st.session_state.portfolio_news = news_message["analysis"]
                    elif "error" in news_message:
                        st.error(f"Error: {news_message['error']}")

        if 'portfolio_news' in st.session_state and st.session_state.portfolio_news:
            st.markdown(st.session_state.portfolio_news)

# Market Analysis Tab
with tab_market:
    if st.button("Analyze Market Conditions", key="market_analyze"):
        with st.spinner("Analyzing market indicators..."):
            result = st.session_state.hub.market_agent.run_analysis()
            
            try:
                if isinstance(result, dict) and 'analysis' in result:
                    # Parse the JSON string from the analysis
                    import json
                    analysis_data = json.loads(result['analysis'])
                    
                    # Create two columns for the indicators
                    col1, col2 = st.columns(2)
                    
                    # Split indicators into two groups for two-column layout
                    indicators = analysis_data['indicators']
                    
                    # First column
                    with col1:
                        # Show last update time
                        st.caption(f"Last updated: {analysis_data['last_updated']}")

                        for i, (key, indicator) in enumerate(indicators.items()):
                            label, trend_symbol, delta_color = format_indicator_metric(
                                indicator['name'],
                                indicator['current_value'],
                                indicator['trend'],
                                indicator['impact']
                            )
                            # Map delta_color to CSS color
                            color = {
                                "normal": "#0fba81",  # Green
                                "inverse": "#ff4b4b", # Red
                                "off": "#868e96"      # Gray
                            }.get(delta_color, "#868e96")
                            
                            st.html(
                                f"""
                                <div style="padding: 1rem; border-radius: 0.5rem; border: 1px solid {color}; margin-bottom: 1rem;">
                                    <p style="color: {color}; font-weight: bold; margin-bottom: 0.5rem;">{label} - {indicator['impact']}</p>
                                    <span>{indicator['current_value']}{trend_symbol}</span>
                                </div>
                                """
                            )
                    
                    # Second column
                    with col2:
                    # Add Yield Curve section
                        st.subheader("Treasury Yield Curve")
                        try:
                            from util.market.yield_curve import load_yield_curve_polygon
                            import plotly.express as px
                            
                            # Load yield curve data
                            with st.spinner("Loading yield curve data..."):
                                df_yields = load_yield_curve_polygon()
                                latest_date = df_yields["date"].max()
                                latest_curve = df_yields[df_yields["date"] == latest_date]
                                
                                # Create yield curve plot using plotly express
                                fig = px.line(
                                    latest_curve, 
                                    x="maturity_years", 
                                    y="yield",
                                    title=f"Treasury Yield Curve (as of {latest_date})",
                                    labels={
                                        "maturity_years": "Years to Maturity",
                                        "yield": "Yield (%)"
                                    }
                                )
                                fig.update_layout(
                                    showlegend=False,
                                    hovermode='x',
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display yield data in a table
                                st.caption("Treasury Yields")
                                st.dataframe(
                                    latest_curve[["maturity_years", "yield"]]
                                    .rename(columns={
                                        "maturity_years": "Maturity (Years)",
                                        "yield": "Yield (%)"
                                    }),
                                    hide_index=True
                                )
                        except Exception as e:
                            st.error(f"Error loading yield curve: {str(e)}")
                else:
                    st.error("Unable to parse market analysis data")
            except Exception as e:
                st.error(f"Error displaying market analysis: {str(e)}")
                st.code(result, language='json')
            
            # Update dashboard metrics if we have new values
            if 'latest_values' in result:
                values = result['latest_values']
                
                # Update VIX
                if 'Vix (Volatility Index)' in values:
                    st.session_state.vix_value = values['Vix (Volatility Index)']['value']
                    
                # Update ISM PMI
                if 'ISM Manufacturing PMI' in values:
                    st.session_state.pmi_value = values['ISM Manufacturing PMI']['value']
                    
                # Update Consumer Sentiment
                if 'Consumer Sentiment (University of Michigan)' in values:
                    st.session_state.sentiment_value = values['Consumer Sentiment (University of Michigan)']['value']

# Financial Education Tab
with tab_education:
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Example questions as buttons
    st.markdown("### Example Questions")
    example_questions = [
        "What is alpha?",
        "Explain market neutral",
        "What is arbitrage?"
    ]
    cols = st.columns(len(example_questions))
    for col, q in zip(cols, example_questions):
        with col:
            st.button(q, key=f"edu_{q}")
    
    # Chat input
    if prompt := st.chat_input("Ask about any finance or tax-related topic...", key="fin_edu_chat"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                response = st.session_state.hub.literacy_bot.get_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Portfolio Management Tab
with tab_portfolio:
    # File upload section
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload your portfolio CSV file (optional)", type=['csv'], key="portfolio_upload")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        if st.button("🔄 Refresh Analysis", key="refresh_portfolio"):
            st.cache_data.clear()
            st.rerun()
    
    with st.spinner("Analyzing portfolio..."):
        try:
            # Initialize analysis with sample portfolio by default
            # Pass None for sample portfolio, or the uploaded file for user portfolio
            details_df, metrics, analysis = st.session_state.hub.portfolio_agent.run_analysis(
                csv_file=uploaded_file if uploaded_file is not None else None
            )
            
            # Create tabs for different views
            summary_tab, holdings_tab, analysis_tab, attribution_tab = st.tabs(["Summary", "Holdings", "Analysis", "Performance Attribution"])
            
            with summary_tab:
                # Portfolio value and metrics
                st.subheader("Portfolio Summary")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric(
                        label="Total Portfolio Value",
                        value=f"${metrics['total_value']:,.2f}"
                    )
                with metric_col2:
                    st.metric(
                        label="Number of Holdings",
                        value=len(details_df) if not details_df.empty else 0
                    )
                with metric_col3:
                    st.metric(
                        label="Sectors",
                        value=len(metrics['sectors'])
                    )
                
                # Sector breakdown
                st.subheader("Sector Allocation")
                st.write(", ".join(metrics['sectors']))
                
                if uploaded_file:
                    st.caption("Showing analysis for uploaded portfolio")
                else:
                    st.caption("Showing analysis for sample portfolio - Upload your own CSV to analyze your portfolio")
            
            with holdings_tab:
                st.subheader("Holdings Details")
                if not details_df.empty:
                    # Format the dataframe for display
                    display_df = details_df.copy()
                    if 'position_value' in display_df.columns:
                        display_df['position_value'] = display_df['position_value'].map('${:,.2f}'.format)
                    if 'current_price' in display_df.columns:
                        display_df['current_price'] = display_df['current_price'].map('${:,.2f}'.format)
                    if 'portfolio_weight' in display_df.columns:
                        display_df['portfolio_weight'] = display_df['portfolio_weight'].map('{:.1f}%'.format)
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.warning("No holdings data available")
            
            with analysis_tab:
                st.subheader("Portfolio Analysis")
                st.write(analysis)
            
            with attribution_tab:
                st.subheader("Performance Attribution")
                
                # Create a waterfall chart showing contribution to total return
                if not details_df.empty and 'daily_change' in details_df.columns:
                    # Calculate contribution to total return
                    contribution_data = details_df.copy()
                    contribution_data['contribution'] = (
                        contribution_data['daily_change'] * 
                        contribution_data['portfolio_weight'] / 100
                    )
                    
                    # Sort by absolute contribution for better visualization
                    contribution_data = contribution_data.sort_values(
                        by='contribution', 
                        key=abs,
                        ascending=False
                    )
                    
                    # Create the waterfall chart using plotly
                    fig = go.Figure()
                    
                    # Add individual bars for each holding
                    fig.add_trace(go.Waterfall(
                        name="Performance Attribution",
                        orientation="v",
                        measure=["relative"] * len(contribution_data),
                        x=contribution_data['ticker'],
                        y=contribution_data['contribution'],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                        increasing={"marker": {"color": "#00B050"}},
                        decreasing={"marker": {"color": "#FF0000"}},
                        text=contribution_data['contribution'].apply(lambda x: f"{x:.2f}%"),
                        textposition="outside"
                    ))
                    
                    # Add total return bar
                    fig.add_trace(go.Waterfall(
                        name="Total Return",
                        orientation="v",
                        measure=["total"],
                        x=["Total"],
                        y=[contribution_data['contribution'].sum()],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                        increasing={"marker": {"color": "#4472C4"}},
                        decreasing={"marker": {"color": "#4472C4"}},
                        text=[f"{contribution_data['contribution'].sum():.2f}%"],
                        textposition="outside"
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Daily Performance Attribution",
                        showlegend=False,
                        xaxis_title="Holdings",
                        yaxis_title="Contribution to Return (%)",
                        height=500,
                        waterfallgap=0.2,
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display contribution table
                    st.subheader("Detailed Attribution")
                    attribution_display = contribution_data[['ticker', 'portfolio_weight', 'daily_change', 'contribution']].copy()
                    attribution_display.columns = ['Ticker', 'Portfolio Weight (%)', 'Daily Change (%)', 'Contribution (%)']
                    attribution_display = attribution_display.round(2)
                    st.dataframe(attribution_display, use_container_width=True)
                else:
                    st.warning("No performance data available for attribution analysis")
                
        except Exception as e:
            st.error(f"Error analyzing portfolio: {str(e)}")

# # Idea Builder Tab
# with tab_ideas:
#     # Use the modular UI implementation
#     # render_idea_builder_tab()
#     from ui.idea_builder_ui import render_multi_step_form
#     render_multi_step_form()

# Footer
st.markdown("---")
st.caption("Finance AI Hub - Powered by Multiple Specialized Agents")