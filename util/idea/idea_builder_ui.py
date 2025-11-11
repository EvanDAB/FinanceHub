# Updated Idea Builder Implementation
import streamlit as st
from datetime import datetime
import yfinance as yf
from typing import List, Dict, Optional
from agents.idea_builder_goal_planner_agent import InvestmentIdea

def get_instrument_suggestions(search_term: str, instrument_type: str) -> List[Dict]:
    """
    Get instrument suggestions based on search term and type.
    Returns a list of dictionaries with symbol and name.
    """
    if not search_term:
        return []
    
    # Map instrument types to Yahoo Finance exchanges/suffixes
    type_mappings = {
        "Stocks": ["", ".NE", ".L", ".TO"],  # US, NEO, London, Toronto
        "ETFs": ["", ".TO"],  # US and Toronto ETFs
        "Commodities": [".CMX", ".NYM"],  # COMEX and NYMEX
        "Crypto": ["-USD", "-USDT"],  # Major crypto pairs
    }
    
    results = []
    if instrument_type in type_mappings:
        for suffix in type_mappings[instrument_type]:
            try:
                # Search for the symbol with the appropriate suffix
                search_symbol = f"{search_term}{suffix}"
                ticker = yf.Ticker(search_symbol)
                info = ticker.info
                
                if info and 'symbol' in info:
                    results.append({
                        'symbol': info['symbol'],
                        'name': info.get('longName', info.get('shortName', 'Unknown')),
                        'type': instrument_type
                    })
            except Exception:
                continue
                
    return results[:5]  # Limit to top 5 results

def render_idea_builder_tab():
    """Renders the Idea Builder tab content"""
    # Header with Create Button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Investment Ideas")
    with col2:
        st.write("")  # Add some spacing
        if st.button("➕ Create New Idea", use_container_width=True):
            st.session_state.show_new_idea = True
            st.rerun()
    
    # Show either the idea form or the list of ideas
    if st.session_state.get('show_new_idea', False):
        # Show the idea creation form
        st.markdown("### Create New Investment Idea")
        
        # Idea input section
        thesis = st.text_area(
            "What's your investment idea?",
            placeholder="Describe your investment thesis or trading idea...",
            key="idea_thesis"
        )
        
        # Investment Type Selection
        st.markdown("### 1. Investment Type")
        investment_types = st.multiselect(
            "Select Investment Type(s)",
            ["Stocks", "ETFs", "Options", "Futures", "Commodities", "Crypto"],
            key="idea_type"
        )
        
        # Initialize selected instrument
        selected_instrument = None
        
        if investment_types:
            # Show instrument selector for supported types
            valid_types = ["Stocks", "ETFs", "Commodities", "Crypto"]
            if investment_types[0] in valid_types:
                st.markdown("### 2. Select Instrument")
                search_term = st.text_input(
                    f"Search for {investment_types[0]}",
                    key=f"search_{investment_types[0]}",
                    placeholder=f"Enter symbol or name..."
                )
                
                if search_term:
                    suggestions = get_instrument_suggestions(search_term, investment_types[0])
                    if suggestions:
                        options = [f"{s['symbol']} - {s['name']}" for s in suggestions]
                        selected = st.selectbox(
                            f"Select {investment_types[0]}",
                            options,
                            key=f"select_{investment_types[0]}"
                        )
                        
                        if selected:
                            idx = options.index(selected)
                            selected_instrument = suggestions[idx]
                            
                            # Display instrument info
                            try:
                                ticker = yf.Ticker(selected_instrument['symbol'])
                                info = ticker.info
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "Current Price",
                                        f"${info.get('currentPrice', 'N/A'):,.2f}"
                                    )
                                with col2:
                                    if 'fiftyTwoWeekRange' in info:
                                        st.metric(
                                            "52-Week Range",
                                            info['fiftyTwoWeekRange']
                                        )
                            except Exception:
                                st.warning("Could not fetch detailed information")
                    else:
                        st.info(f"No matching {investment_types[0]} found")
        
        # Parameters section
        st.markdown("### 3. Investment Parameters")
        col1, col2 = st.columns(2)
        with col1:
            timeframe = st.selectbox(
                "Investment Timeframe",
                ["Short-term (< 1 month)", "Medium-term (1-6 months)", "Long-term (> 6 months)"],
                key="idea_timeframe"
            )
            risk_level = st.select_slider(
                "Risk Level",
                options=["Low", "Medium", "High"],
                value="Medium",
                key="idea_risk"
            )
        
        with col2:
            position_size = st.number_input(
                "Position Size (%)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                key="idea_size"
            )
            direction = st.radio(
                "Position Direction",
                ["Long", "Short"],
                key="idea_direction"
            )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Analyze Idea", key="analyze_idea", use_container_width=True):
                if not thesis.strip():
                    st.error("Please enter an investment thesis")
                elif not investment_types:
                    st.error("Please select at least one investment type")
                else:
                    with st.spinner("Analyzing investment idea..."):
                        idea = InvestmentIdea(
                            thesis=thesis,
                            timeframe=timeframe,
                            risk_level=risk_level,
                            investment_types=investment_types,
                            position_size=position_size,
                            direction=direction,
                            created_at=datetime.now(),
                            instrument=selected_instrument
                        )
                        analysis = st.session_state.hub.idea_builder.analyze_idea(idea)
                        idea.analysis = analysis
                        st.session_state.current_idea = idea
                        st.session_state.current_analysis = analysis
                        st.markdown("### Analysis Results")
                        st.markdown(analysis)
        
        with col2:
            if st.button("Save Idea", key="save_idea", use_container_width=True):
                if not thesis.strip():
                    st.error("Please enter an investment thesis")
                elif not investment_types:
                    st.error("Please select at least one investment type")
                else:
                    idea = InvestmentIdea(
                        thesis=thesis,
                        timeframe=timeframe,
                        risk_level=risk_level,
                        investment_types=investment_types,
                        position_size=position_size,
                        direction=direction,
                        created_at=datetime.now(),
                        instrument=selected_instrument,
                        analysis=st.session_state.get("current_analysis")
                    )
                    idea_id = st.session_state.hub.idea_storage.save_idea(idea)
                    st.success("Investment idea saved!")
                    st.session_state.show_new_idea = False
                    st.rerun()
        
        with col3:
            if st.button("Cancel", key="cancel_idea", use_container_width=True):
                st.session_state.show_new_idea = False
                st.rerun()
            
    else:
        # Display list of saved ideas
        saved_ideas = st.session_state.hub.idea_storage.list_ideas()
        
        if not saved_ideas:
            st.info("No investment ideas yet. Click 'Create New Idea' to get started!")
        else:
            for idea in saved_ideas:
                with st.expander(f"{idea['thesis']} ({idea['created_at']})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Timeframe:** {idea['timeframe']}")
                        st.write(f"**Risk Level:** {idea['risk_level']}")
                    
                    with col2:
                        if st.button("View Analysis", key=f"view_{idea['id']}", use_container_width=True):
                            full_idea = st.session_state.hub.idea_storage.load_idea(idea['id'])
                            if full_idea and full_idea.analysis:
                                st.markdown("### Analysis")
                                st.markdown(full_idea.analysis)
                        
                        if st.button("Delete", key=f"delete_{idea['id']}", use_container_width=True):
                            if st.session_state.hub.idea_storage.delete_idea(idea['id']):
                                st.success("Idea deleted!")
                                st.rerun()