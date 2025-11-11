import os
import json
import finnhub
import streamlit as st
from typing import List, Dict
import requests
from datetime import datetime, timedelta

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# API Keys and URLs
FRED_API_KEY = os.environ.get('FRED_API_KEY')
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY')
OPEN_AI_API_KEY = os.environ.get('OPEN_AI_API_KEY')

# Data source URLs
URLS = {
    'fred': "https://api.stlouisfed.org/fred",  # API base URL
    'fred_docs': "https://fred.stlouisfed.org/docs/api/fred/",  # Documentation
    'bls': 'https://www.bls.gov/developrs',
    'census': 'https://www.census.gov/data/developers/data-sets.html',
    'ism': 'https://www.ismworld.org/',
    'eia': "https://www.eia.gov/opendata",
    'nasdaq': "https://data.nasdaq.com",
    'worldbank': "https://www.worldbank.org/en/research.commodity-markets"
}
    

class MarketRegimeRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=OPEN_AI_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-16k",
            api_key=OPEN_AI_API_KEY,
            temperature=0.1
        )
        self.vectorstore = None
        
    @st.cache_data(ttl=300)
    def fetch_market_data(_self) -> List[Dict]:
        data_sources = []
        # Fetch FRED data if available
        if FRED_API_KEY:
            # Get key economic indicators - only including active/available series
            indicator_list = {
                'GDP (QoQ SAAR)': 'GDPC1',
                'M2 Money Supply (MoM %)': 'M2SL',
                'Fed Funds Rate (Upper Bound)': 'DFEDTARU',
                'Conference Board LEI': 'USSLIND',
                'Building Permits (Annualized Rate)': 'PERMIT',
                'Employment Situation Report (NFP)': 'PAYEMS',
                'Employment Situation Report (UR)': 'UNRATE',
                'Weekly Jobless Claims': 'ICSA',
                'Durable Goods Orders': 'DGORDER',
                'Industrial Production (MoM %)': 'INDPRO',
                'Vix (Volatility Index)': 'VIXCLS'
            }

            for series_name, series_id in indicator_list.items():
                fred_url = f"{URLS['fred']}/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': FRED_API_KEY,
                    'file_type': 'json',
                    'limit': 2,  # Get last two observations for trend
                    'sort_order': 'desc',
                    'observation_start': (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                }
                
                try:
                    response = requests.get(fred_url, params=params)
                    response.raise_for_status()  # Raise an error for bad status codes
                    data = response.json()
                    
                    if 'observations' in data and len(data['observations']) >= 1:
                        latest_observation = data['observations'][0]
                        latest_value = latest_observation.get('value', 'N/A')
                        latest_date = latest_observation.get('date', 'N/A')
                        
                        # Calculate trend if we have previous data
                        trend = "N/A"
                        if len(data['observations']) >= 2:
                            prev_value = float(data['observations'][1].get('value', 0))
                            curr_value = float(latest_value)
                            if curr_value > prev_value * 1.01:
                                trend = "Growing"
                            elif curr_value < prev_value * 0.99:
                                trend = "Decreasing"
                            else:
                                trend = "Stable"
                        
                        data_sources.append({
                            'url': URLS['fred_docs'],
                            'name': f'FRED {series_name}',
                            'content': {
                                'indicator': series_name,
                                'series_id': series_id,
                                'value': latest_value,
                                'date': latest_date,
                                'trend': trend
                            }
                        })
                except requests.RequestException as e:
                    st.warning(f"Error fetching FRED {series_name} data: {str(e)}")
                except ValueError as e:  # JSON decode error
                    st.warning(f"Error parsing FRED {series_name} response: {str(e)}")

        return data_sources

    def create_analysis_chain(self):
        """Create the analysis chain for indicator JSON format"""
        prompt_template = """You are a senior macro market analyst. Based on the following market information, create a JSON analysis.

Context information:
{context}

Return a valid JSON object EXACTLY in this format (including the outer curly braces):
{
    "indicators": {
        "gdp": {
            "name": "GDP (QoQ SAAR)",
            "current_value": "value",
            "trend": "Growing/Stable/Decreasing",
            "impact": "Brief one-line impact"
        },
        "money_supply": {
            "name": "M2 Money Supply",
            "current_value": "value",
            "trend": "Growing/Stable/Decreasing",
            "impact": "Brief one-line impact"
        },
        "fed_rate": {
            "name": "Federal Funds Rate",
            "current_value": "value",
            "trend": "Growing/Stable/Decreasing",
            "impact": "Brief one-line impact"
        },
        "building_permits": {
            "name": "Building Permits",
            "current_value": "value",
            "trend": "Growing/Stable/Decreasing",
            "impact": "Brief one-line impact"
        },
        "nonfarm_payrolls": {
            "name": "Employment (Non-Farm Payrolls)",
            "current_value": "value",
            "trend": "Growing/Stable/Decreasing",
            "impact": "Brief one-line impact"
        },
        "unemployment": {
            "name": "Unemployment Rate",
            "current_value": "value",
            "trend": "Growing/Stable/Decreasing",
            "impact": "Brief one-line impact"
        },
        "jobless_claims": {
            "name": "Initial Jobless Claims",
            "current_value": "value",
            "trend": "Growing/Stable/Decreasing",
            "impact": "Brief one-line impact"
        },
        "durable_goods": {
            "name": "Durable Goods Orders",
            "current_value": "value",
            "trend": "Growing/Stable/Decreasing",
            "impact": "Brief one-line impact"
        },
        "industrial_production": {
            "name": "Industrial Production",
            "current_value": "value",
            "trend": "Growing/Stable/Decreasing",
            "impact": "Brief one-line impact"
        },
        "vix": {
            "name": "VIX Index",
            "current_value": "value",
            "trend": "Growing/Stable/Decreasing",
            "impact": "Brief one-line impact"
        }
    },
    "last_updated": "current_date_time"
}

Use exact values from the data where available. For trend, compare to previous periods if that information is available.
Keep impact assessments very brief and focused on market implications.
Ensure the output is valid JSON that can be parsed."""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        return prompt | self.llm | StrOutputParser()

    @st.cache_data(ttl=300)
    def run_analysis(_self):
        """Execute the full analysis pipeline"""
        try:
            with st.spinner("Fetching market data..."):
                data_sources = _self.fetch_market_data()
                
                # Extract the latest values first
                latest_values = {}
                for source in data_sources:
                    if isinstance(source.get('content', {}), dict):
                        if 'indicator' in source['content']:
                            latest_values[source['content']['indicator']] = {
                                'value': source['content']['value'],
                                'date': source['content']['date'],
                                'trend': source['content'].get('trend', 'N/A')
                            }
            
            with st.spinner("Processing market information..."):
                if not data_sources:
                    error_json = {
                        "indicators": {},
                        "error": "No market data available for analysis.",
                        "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    return {'analysis': json.dumps(error_json), 'latest_values': {}}
            
            with st.spinner("Analyzing market indicators..."):
                # Create structured JSON with all indicators
                indicators_json = {
                    "indicators": {
                        "gdp": {
                            "name": "GDP (QoQ SAAR)",
                            "current_value": latest_values.get('GDP (QoQ SAAR)', {}).get('value', 'N/A'),
                            "trend": latest_values.get('GDP (QoQ SAAR)', {}).get('trend', 'N/A'),
                            "impact": "Impact assessment based on GDP trends"
                        },
                        "money_supply": {
                            "name": "M2 Money Supply",
                            "current_value": latest_values.get('M2 Money Supply (MoM %)', {}).get('value', 'N/A'),
                            "trend": latest_values.get('M2 Money Supply (MoM %)', {}).get('trend', 'N/A'),
                            "impact": "Indicates liquidity conditions in the market"
                        },
                        "fed_rate": {
                            "name": "Federal Funds Rate",
                            "current_value": latest_values.get('Fed Funds Rate (Upper Bound)', {}).get('value', 'N/A'),
                            "trend": latest_values.get('Fed Funds Rate (Upper Bound)', {}).get('trend', 'N/A'),
                            "impact": "Key indicator of monetary policy stance"
                        },
                        "building_permits": {
                            "name": "Building Permits",
                            "current_value": latest_values.get('Building Permits (Annualized Rate)', {}).get('value', 'N/A'),
                            "trend": latest_values.get('Building Permits (Annualized Rate)', {}).get('trend', 'N/A'),
                            "impact": "Leading indicator of construction activity"
                        },
                        "nonfarm_payrolls": {
                            "name": "Employment (Non-Farm Payrolls)",
                            "current_value": latest_values.get('Employment Situation Report (NFP)', {}).get('value', 'N/A'),
                            "trend": latest_values.get('Employment Situation Report (NFP)', {}).get('trend', 'N/A'),
                            "impact": "Key measure of employment strength"
                        },
                        "unemployment": {
                            "name": "Unemployment Rate",
                            "current_value": latest_values.get('Employment Situation Report (UR)', {}).get('value', 'N/A'),
                            "trend": latest_values.get('Employment Situation Report (UR)', {}).get('trend', 'N/A'),
                            "impact": "Indicates overall labor market health"
                        },
                        "jobless_claims": {
                            "name": "Initial Jobless Claims",
                            "current_value": latest_values.get('Weekly Jobless Claims', {}).get('value', 'N/A'),
                            "trend": latest_values.get('Weekly Jobless Claims', {}).get('trend', 'N/A'),
                            "impact": "Leading indicator of employment trends"
                        },
                        "durable_goods": {
                            "name": "Durable Goods Orders",
                            "current_value": latest_values.get('Durable Goods Orders', {}).get('value', 'N/A'),
                            "trend": latest_values.get('Durable Goods Orders', {}).get('trend', 'N/A'),
                            "impact": "Indicates business and consumer spending"
                        },
                        "industrial_production": {
                            "name": "Industrial Production",
                            "current_value": latest_values.get('Industrial Production (MoM %)', {}).get('value', 'N/A'),
                            "trend": latest_values.get('Industrial Production (MoM %)', {}).get('trend', 'N/A'),
                            "impact": "Measure of economic output and capacity"
                        },
                        "vix": {
                            "name": "VIX Index",
                            "current_value": latest_values.get('Vix (Volatility Index)', {}).get('value', 'N/A'),
                            "trend": latest_values.get('Vix (Volatility Index)', {}).get('trend', 'N/A'),
                            "impact": "Market volatility and risk sentiment indicator"
                        }
                    },
                    "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Store in session state
                st.session_state.analysis = indicators_json
                st.session_state.last_update = datetime.now()
                st.session_state.latest_values = latest_values
                
                return {
                    'analysis': json.dumps(indicators_json),  # Convert to JSON string
                    'latest_values': latest_values
                }
                
        except Exception as e:
            error_msg = f"Analysis pipeline error: {str(e)}"
            st.error(error_msg)
            return {'analysis': json.dumps({"error": error_msg}), 'latest_values': {}}

# Development tools (hidden in production)
with st.expander("Developer Tools"):
    if st.button("Clear Cache", key="clear_cache"):
        st.cache_data.clear()
        st.rerun()