import os
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

from langchain_openai import ChatOpenAI
from util.a2a.agent_messaging import AgentMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class PortfolioAgent:
    def __init__(self):
        self.OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")
        self.llm = ChatOpenAI(
            model="gpt-4",
            api_key=self.OPEN_AI_API_KEY,
            temperature=0.3
        )
        
    def load_portfolio(self, csv_file=None) -> pd.DataFrame:
        """Load and validate portfolio CSV file. If no file provided, loads sample portfolio."""
        try:
            # Debug information
            st.write("Debug: Loading portfolio...")
            st.write(f"Debug: CSV file type: {type(csv_file)}")
            
            if csv_file is None:
                st.write("Debug: Using sample portfolio")
                # Load sample portfolio from util/portfolio directory
                base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to project root
                sample_path = os.path.join(base_dir, 'util', 'portfolio', 'sample_portfolio.csv')
                st.write(f"Debug: Sample path: {sample_path}")
                if os.path.exists(sample_path):
                    st.write("Debug: Sample file exists, loading...")
                    df = pd.read_csv(sample_path)
                else:
                    st.write("Debug: Using hardcoded sample data")
                    # Fallback to hardcoded sample if file doesn't exist
                    data = {
                        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JNJ'],
                        'shares': [100, 75, 25, 30, 50]
                    }
                    df = pd.DataFrame(data)
            else:
                st.write("Debug: Using uploaded file")
                # For Streamlit uploaded files, need to read the bytes
                if hasattr(csv_file, 'getvalue'):
                    content = csv_file.getvalue()
                    st.write("Debug: Reading from Streamlit UploadedFile")
                    df = pd.read_csv(pd.io.common.BytesIO(content))
                else:
                    st.write("Debug: Attempting direct CSV read")
                    df = pd.read_csv(csv_file)
            
            # Validate required columns
            required_cols = ['ticker', 'shares']
            if not all(col in df.columns for col in required_cols):
                raise ValueError("CSV must contain 'ticker' and 'shares' columns")
            
            # Clean and standardize
            df['ticker'] = df['ticker'].str.strip().str.upper()
            df['shares'] = pd.to_numeric(df['shares'])
            
            return df
        except Exception as e:
            raise ValueError(f"Error loading portfolio: {str(e)}")
    
    def fetch_stock_data(self, tickers: List[str]) -> Dict[str, yf.Ticker]:
        """Fetch current stock data for portfolio"""
        stock_data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                stock_data[ticker] = stock
            except Exception as e:
                st.warning(f"Could not fetch data for {ticker}: {str(e)}")
        return stock_data
    
    def calculate_portfolio_metrics(self, portfolio_df: pd.DataFrame, stock_data: Dict[str, yf.Ticker]) -> Tuple[pd.DataFrame, Dict]:
        """Calculate key portfolio metrics"""
        metrics = {
            'total_value': 0,
            'daily_change': 0,
            'sectors': set(),
            'risk_level': 'Unknown'
        }
        
        # Add current prices and calculations
        portfolio_details = []
        
        for _, row in portfolio_df.iterrows():
            ticker = row['ticker']
            shares = row['shares']
            
            if ticker in stock_data:
                stock = stock_data[ticker]
                info = stock.info
                current_price = info.get('regularMarketPrice', 0)
                prev_close = info.get('previousClose', current_price)
                position_value = current_price * shares
                
                details = {
                    'ticker': ticker,
                    'shares': shares,
                    'current_price': current_price,
                    'position_value': position_value,
                    'daily_change': ((current_price - prev_close) / prev_close) * 100
                }
                portfolio_details.append(details)
                
                metrics['total_value'] += position_value
                metrics['sectors'].add(info.get('sector', 'Unknown'))
                
        # Create details dataframe
        details_df = pd.DataFrame(portfolio_details)
        if not details_df.empty:
            # Calculate portfolio weights
            details_df['portfolio_weight'] = (details_df['position_value'] / metrics['total_value']) * 100
            metrics['daily_change'] = (details_df['daily_change'] * details_df['portfolio_weight']).sum() / 100
        
        metrics['sectors'] = list(metrics['sectors'])
        
        return details_df, metrics

    def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming messages from other agents"""
        if message.intent == "GET_HOLDINGS":
            try:
                # Load the current portfolio
                portfolio_df = self.load_portfolio()
                
                # Get the list of tickers
                symbols = portfolio_df['ticker'].tolist()
                
                return AgentMessage(
                    sender="portfolio_agent",
                    intent="HOLDINGS_DATA",
                    content={
                        'symbols': symbols
                    }
                )
            except Exception as e:
                return AgentMessage(
                    sender="portfolio_agent",
                    intent="HOLDINGS_DATA",
                    content={
                        'error': str(e)
                    }
                )
        
        # Add current prices and calculations
        portfolio_details = []
        
        for _, row in portfolio_df.iterrows():
            ticker = row['ticker']
            shares = row['shares']
            
            if ticker in stock_data:
                stock = stock_data[ticker]
                info = stock.info
                current_price = info.get('regularMarketPrice', 0)
                prev_close = info.get('previousClose', current_price)
                position_value = current_price * shares
                daily_change = ((current_price - prev_close) / prev_close) * 100
                
                portfolio_details.append({
                    'ticker': ticker,
                    'shares': shares,
                    'current_price': current_price,
                    'position_value': position_value,
                    'daily_change': daily_change,
                    'sector': info.get('sector', 'Unknown')
                })
                
                metrics['total_value'] += position_value
                metrics['sectors'].add(info.get('sector', 'Unknown'))
        
        details_df = pd.DataFrame(portfolio_details)
        
        # Calculate portfolio percentages
        if not details_df.empty:
            details_df['portfolio_weight'] = (details_df['position_value'] / metrics['total_value']) * 100
            
        # Convert sectors to list
        metrics['sectors'] = list(metrics['sectors'])
        
        return details_df, metrics
    
    def analyze_portfolio(self, details_df: pd.DataFrame, metrics: Dict) -> str:
        """Generate portfolio analysis using LLM"""
        analysis_prompt = ChatPromptTemplate.from_template("""
        You are a professional portfolio analyst. Analyze the following portfolio details and provide insights:

        Portfolio Summary:
        - Total Value: ${total_value:,.2f}
        - Sectors: {sectors}
        
        Holdings:
        {holdings}
        
        Please provide a comprehensive analysis including:
        1. Portfolio diversification assessment
        2. Sector concentration risks
        3. Key recommendations for portfolio optimization
        4. Any notable position sizes that might need attention
        
        Keep the analysis professional, fact-based, and actionable.
        """)
        
        # Format holdings for prompt
        holdings_text = details_df.to_string() if not details_df.empty else "No holdings data available"
        
        # Create and run the chain
        chain = analysis_prompt | self.llm | StrOutputParser()
        
        analysis = chain.invoke({
            "total_value": metrics['total_value'],
            "sectors": ", ".join(metrics['sectors']),
            "holdings": holdings_text
        })
        
        return analysis
    
    def run_analysis(self, csv_file=None) -> Tuple[pd.DataFrame, Dict, str]:
        """Main method to analyze portfolio. If no CSV file provided, uses sample portfolio."""
        try:
            # Load and validate portfolio
            portfolio_df = self.load_portfolio(csv_file)
            
            # Fetch current market data
            stock_data = self.fetch_stock_data(portfolio_df['ticker'].unique())
            
            # Calculate portfolio metrics
            details_df, metrics = self.calculate_portfolio_metrics(portfolio_df, stock_data)
            
            # Generate analysis
            analysis = self.analyze_portfolio(details_df, metrics)
            
            return details_df, metrics, analysis
            
        except Exception as e:
            raise Exception(f"Portfolio analysis failed: {str(e)}")
