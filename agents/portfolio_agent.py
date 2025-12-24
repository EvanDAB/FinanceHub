import os
import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

from langchain_openai import ChatOpenAI
from util.a2a.agent_messaging import AgentMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
            if csv_file is None:
                # Load sample portfolio from util/portfolio directory
                base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to project root
                sample_path = os.path.join(base_dir, 'util', 'portfolio', 'sample_portfolio.csv')
                if os.path.exists(sample_path):
                    df = pd.read_csv(sample_path)
                else:
                    # Fallback to hardcoded sample if file doesn't exist
                    data = {
                        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JNJ'],
                        'shares': [100, 75, 25, 30, 50]
                    }
                    df = pd.DataFrame(data)
            else:
                # For Streamlit uploaded files, need to read the bytes
                if hasattr(csv_file, 'getvalue'):
                    content = csv_file.getvalue()
                    df = pd.read_csv(pd.io.common.BytesIO(content))
                else:
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
    
    def calculate_portfolio_beta(self, portfolio_df: pd.DataFrame, period: str = "1y") -> float:
        """
        Calculate portfolio beta relative to S&P 500 (SPY).
        
        Args:
            portfolio_df: DataFrame with ticker and shares columns
            period: Historical period for calculation (e.g., "1y", "6mo", "3mo")
            
        Returns:
            Portfolio beta (weighted average of individual stock betas)
        """
        try:
            # Download historical data for portfolio stocks and SPY
            tickers = portfolio_df['ticker'].tolist() + ['SPY']
            
            print(f"[BETA] Starting calculation for {len(tickers)} tickers: {tickers[:5]}...")
            
            # Get historical prices
            end_date = datetime.now()
            if period == "1y":
                start_date = end_date - timedelta(days=365)
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=365)
            
            print(f"[BETA] Downloading data from {start_date.date()} to {end_date.date()}")
            
            # Download data with group_by='ticker' and auto_adjust=True
            # This gives us MultiIndex columns: ('TICKER', 'Close'), ('TICKER', 'Open'), etc.
            raw_data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date, 
                progress=False,
                auto_adjust=True,
                group_by='ticker'
            )
            
            print(f"[BETA] Download complete. Shape: {raw_data.shape}, Empty: {raw_data.empty}")
            
            if raw_data.empty:
                print("[BETA] ERROR: Historical data download returned empty dataset.")
                return 1.0  # Default beta

            print(f"[BETA] Downloaded {len(raw_data)} days of historical data for {len(tickers)} tickers")

            # Extract close prices for each ticker
            data = pd.DataFrame()
            extracted_count = 0
            
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        # Single ticker - no MultiIndex
                        if 'Close' in raw_data.columns:
                            data[ticker] = raw_data['Close']
                            extracted_count += 1
                        elif 'Adj Close' in raw_data.columns:
                            data[ticker] = raw_data['Adj Close']
                            extracted_count += 1
                    else:
                        # Multiple tickers - MultiIndex columns ('TICKER', 'Price')
                        if ticker in raw_data.columns.get_level_values(0):
                            ticker_df = raw_data[ticker]
                            if 'Close' in ticker_df.columns:
                                data[ticker] = ticker_df['Close']
                                extracted_count += 1
                            elif 'Adj Close' in ticker_df.columns:
                                data[ticker] = ticker_df['Adj Close']
                                extracted_count += 1
                except Exception as e:
                    print(f"[BETA] WARNING: Skipped {ticker}: {str(e)}")
                    continue
            
            print(f"[BETA] Successfully extracted price data for {extracted_count}/{len(tickers)} tickers")
            
            if data.empty or len(data.columns) == 0:
                print("[BETA] ERROR: No price data could be extracted from any ticker.")
                return 1.0
            
            # Calculate daily returns
            returns = data.pct_change().dropna()
            
            # Get current prices for weighting
            current_prices = {}
            for ticker in portfolio_df['ticker']:
                if ticker in data.columns:
                    current_prices[ticker] = float(data[ticker].iloc[-1])
            
            # Calculate portfolio weights
            total_value = 0
            for _, row in portfolio_df.iterrows():
                ticker = row['ticker']
                shares = row['shares']
                if ticker in current_prices:
                    total_value += current_prices[ticker] * shares
            
            if total_value == 0:
                return 1.0  # Default beta if no valid data
            
            # Calculate weighted beta
            portfolio_beta = 0
            valid_weights = 0
            
            for _, row in portfolio_df.iterrows():
                ticker = row['ticker']
                shares = row['shares']
                
                # Check if we have data for this ticker and SPY
                has_ticker_data = ticker in returns.columns
                has_spy_data = 'SPY' in returns.columns
                
                if ticker in current_prices and has_ticker_data and has_spy_data:
                    # Calculate weight
                    weight = (current_prices[ticker] * shares) / total_value
                    valid_weights += weight
                    
                    # Calculate individual stock beta using covariance method
                    stock_returns = returns[ticker].values
                    market_returns = returns['SPY'].values
                    
                    # Beta = Cov(stock, market) / Var(market)
                    covariance = np.cov(stock_returns, market_returns)[0][1]
                    market_variance = np.var(market_returns)
                    
                    if market_variance > 0:
                        stock_beta = covariance / market_variance
                        portfolio_beta += weight * stock_beta
            
            # Normalize if we didn't get all weights (some stocks may have failed)
            if valid_weights > 0 and valid_weights < 1:
                portfolio_beta = portfolio_beta / valid_weights
            
            final_beta = round(portfolio_beta, 2) if portfolio_beta != 0 else 1.0
            print(f"[BETA] ✓ Portfolio beta calculated successfully: {final_beta}")
            return final_beta
            
        except Exception as e:
            # Return default beta on any error
            print(f"[BETA] ERROR: Could not calculate portfolio beta: {str(e)}. Using default value of 1.0")
            import traceback
            traceback.print_exc()
            return 1.0
    
    def calculate_ytd_performance(self, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate year-to-date (YTD) performance of the portfolio compared to S&P 500.
        
        Args:
            portfolio_df: DataFrame with ticker and shares columns
            
        Returns:
            Dictionary with portfolio_ytd, spy_ytd, and outperformance percentages
        """
        try:
            # Get start of current year
            current_year = datetime.now().year
            start_of_year = datetime(current_year, 1, 1)
            end_date = datetime.now()
            
            print(f"[YTD] Calculating YTD performance from {start_of_year.date()} to {end_date.date()}")
            
            # Get all tickers including SPY
            tickers = portfolio_df['ticker'].tolist() + ['SPY']
            
            # Download historical data
            raw_data = yf.download(
                tickers,
                start=start_of_year,
                end=end_date,
                progress=False,
                auto_adjust=True,
                group_by='ticker'
            )
            
            if raw_data.empty:
                print("[YTD] ERROR: Historical data download returned empty dataset.")
                return {'portfolio_ytd': 0.0, 'spy_ytd': 0.0, 'outperformance': 0.0}
            
            print(f"[YTD] Downloaded {len(raw_data)} days of data")
            
            # Extract close prices
            data = pd.DataFrame()
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        if 'Close' in raw_data.columns:
                            data[ticker] = raw_data['Close']
                    else:
                        if ticker in raw_data.columns.get_level_values(0):
                            ticker_df = raw_data[ticker]
                            if 'Close' in ticker_df.columns:
                                data[ticker] = ticker_df['Close']
                except Exception as e:
                    print(f"[YTD] WARNING: Skipped {ticker}: {str(e)}")
                    continue
            
            if data.empty:
                print("[YTD] ERROR: No price data could be extracted.")
                return {'portfolio_ytd': 0.0, 'spy_ytd': 0.0, 'outperformance': 0.0}
            
            # Calculate SPY YTD
            if 'SPY' in data.columns:
                spy_start = data['SPY'].iloc[0]
                spy_end = data['SPY'].iloc[-1]
                spy_ytd = ((spy_end - spy_start) / spy_start) * 100
                print(f"[YTD] SPY YTD: {spy_ytd:.2f}%")
            else:
                print("[YTD] WARNING: SPY data not available")
                spy_ytd = 0.0
            
            # Calculate portfolio YTD (weighted by current holdings)
            portfolio_ytd = 0.0
            total_weight = 0.0
            
            for _, row in portfolio_df.iterrows():
                ticker = row['ticker']
                shares = row['shares']
                
                if ticker in data.columns and len(data[ticker].dropna()) > 0:
                    # Get start and end prices
                    ticker_data = data[ticker].dropna()
                    start_price = ticker_data.iloc[0]
                    end_price = ticker_data.iloc[-1]
                    
                    # Calculate return for this stock
                    stock_return = ((end_price - start_price) / start_price) * 100
                    
                    # Weight by current position value
                    position_value = end_price * shares
                    portfolio_ytd += stock_return * position_value
                    total_weight += position_value
                    
                    print(f"[YTD] {ticker}: {stock_return:.2f}% (weight: ${position_value:,.2f})")
            
            # Calculate weighted average return
            if total_weight > 0:
                portfolio_ytd = portfolio_ytd / total_weight
                print(f"[YTD] Portfolio YTD: {portfolio_ytd:.2f}%")
            else:
                print("[YTD] WARNING: No valid position data")
                portfolio_ytd = 0.0
            
            # Calculate outperformance
            outperformance = portfolio_ytd - spy_ytd
            
            print(f"[YTD] ✓ Calculation complete. Outperformance: {outperformance:+.2f}%")
            
            return {
                'portfolio_ytd': round(portfolio_ytd, 2),
                'spy_ytd': round(spy_ytd, 2),
                'outperformance': round(outperformance, 2)
            }
            
        except Exception as e:
            print(f"[YTD] ERROR: Could not calculate YTD performance: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'portfolio_ytd': 0.0, 'spy_ytd': 0.0, 'outperformance': 0.0}
    
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
