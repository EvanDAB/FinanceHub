"""
Unit test for portfolio beta calculation
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

def test_beta_calculation():
    """Test the beta calculation logic independently"""
    
    # Create sample portfolio
    portfolio_df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'shares': [100, 75, 25]
    })
    
    print("=" * 60)
    print("TESTING PORTFOLIO BETA CALCULATION")
    print("=" * 60)
    print(f"\nPortfolio: {portfolio_df['ticker'].tolist()}")
    
    # Download historical data
    tickers = portfolio_df['ticker'].tolist() + ['SPY']
    print(f"\nDownloading data for: {tickers}")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"Period: {start_date.date()} to {end_date.date()}")
    
    # Test different parameter combinations
    print("\n" + "=" * 60)
    print("TEST 1: group_by='ticker', auto_adjust=True")
    print("=" * 60)
    
    raw_data = yf.download(
        tickers, 
        start=start_date, 
        end=end_date, 
        progress=False,
        auto_adjust=True,
        group_by='ticker'
    )
    
    print(f"\nData shape: {raw_data.shape}")
    print(f"Data empty: {raw_data.empty}")
    print(f"\nColumn structure:")
    print(f"  Columns: {raw_data.columns.tolist()[:10]}...")
    print(f"  Column levels: {raw_data.columns.nlevels}")
    
    if raw_data.columns.nlevels > 1:
        print(f"  Level 0 (tickers): {raw_data.columns.get_level_values(0).unique().tolist()}")
        print(f"  Level 1 (prices): {raw_data.columns.get_level_values(1).unique().tolist()}")
    else:
        print(f"  Flat columns: {raw_data.columns.tolist()}")
    
    print(f"\nFirst few rows:")
    print(raw_data.head())
    
    # Try to extract close prices
    print("\n" + "-" * 60)
    print("Attempting to extract Close prices...")
    print("-" * 60)
    
    data = pd.DataFrame()
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}:")
        try:
            if len(tickers) == 1:
                print(f"  Single ticker mode")
                print(f"  Available columns: {raw_data.columns.tolist()}")
                if 'Close' in raw_data.columns:
                    data[ticker] = raw_data['Close']
                    print(f"  ✓ Extracted 'Close' column")
                elif 'Adj Close' in raw_data.columns:
                    data[ticker] = raw_data['Adj Close']
                    print(f"  ✓ Extracted 'Adj Close' column")
                else:
                    print(f"  ✗ Neither 'Close' nor 'Adj Close' found!")
            else:
                print(f"  Multi-ticker mode")
                if ticker in raw_data.columns.get_level_values(0):
                    ticker_df = raw_data[ticker]
                    print(f"  Available columns for {ticker}: {ticker_df.columns.tolist()}")
                    
                    if 'Close' in ticker_df.columns:
                        data[ticker] = ticker_df['Close']
                        print(f"  ✓ Extracted 'Close' column")
                    elif 'Adj Close' in ticker_df.columns:
                        data[ticker] = ticker_df['Adj Close']
                        print(f"  ✓ Extracted 'Adj Close' column")
                    else:
                        print(f"  ✗ Neither 'Close' nor 'Adj Close' found!")
                else:
                    print(f"  ✗ {ticker} not found in level 0 columns!")
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    print("\n" + "-" * 60)
    print("Extracted data summary:")
    print("-" * 60)
    print(f"Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(data.head())
    
    if not data.empty and len(data.columns) > 0:
        # Calculate returns
        returns = data.pct_change().dropna()
        print(f"\nReturns shape: {returns.shape}")
        
        # Calculate beta for first stock as example
        if 'SPY' in returns.columns and len(returns.columns) > 1:
            test_ticker = [t for t in returns.columns if t != 'SPY'][0]
            
            stock_returns = returns[test_ticker].values
            market_returns = returns['SPY'].values
            
            covariance = np.cov(stock_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)
            
            if market_variance > 0:
                beta = covariance / market_variance
                print(f"\n{test_ticker} Beta: {beta:.2f}")
            
            print("\n✓ Beta calculation successful!")
    
    # Test alternative: group_by='column'
    print("\n\n" + "=" * 60)
    print("TEST 2: group_by='column', auto_adjust=True")
    print("=" * 60)
    
    raw_data2 = yf.download(
        tickers, 
        start=start_date, 
        end=end_date, 
        progress=False,
        auto_adjust=True,
        group_by='column'
    )
    
    print(f"\nData shape: {raw_data2.shape}")
    print(f"\nColumn structure:")
    print(f"  Columns: {raw_data2.columns.tolist()[:10]}...")
    print(f"  Column levels: {raw_data2.columns.nlevels}")
    
    if raw_data2.columns.nlevels > 1:
        print(f"  Level 0: {raw_data2.columns.get_level_values(0).unique().tolist()}")
        print(f"  Level 1: {raw_data2.columns.get_level_values(1).unique().tolist()}")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_beta_calculation()
