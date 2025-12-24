"""
Test beta calculation with the actual sample portfolio
"""
import pandas as pd
import sys
import os

# Add parent directory to path to import the agent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.portfolio_agent import PortfolioAgent

def test_sample_portfolio():
    """Test beta calculation with the actual sample portfolio"""
    
    print("=" * 60)
    print("TESTING BETA WITH SAMPLE PORTFOLIO")
    print("=" * 60)
    
    # Create agent
    agent = PortfolioAgent()
    
    # Load the sample portfolio
    print("\nLoading sample portfolio...")
    portfolio_df = agent.load_portfolio()
    
    print(f"\nPortfolio loaded successfully:")
    print(portfolio_df)
    print(f"\nTotal tickers: {len(portfolio_df)}")
    print(f"Tickers: {portfolio_df['ticker'].tolist()}")
    
    # Try to calculate beta
    print("\n" + "=" * 60)
    print("CALCULATING BETA...")
    print("=" * 60)
    
    try:
        beta = agent.calculate_portfolio_beta(portfolio_df)
        print(f"\n✓ SUCCESS! Portfolio Beta: {beta}")
    except Exception as e:
        print(f"\n✗ FAILED! Error: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_sample_portfolio()
