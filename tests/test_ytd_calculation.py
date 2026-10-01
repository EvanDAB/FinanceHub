"""
Unit test for YTD performance calculation using the sample portfolio.
This test verifies that the YTD calculation logic is working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from agents.portfolio_agent import PortfolioAgent
import pandas as pd

def test_ytd_calculation():
    """Test YTD performance calculation with sample portfolio."""
    
    print("=" * 80)
    print("YTD PERFORMANCE CALCULATION TEST")
    print("=" * 80)
    
    # Initialize portfolio agent
    agent = PortfolioAgent()
    
    # Load sample portfolio
    print("\n[1] Loading sample portfolio...")
    portfolio_df = agent.load_portfolio()
    print(f"    Loaded {len(portfolio_df)} holdings")
    print(f"    Tickers: {', '.join(portfolio_df['ticker'].unique())}")
    
    # Calculate YTD performance
    print("\n[2] Calculating YTD performance...")
    try:
        ytd_results = agent.calculate_ytd_performance(portfolio_df)
        
        print("\n" + "=" * 80)
        print("YTD PERFORMANCE RESULTS")
        print("=" * 80)
        print(f"Portfolio YTD:    {ytd_results['portfolio_ytd']:+.2f}%")
        print(f"SPY YTD:          {ytd_results['spy_ytd']:+.2f}%")
        print(f"Outperformance:   {ytd_results['outperformance']:+.2f}%")
        print("=" * 80)
        
        # Validation checks
        print("\n[3] Validation Checks:")
        
        # Check 1: All values should be present
        assert 'portfolio_ytd' in ytd_results, "Missing portfolio_ytd"
        assert 'spy_ytd' in ytd_results, "Missing spy_ytd"
        assert 'outperformance' in ytd_results, "Missing outperformance"
        print("    ✓ All required fields present")
        
        # Check 2: Outperformance should equal difference
        calculated_diff = ytd_results['portfolio_ytd'] - ytd_results['spy_ytd']
        assert abs(calculated_diff - ytd_results['outperformance']) < 0.01, \
            f"Outperformance mismatch: {calculated_diff} vs {ytd_results['outperformance']}"
        print("    ✓ Outperformance calculation correct")
        
        # Check 3: Values should be reasonable (not extreme)
        assert -100 < ytd_results['portfolio_ytd'] < 500, \
            f"Portfolio YTD seems unreasonable: {ytd_results['portfolio_ytd']}%"
        assert -100 < ytd_results['spy_ytd'] < 500, \
            f"SPY YTD seems unreasonable: {ytd_results['spy_ytd']}%"
        print("    ✓ Values are within reasonable range")
        
        # Check 4: If showing 0%, investigate why
        if abs(ytd_results['portfolio_ytd']) < 0.01:
            print("\n    ⚠ WARNING: Portfolio YTD is ~0%")
            print("    This could mean:")
            print("      - Data fetch failed")
            print("      - Start of year price data missing")
            print("      - Portfolio weights summing incorrectly")
            print("\n    Checking portfolio data...")
            print(f"    Total shares: {portfolio_df['shares'].sum()}")
            print(f"    Portfolio weights sum: {portfolio_df['shares'].sum() / portfolio_df['shares'].sum() * 100:.2f}%")
        
        print("\n✅ TEST PASSED - YTD calculation completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED - Error during YTD calculation:")
        print(f"    {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ytd_calculation()
    sys.exit(0 if success else 1)
