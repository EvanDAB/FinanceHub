from typing import Dict, List, Optional
from util.a2a.agent_messaging import AgentMessage

def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
    """Handle incoming messages from other agents"""
    if message.intent == "GET_HOLDINGS":
        # Load current portfolio
        portfolio_df = self.load_portfolio()
        
        # Extract symbols and shares
        holdings = {
            'symbols': portfolio_df['ticker'].tolist(),
            'shares': portfolio_df['shares'].tolist()
        }
        
        # Return portfolio data
        return AgentMessage(
            sender="portfolio_agent",
            intent="HOLDINGS_DATA",
            content=holdings
        )
    return None