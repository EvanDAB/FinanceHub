from typing import Dict, List, Optional
from util.a2a.agent_messaging import AgentMessage

def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
    """Handle incoming messages from other agents"""
    if message.intent == "HOLDINGS_DATA":
        # Extract symbols from the message
        symbols = message.content.get('symbols', [])
        
        if not symbols:
            return AgentMessage(
                sender="news_agent",
                intent="NEWS_ANALYSIS",
                content={"error": "No symbols provided"}
            )
            
        # Create a targeted query for the portfolio
        symbols_str = ', '.join(symbols)
        query = f"Find and analyze recent important news about these companies: {symbols_str}"
        
        try:
            # Get relevant news using our existing tools
            retrieved_info = self.retriever_tool.invoke({"query": query})
            
            # Analyze the news
            analysis = self.news_chain.invoke({
                "input": query,
                "retrieved_info": retrieved_info
            })
            
            return AgentMessage(
                sender="news_agent",
                intent="NEWS_ANALYSIS",
                content={
                    "analysis": analysis,
                    "symbols": symbols,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            return AgentMessage(
                sender="news_agent",
                intent="NEWS_ANALYSIS",
                content={"error": str(e)}
            )
            
    return None

def get_portfolio_news(self, portfolio_agent) -> Dict:
    """
    Initiate communication with portfolio agent to get news about holdings
    """
    # Request holdings data from portfolio agent
    request = AgentMessage(
        sender="news_agent",
        intent="GET_HOLDINGS",
        content={}
    )
    
    # Get holdings data
    holdings_response = portfolio_agent.handle_message(request)
    if not holdings_response:
        return {"error": "No response from portfolio agent"}
        
    # Process the holdings data and return news analysis
    news_analysis = self.handle_message(holdings_response)
    return news_analysis.content if news_analysis else {"error": "Failed to analyze news"}