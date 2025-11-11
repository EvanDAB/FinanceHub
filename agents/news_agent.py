import os
import finnhub
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools.retriever import create_retriever_tool

from util.a2a.agent_messaging import AgentMessage

class NewsAgent:
    def __init__(self):
        self.FH_API_KEY = os.environ.get("FINNHUB_API_KEY")
        self.OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")
        self.embeddings = None
        self.text_splitter = None
        self.vectorstore = None
        self.retriever_tool = None
        self.llm = None
        self.news_chain = None

    def fetch_news(self):
        """Fetch news from Finnhub API."""
        try:
            if not self.FH_API_KEY:
                raise ValueError("Finnhub API key is not set")
                
            finnhub_client = finnhub.Client(api_key=self.FH_API_KEY)
            
            # Fetch each type of news with error handling
            news_items = []
            
            try:
                business_news = finnhub_client.general_news('business', min_id=0)[:25]
                news_items.extend(business_news)
            except Exception as e:
                st.warning(f"Error fetching business news: {str(e)}")
            
            try:
                market_news = finnhub_client.general_news('technology', min_id=0)[:20]
                news_items.extend(market_news)
            except Exception as e:
                st.warning(f"Error fetching market news: {str(e)}")
            
            try:
                econ_news = finnhub_client.general_news('economic', min_id=0)[:10]
                news_items.extend(econ_news)
            except Exception as e:
                st.warning(f"Error fetching economic news: {str(e)}")
            
            if not news_items:
                raise ValueError("No news items could be fetched")
                
            return news_items
            
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return []

    @st.cache_data(ttl=300)
    def _load_documents(_self, urls):
        """Cache document loading to avoid repeated web requests."""
        docs = []
        for url in urls:
            try:
                docs.extend(WebBaseLoader(url).load())
            except Exception:
                continue
        return docs

    def initialize_components(self):
        """Initialize all news agent components."""
        try:
            if not self.OPEN_AI_API_KEY:
                raise ValueError("OpenAI API key is not set")

            # Initialize embeddings and text splitter
            self.embeddings = OpenAIEmbeddings(api_key=self.OPEN_AI_API_KEY)
            self.text_splitter = RecursiveCharacterTextSplitter()
            
            # Initialize LLM
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=self.OPEN_AI_API_KEY)
            
            # Get news data
            news_data = self.fetch_news()
            
            if not news_data:
                st.warning("No news data fetched, initializing with empty vectorstore")
                self.vectorstore = FAISS.from_texts(["No news available"], self.embeddings)
            else:
                # Process documents
                url_list = [item['url'] for item in news_data]
                doc_list = self._load_documents(url_list)
                
                if not doc_list:
                    st.warning("No documents loaded, initializing with empty vectorstore")
                    self.vectorstore = FAISS.from_texts(["No news available"], self.embeddings)
                else:
                    # Split and create vectorstore
                    doc_splits = self.text_splitter.split_documents(doc_list)
                    self.vectorstore = FAISS.from_documents(doc_splits, self.embeddings)
            
            # Create retriever tool
            if not self.vectorstore:
                raise ValueError("Vectorstore initialization failed")
                
            self.retriever_tool = create_retriever_tool(
                self.vectorstore.as_retriever(),
                "search_news",
                "Search for relevant market news and analysis"
            )
            
            # Create news chain
            self.news_chain = self.create_chain()
            
            if not all([self.embeddings, self.text_splitter, self.llm, 
                       self.vectorstore, self.retriever_tool, self.news_chain]):
                raise ValueError("One or more components failed to initialize")
                
        except Exception as e:
            st.error(f"Error initializing news components: {str(e)}")
            raise
        @st.cache_data(ttl=300)
        def load_documents(url_list):
            doc_list = []
            for url in url_list:
                try:
                    doc_list.extend(WebBaseLoader(url).load())
                except Exception:
                    continue  # Skip failed URLs silently
            return doc_list
    # except Exception as e:
    #     st.error(f"Error initializing news components: {str(e)}")
    #     # Initialize with empty vectorstore
    #     vectorstore = FAISS.from_texts(["No news available"], embeddings)
    #     retriever_tool = create_retriever_tool(
    #         vectorstore.as_retriever(),
    #         "search_news",
    #         "Search for relevant market news and analysis"
    #     )

    def create_chain(self):
        """Create and return the news analysis chain."""
        prompt_template = """You are a financial analyst tasked with analyzing market news.
        Based on the following market news information, provide a concise analysis:

        {retrieved_info}

        Focus on:
        1. Key market events and their potential impact
        2. Notable company news or earnings
        3. Economic indicators and policy changes
        4. Market sentiment and trends

        Use clear, concise bullet points."""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        return prompt | self.llm | StrOutputParser()

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

# Add a refresh button
if st.button("Refresh Analysis"):
    st.session_state.initialized = False
    st.rerun()  # Updated from experimental_rerun()