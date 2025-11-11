import os
import gradio as gr
import finnhub
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS, InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize API keys
FH_API_KEY = os.environ.get("FINNHUB_API_KEY")
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")

class FinancialNewsBot:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=OPEN_AI_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPEN_AI_API_KEY)
        self.finnhub_client = finnhub.Client(api_key=FH_API_KEY)
        self.vectorstore = None
        self.setup_retriever()
        
    def setup_retriever(self):
        # Get latest news
        finnhub_gen_news_resp = self.finnhub_client.general_news('business', min_id=0)[:5]
        url_list = [item['url'] for item in finnhub_gen_news_resp]
        
        # Load and process documents
        doc_list = [WebBaseLoader(url).load() for url in url_list]
        format_doc_list = [item for sublist in doc_list for item in sublist]
        doc_splits = self.text_splitter.split_documents(format_doc_list)
        
        # Setup vector store and retriever
        self.vectorstore = InMemoryVectorStore.from_documents(
            documents=doc_splits,
            embedding=self.embeddings
        )
        retriever = self.vectorstore.as_retriever()
        self.retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_news",
            "Search and return information that is relevant to market news"
        )
        
    def create_prompt(self, query_type="general"):
        if query_type == "market_analysis":
            return """You are a financial analyst tasked with analyzing market news.
            Based on the following market news information, provide a concise analysis:
            
            {retrieved_info}
            
            Focus on:
            1. Key market-moving events
            2. Major sector impacts
            3. Overall market sentiment
            4. Potential market implications
            
            Keep the analysis professional and fact-based."""
        else:
            return """You are a helpful financial news assistant. Based on the following news information,
            provide a clear and informative response to the user's question:
            
            {retrieved_info}
            
            User Question: {user_question}
            
            Provide a concise and relevant answer focusing on the requested information."""

    def get_response(self, user_input):
        try:
            # Create prompt template based on input
            prompt_template = self.create_prompt()
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            # Create chain
            chain = prompt | self.llm | StrOutputParser()
            
            # Retrieve relevant information
            retrieved_info = self.retriever_tool.invoke({"query": user_input})
            
            # Generate response
            response = chain.invoke({
                "retrieved_info": retrieved_info,
                "user_question": user_input
            })
            
            return response
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again with a different question."

def chat_interface(message, history):
    bot = FinancialNewsBot()
    response = bot.get_response(message)
    return response

# Create and launch the Gradio interface
iface = gr.ChatInterface(
    chat_interface,
    title="Financial News Assistant",
    description="Ask me anything about current market news and financial information.",
    theme="soft",
    examples=[
        "What are the major market events today?",
        "How are tech stocks performing?",
        "What's the current market sentiment?",
        "Any significant mergers or acquisitions today?",
    ]
)

if __name__ == "__main__":
    iface.launch(share=True)