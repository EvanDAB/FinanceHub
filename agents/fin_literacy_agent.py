
import os
import streamlit as st
from typing import List, Dict
from datetime import datetime

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import TavilySearchAPIRetriever  # Updated import path
from util.literacy.pdf_reader import read_pdf  # Updated to use absolute import from util

class FinancialLiteracyBot:
    def __init__(self):
        self.OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")
        self.TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(api_key=self.OPEN_AI_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.llm = ChatOpenAI(
            model="gpt-4",  # Using GPT-4 for better comprehension
            api_key=self.OPEN_AI_API_KEY,
            temperature=0.3
        )
        
        # Initialize search retriever
        self.web_retriever = TavilySearchAPIRetriever(
            api_key=self.TAVILY_API_KEY,
            k=5  # Get top 5 results
        )
        
        # Initialize FAISS vectorstore
        self.vectorstore = None
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load and process financial knowledge base documents"""
        documents = []
        # Get absolute path to the data directory in agent_3
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to project root
        data_path = os.path.join(base_dir, 'agent_3', 'data')
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            
        text_files = [
            "cj-fl-terms.txt",
            "consumer-finance-glossary.txt",
            "hbs-cheat-sheet.txt"
        ]
        
        for file in text_files:
            file_path = os.path.join(data_path, file)
            try:
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                st.warning(f"Error loading {file}: {str(e)}")
        
        # Load PDFs if any exist in data directory
        pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            file_path = os.path.join(data_path, pdf_file)
            try:
                pdf_text = read_pdf(file_path)
                if pdf_text:
                    documents.append({
                        'page_content': pdf_text,
                        'metadata': {'source': pdf_file}
                    })
            except Exception as e:
                st.warning(f"Error loading PDF {pdf_file}: {str(e)}")
        
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        
        # Create FAISS index
        self.vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
    
    def create_chat_chain(self):
        """Create the chat response chain"""
        prompt_template = """You are an expert financial educator specializing in hedge fund terminology and advanced financial concepts.
        Your goal is to explain complex financial terms in a clear, accessible way while maintaining accuracy.

        Use the following information to provide a detailed explanation:

        Context from knowledge base and web search:
        {context}

        User Question: {question}

        Additional instructions:
        1. If the term is commonly used in hedge funds, highlight that specifically
        2. Provide a simple explanation first, followed by more technical details
        3. If relevant, include:
           - Common usage examples
           - Related terms
           - Why this concept is important
        4. If the information seems incomplete, acknowledge that and stick to what you know
        5. Use analogies when helpful for complex concepts

        Keep your response clear and educational but maintain professional accuracy."""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        return prompt | self.llm | StrOutputParser()

    def get_response(self, user_question: str) -> str:
        """Generate a response using both local knowledge and web search"""
        try:
            # Get relevant documents from local knowledge base
            local_docs = self.vectorstore.similarity_search(user_question, k=3)
            local_context = "\n\n".join(doc.page_content for doc in local_docs)
            
            # Get web search results
            web_results = self.web_retriever.invoke(
                f"hedge fund term financial definition {user_question}"
            )
            web_context = "\n\n".join(doc.page_content for doc in web_results)
            
            # Combine contexts
            combined_context = f"Local Knowledge:\n{local_context}\n\nWeb Sources:\n{web_context}"
            
            # Create and run the chain
            chain = self.create_chat_chain()
            response = chain.invoke({
                "context": combined_context,
                "question": user_question
            })
            
            return response
            
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try rephrasing your question."

# # Initialize session state for the bot
# if 'fin_bot' not in st.session_state:
#     st.session_state.fin_bot = FinancialLiteracyBot()
#     st.session_state.messages = []

# # Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Chat input
# if prompt := st.chat_input("Ask about any financial term..."):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # Generate and display assistant response
#     with st.chat_message("assistant"):
#         with st.spinner("Researching..."):
#             response = st.session_state.fin_bot.get_response(prompt)
#             st.markdown(response)
#             st.session_state.messages.append({"role": "assistant", "content": response})

# # Initialize session state for the bot
# if 'fin_bot' not in st.session_state:
#     st.session_state.fin_bot = FinancialLiteracyBot()
#     st.session_state.messages = []