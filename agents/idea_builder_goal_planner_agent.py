import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

@dataclass
class InvestmentIdea:
    thesis: str
    timeframe: str
    risk_level: str
    investment_types: List[str]
    position_size: float
    created_at: datetime
    analysis: Optional[str] = None
    status: str = "draft"

class IdeaBuilderAgent:
    def __init__(self):
        """Initialize the Idea Builder agent with necessary components."""
        self.OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")
        self.llm = ChatOpenAI(
            model="gpt-4",
            api_key=self.OPEN_AI_API_KEY,
            temperature=0.3
        )
        
    def analyze_idea(self, idea: InvestmentIdea) -> str:
        """
        Analyze an investment idea and provide comprehensive feedback.
        
        Args:
            idea: InvestmentIdea object containing the investment thesis and parameters
            
        Returns:
            str: Detailed analysis and recommendations
        """
        analysis_prompt = ChatPromptTemplate.from_template("""
        You are an expert investment advisor. Analyze the following investment idea and provide detailed feedback:

        Investment Thesis:
        {thesis}

        Parameters:
        - Timeframe: {timeframe}
        - Risk Level: {risk_level}
        - Investment Types: {investment_types}
        - Position Size: {position_size}%

        Please provide a comprehensive analysis including:
        1. Thesis Strength Assessment
        2. Risk/Reward Analysis
        3. Timeline Considerations
        4. Position Sizing Recommendations
        5. Implementation Strategy
        6. Key Risk Factors to Monitor
        7. Success Metrics and Exit Criteria

        Be specific, actionable, and highlight both potential opportunities and risks.
        """)
        
        chain = analysis_prompt | self.llm | StrOutputParser()
        
        analysis = chain.invoke({
            "thesis": idea.thesis,
            "timeframe": idea.timeframe,
            "risk_level": idea.risk_level,
            "investment_types": ", ".join(idea.investment_types),
            "position_size": idea.position_size
        })
        
        return analysis

    def save_idea(self, idea: InvestmentIdea, file_path: str) -> bool:
        """
        Save an investment idea to storage.
        
        Args:
            idea: InvestmentIdea object to save
            file_path: Path to save the idea
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Convert idea to dictionary
            idea_dict = {
                "thesis": idea.thesis,
                "timeframe": idea.timeframe,
                "risk_level": idea.risk_level,
                "investment_types": idea.investment_types,
                "position_size": idea.position_size,
                "created_at": idea.created_at.isoformat(),
                "analysis": idea.analysis,
                "status": idea.status
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(idea_dict, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving idea: {str(e)}")
            return False

    def load_idea(self, file_path: str) -> Optional[InvestmentIdea]:
        """
        Load an investment idea from storage.
        
        Args:
            file_path: Path to load the idea from
            
        Returns:
            Optional[InvestmentIdea]: Loaded idea or None if loading fails
        """
        try:
            with open(file_path, 'r') as f:
                idea_dict = json.load(f)
            
            return InvestmentIdea(
                thesis=idea_dict["thesis"],
                timeframe=idea_dict["timeframe"],
                risk_level=idea_dict["risk_level"],
                investment_types=idea_dict["investment_types"],
                position_size=idea_dict["position_size"],
                created_at=datetime.fromisoformat(idea_dict["created_at"]),
                analysis=idea_dict.get("analysis"),
                status=idea_dict.get("status", "draft")
            )
            
        except Exception as e:
            print(f"Error loading idea: {str(e)}")
            return None

    def validate_idea(self, idea: InvestmentIdea) -> List[str]:
        """
        Validate an investment idea for completeness and correctness.
        
        Args:
            idea: InvestmentIdea to validate
            
        Returns:
            List[str]: List of validation errors, empty if valid
        """
        errors = []
        
        if not idea.thesis or len(idea.thesis.strip()) < 10:
            errors.append("Investment thesis is required and should be detailed")
            
        if not idea.investment_types:
            errors.append("At least one investment type must be selected")
            
        if not 0 <= idea.position_size <= 100:
            errors.append("Position size must be between 0% and 100%")
            
        return errors

    def generate_implementation_plan(self, idea: InvestmentIdea) -> str:
        """
        Generate a detailed implementation plan for an investment idea.
        
        Args:
            idea: InvestmentIdea to create plan for
            
        Returns:
            str: Detailed implementation plan
        """
        plan_prompt = ChatPromptTemplate.from_template("""
        Create a detailed implementation plan for the following investment idea:

        Investment Thesis:
        {thesis}

        Parameters:
        - Timeframe: {timeframe}
        - Risk Level: {risk_level}
        - Investment Types: {investment_types}
        - Position Size: {position_size}%

        Please provide a step-by-step implementation plan including:
        1. Pre-Investment Checklist
        2. Entry Strategy
           - Timing considerations
           - Order types and execution strategy
           - Position building approach
        3. Monitoring Plan
           - Key metrics to track
           - Risk indicators to watch
           - Regular review schedule
        4. Exit Strategy
           - Profit targets
           - Stop-loss levels
           - Position adjustment criteria
        5. Risk Management Rules
           - Position size management
           - Hedging considerations (if applicable)
           - Correlation with existing positions

        Make the plan specific, actionable, and include concrete numbers/levels where appropriate.
        """)
        
        chain = plan_prompt | self.llm | StrOutputParser()
        
        plan = chain.invoke({
            "thesis": idea.thesis,
            "timeframe": idea.timeframe,
            "risk_level": idea.risk_level,
            "investment_types": ", ".join(idea.investment_types),
            "position_size": idea.position_size
        })
        
        return plan
