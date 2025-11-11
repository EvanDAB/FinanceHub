import os
import json
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import asdict
from agents.idea_builder_goal_planner_agent import InvestmentIdea

class IdeaStorage:
    def __init__(self, storage_dir: str = "investment_ideas"):
        """Initialize storage with directory path."""
        self.storage_dir = storage_dir
        self.ensure_storage_exists()
        
    def ensure_storage_exists(self):
        """Create storage directory if it doesn't exist."""
        os.makedirs(self.storage_dir, exist_ok=True)
        
    def _generate_filename(self, idea: InvestmentIdea) -> str:
        """Generate a unique filename for an idea."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a slug from the first few words of the thesis
        slug = "_".join(idea.thesis.split()[:5]).lower()
        # Remove special characters and limit length
        slug = "".join(c for c in slug if c.isalnum() or c == "_")[:50]
        return f"{timestamp}_{slug}.json"
        
    def save_idea(self, idea: InvestmentIdea) -> str:
        """
        Save an investment idea and return its ID.
        
        Args:
            idea: InvestmentIdea object to save
            
        Returns:
            str: The ID (filename) of the saved idea
        """
        filename = self._generate_filename(idea)
        filepath = os.path.join(self.storage_dir, filename)
        
        # Convert idea to dictionary, handling datetime
        idea_dict = asdict(idea)
        idea_dict["created_at"] = idea_dict["created_at"].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(idea_dict, f, indent=2)
            
        return filename
        
    def load_idea(self, idea_id: str) -> Optional[InvestmentIdea]:
        """
        Load an investment idea by its ID.
        
        Args:
            idea_id: ID (filename) of the idea to load
            
        Returns:
            Optional[InvestmentIdea]: The loaded idea or None if not found
        """
        filepath = os.path.join(self.storage_dir, idea_id)
        try:
            with open(filepath, 'r') as f:
                idea_dict = json.load(f)
            
            # Convert string back to datetime
            idea_dict["created_at"] = datetime.fromisoformat(idea_dict["created_at"])
            
            return InvestmentIdea(**idea_dict)
        except Exception as e:
            print(f"Error loading idea {idea_id}: {str(e)}")
            return None
            
    def list_ideas(self, status: Optional[str] = None) -> List[Dict]:
        """
        List all saved ideas, optionally filtered by status.
        
        Args:
            status: Optional status to filter by
            
        Returns:
            List[Dict]: List of ideas with metadata
        """
        ideas = []
        for filename in os.listdir(self.storage_dir):
            if not filename.endswith('.json'):
                continue
                
            try:
                filepath = os.path.join(self.storage_dir, filename)
                with open(filepath, 'r') as f:
                    idea_dict = json.load(f)
                    
                if status is None or idea_dict.get("status") == status:
                    # Add filename as ID and include basic metadata
                    idea_dict["id"] = filename
                    ideas.append({
                        "id": filename,
                        "thesis": idea_dict["thesis"][:100] + "..." if len(idea_dict["thesis"]) > 100 else idea_dict["thesis"],
                        "created_at": idea_dict["created_at"],
                        "status": idea_dict.get("status", "draft"),
                        "timeframe": idea_dict["timeframe"],
                        "risk_level": idea_dict["risk_level"]
                    })
            except Exception as e:
                print(f"Error loading idea {filename}: {str(e)}")
                continue
                
        # Sort by creation date, newest first
        return sorted(ideas, key=lambda x: x["created_at"], reverse=True)
        
    def update_idea_status(self, idea_id: str, new_status: str) -> bool:
        """
        Update the status of an idea.
        
        Args:
            idea_id: ID of the idea to update
            new_status: New status to set
            
        Returns:
            bool: True if update was successful
        """
        idea = self.load_idea(idea_id)
        if idea:
            idea.status = new_status
            self.save_idea(idea)
            return True
        return False
        
    def delete_idea(self, idea_id: str) -> bool:
        """
        Delete an investment idea.
        
        Args:
            idea_id: ID of the idea to delete
            
        Returns:
            bool: True if deletion was successful
        """
        filepath = os.path.join(self.storage_dir, idea_id)
        try:
            os.remove(filepath)
            return True
        except Exception:
            return False