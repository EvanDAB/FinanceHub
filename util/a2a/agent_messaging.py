from typing import Dict, List

class AgentMessage:
    def __init__(self, sender: str, intent: str, content: Dict):
        self.sender = sender
        self.intent = intent
        self.content = content
        
    def to_dict(self) -> Dict:
        return {
            "sender": self.sender,
            "intent": self.intent,
            "content": self.content
        }
        
    @staticmethod
    def from_dict(data: Dict) -> 'AgentMessage':
        return AgentMessage(
            sender=data["sender"],
            intent=data["intent"],
            content=data["content"]
        )