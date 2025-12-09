
"""
Adaptive Learning Profile Module

Tracks user learning patterns, strengths, weaknesses, and adjusts
the tutoring approach accordingly.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

PROFILES_DIR = Path("learning_profiles")
PROFILES_DIR.mkdir(exist_ok=True)


class LearningProfile:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.profile_path = PROFILES_DIR / f"{user_id}.json"
        self.data = self._load()
        
    def _load(self) -> Dict:
        """Load profile from disk or create new one."""
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r") as f:
                    return json.load(f)
            except:
                pass
        return self._create_default()
    
    def _create_default(self) -> Dict:
        return {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "topics": {},  # topic -> {correct, incorrect, last_seen}
            "preferred_style": "balanced",  # visual, textual, interactive
            "difficulty_level": "medium",  # easy, medium, hard
            "strengths": [],
            "weaknesses": [],
            "total_sessions": 0,
            "total_questions": 0,
            "streak_days": 0,
            "last_active": None
        }
    
    def save(self):
        """Persist profile to disk."""
        with open(self.profile_path, "w") as f:
            json.dump(self.data, f, indent=2)
    
    def record_interaction(self, topic: str, was_correct: bool = True, difficulty: str = None):
        """Record a learning interaction for a topic."""
        if topic not in self.data["topics"]:
            self.data["topics"][topic] = {"correct": 0, "incorrect": 0, "attempts": 0}
        
        self.data["topics"][topic]["attempts"] += 1
        if was_correct:
            self.data["topics"][topic]["correct"] += 1
        else:
            self.data["topics"][topic]["incorrect"] += 1
            
        self.data["topics"][topic]["last_seen"] = datetime.now().isoformat()
        self.data["total_questions"] += 1
        self.data["last_active"] = datetime.now().isoformat()
        
        # Update strengths/weaknesses
        self._analyze_performance()
        self.save()
    
    def _analyze_performance(self):
        """Analyze topic performance to identify strengths and weaknesses."""
        strengths = []
        weaknesses = []
        
        for topic, stats in self.data["topics"].items():
            if stats["attempts"] >= 3:  # Need enough data
                accuracy = stats["correct"] / stats["attempts"]
                if accuracy >= 0.8:
                    strengths.append(topic)
                elif accuracy < 0.5:
                    weaknesses.append(topic)
        
        self.data["strengths"] = strengths[:5]  # Top 5
        self.data["weaknesses"] = weaknesses[:5]
    
    def get_learning_context(self) -> str:
        """Generate context string for the LLM about this learner."""
        if not self.data["topics"]:
            return ""
            
        ctx_parts = []
        
        if self.data["weaknesses"]:
            ctx_parts.append(f"Struggles with: {', '.join(self.data['weaknesses'])}")
        if self.data["strengths"]:
            ctx_parts.append(f"Strong in: {', '.join(self.data['strengths'])}")
        
        ctx_parts.append(f"Preferred style: {self.data['preferred_style']}")
        ctx_parts.append(f"Overall level: {self.data['difficulty_level']}")
        
        return "\n".join(ctx_parts)
    
    def set_preference(self, key: str, value: str):
        """Update user preference."""
        if key in ["preferred_style", "difficulty_level"]:
            self.data[key] = value
            self.save()
    
    def get_suggested_topics(self) -> List[str]:
        """Suggest topics to review based on weaknesses or unseen topics."""
        suggestions = []
        
        # Prioritize weaknesses
        suggestions.extend(self.data["weaknesses"][:3])
        
        # Add topics not seen recently
        for topic, stats in sorted(
            self.data["topics"].items(), 
            key=lambda x: x[1].get("last_seen", ""), 
            reverse=False
        )[:2]:
            if topic not in suggestions:
                suggestions.append(topic)
                
        return suggestions


class LearningProfileManager:
    _profiles: Dict[str, LearningProfile] = {}
    
    @classmethod
    def get_profile(cls, user_id: str) -> LearningProfile:
        if user_id not in cls._profiles:
            cls._profiles[user_id] = LearningProfile(user_id)
        return cls._profiles[user_id]


# Convenience function
def get_learning_profile(user_id: str) -> LearningProfile:
    return LearningProfileManager.get_profile(user_id)
