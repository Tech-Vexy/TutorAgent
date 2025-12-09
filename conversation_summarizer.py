
"""
Conversation Summarizer Module

Provides intelligent summarization of conversation history to maintain
context within token limits while preserving important information.
"""

import os
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing import List, Tuple

# Summary generation prompt
SUMMARIZER_PROMPT = """You are a conversation summarizer for an AI tutoring system.
Summarize the following conversation history concisely, preserving:
1. Key topics discussed
2. Important facts or concepts explained
3. User's learning goals or struggles
4. Any promises or commitments made by the tutor

Keep it under 500 words. Focus on educational context.

CONVERSATION:
{conversation}

CONCISE SUMMARY:"""


class ConversationSummarizer:
    def __init__(self, max_messages_before_summary: int = 15, summary_trigger: int = 20):
        """
        Args:
            max_messages_before_summary: Keep this many recent messages unsummarized
            summary_trigger: When total messages exceed this, trigger summarization
        """
        self.max_messages_before_summary = max_messages_before_summary
        self.summary_trigger = summary_trigger
        self._summaries = {}  # thread_id -> summary string
        
    def should_summarize(self, messages: List[BaseMessage]) -> bool:
        """Check if we should summarize based on message count."""
        return len(messages) > self.summary_trigger
    
    def get_existing_summary(self, thread_id: str) -> str:
        """Get existing summary for a thread."""
        return self._summaries.get(thread_id, "")
    
    def set_summary(self, thread_id: str, summary: str):
        """Store summary for a thread."""
        self._summaries[thread_id] = summary
        
    async def summarize_and_compact(self, messages: List[BaseMessage], thread_id: str, llm) -> Tuple[List[BaseMessage], str]:
        """
        Summarizes older messages and returns compacted message list.
        
        Returns:
            Tuple of (compacted_messages, new_summary)
        """
        if not self.should_summarize(messages):
            return messages, self.get_existing_summary(thread_id)
            
        # Split messages
        old_messages = messages[:-self.max_messages_before_summary]
        recent_messages = messages[-self.max_messages_before_summary:]
        
        # Build conversation text for summarization
        conv_text = ""
        existing_summary = self.get_existing_summary(thread_id)
        if existing_summary:
            conv_text += f"[Previous Summary: {existing_summary}]\n\n"
            
        for msg in old_messages:
            role = "User" if isinstance(msg, HumanMessage) else "Tutor"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            conv_text += f"{role}: {content}\n"
        
        # Generate summary using provided LLM
        try:
            prompt = SUMMARIZER_PROMPT.format(conversation=conv_text)
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            new_summary = response.content.strip()
        except Exception as e:
            print(f"Summarization failed: {e}")
            new_summary = existing_summary or "Previous conversation about tutoring."
            
        self.set_summary(thread_id, new_summary)
        
        # Build compacted message list with summary as first system message
        compacted = []
        if new_summary:
            compacted.append(SystemMessage(content=f"[Conversation Summary: {new_summary}]"))
        compacted.extend(recent_messages)
        
        return compacted, new_summary


# Singleton instance
conversation_summarizer = ConversationSummarizer()
