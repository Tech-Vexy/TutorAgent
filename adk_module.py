
import os
import logging
from google.adk.agents.llm_agent import LlmAgent
from google.adk.core.agent import AgentConfig

logger = logging.getLogger("GoogleADKAgent")

# Simple tool definition
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def create_adk_agent():
    """
    Creates and configures a Google ADK Agent.
    """
    try:
        # Check for API Key (ADK might need GOOGLE_API_KEY or similar)
        # Assuming it uses standard GOOGLE_API_KEY env var
        if not os.getenv("GOOGLE_API_KEY"):
            logger.warning("GOOGLE_API_KEY not found. ADK agent might fail.")

        # Define configuration
        # Note: API might differ slightly based on version, using standard pattern
        config = AgentConfig(
            name="TopScore-ADK",
            model="gemini-1.5-flash", 
            system_prompt="You are a helpful AI Tutor built with Google ADK."
        )

        # Create Agent
        agent = LlmAgent(config=config)
        
        # Register tools
        agent.add_tool(calculator)
        
        return agent
    except Exception as e:
        logger.error(f"Failed to create ADK agent: {e}")
        return None

# Global instance
adk_agent = None

def get_adk_response(user_message: str):
    """
    Process a message using the ADK agent.
    """
    global adk_agent
    if not adk_agent:
        adk_agent = create_adk_agent()
    
    if not adk_agent:
        return "Error: ADK Agent could not be initialized."
    
    try:
        # Run agent
        response = adk_agent.run(user_message)
        return response
    except Exception as e:
        return f"Error running ADK agent: {e}"
