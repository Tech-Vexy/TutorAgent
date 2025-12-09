"""
OpenTelemetry setup for TutorAgent using LangSmith's native OTEL integration.

This module configures OpenTelemetry tracing for LangChain/LangGraph applications
using the built-in langsmith[otel] package. This provides automatic instrumentation
for chains, tools, LLM calls, and more.

Documentation: https://docs.langchain.com/langsmith/trace-with-opentelemetry
"""

import os
import logging

logger = logging.getLogger("OpenTelemetrySetup")


def setup_opentelemetry(app=None):
    """
    Configures OpenTelemetry for LangChain/LangGraph using LangSmith's built-in OTEL support.
    
    This is much simpler than manual instrumentation - just set environment variables
    and langsmith[otel] handles the rest automatically.
    
    Args:
        app: Optional FastAPI app (not needed for LangSmith OTEL)
    
    Environment Variables Required:
        LANGSMITH_OTEL_ENABLED: Set to "true" to enable OTEL tracing
        LANGSMITH_TRACING: Set to "true" to enable tracing
        LANGSMITH_API_KEY: Your LangSmith API key
        LANGSMITH_ENDPOINT: (Optional) Default is https://api.smith.langchain.com
        LANGSMITH_PROJECT: (Optional) Project name for organizing traces
    
    The langsmith[otel] package automatically:
    - Instruments LangChain chains, agents, and tools
    - Captures LLM calls with prompts and responses
    - Tracks token usage and latency
    - Sends traces to LangSmith dashboard
    """
    try:
        # Check if OTEL is enabled
        otel_enabled = os.getenv("LANGSMITH_OTEL_ENABLED", "false").lower() == "true"
        
        if not otel_enabled:
            logger.info("OpenTelemetry is disabled (LANGSMITH_OTEL_ENABLED not set to true)")
            return
        
        # Check if tracing is enabled
        tracing_enabled = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "false")).lower() == "true"
        
        if not tracing_enabled:
            logger.warning("LANGSMITH_OTEL_ENABLED is true but LANGSMITH_TRACING is false. Tracing will not work.")
            return
        
        # Check for API key
        api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        if not api_key:
            logger.warning("LANGSMITH_API_KEY not set. OpenTelemetry tracing will not send data.")
            return
        
        # Get configuration
        endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        project = os.getenv("LANGSMITH_PROJECT", os.getenv("LANGCHAIN_PROJECT", "TopScore-AI"))
        
        logger.info("=" * 60)
        logger.info("OpenTelemetry Configuration (LangSmith Native)")
        logger.info("=" * 60)
        logger.info(f"OTEL Enabled: {otel_enabled}")
        logger.info(f"Tracing Enabled: {tracing_enabled}")
        logger.info(f"Endpoint: {endpoint}")
        logger.info(f"Project: {project}")
        logger.info(f"API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("📊 Automatic instrumentation active for:")
        logger.info("  ✓ LangChain chains and agents")
        logger.info("  ✓ LLM calls (Groq, OpenAI, Google, etc.)")
        logger.info("  ✓ Tools and function calls")
        logger.info("  ✓ Retrievers and vector stores")
        logger.info("  ✓ Prompts and prompt templates")
        logger.info("")
        logger.info(f"🔍 View traces at: https://smith.langchain.com/")
        logger.info("=" * 60)
        
        # Note: With langsmith[otel], instrumentation happens automatically
        # when the environment variables are set. No manual setup needed!
        # The package uses OpenTelemetry under the hood and sends traces to LangSmith.
        
    except Exception as e:
        logger.error(f"Failed to setup OpenTelemetry: {e}", exc_info=True)


def get_otel_config():
    """
    Get current OpenTelemetry configuration.
    
    Returns:
        dict: Configuration dictionary
    """
    return {
        "otel_enabled": os.getenv("LANGSMITH_OTEL_ENABLED", "false").lower() == "true",
        "tracing_enabled": os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "false")).lower() == "true",
        "endpoint": os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        "project": os.getenv("LANGSMITH_PROJECT", os.getenv("LANGCHAIN_PROJECT", "TopScore-AI")),
        "has_api_key": bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")),
    }
