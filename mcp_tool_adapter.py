
import os
import sys
from contextlib import AsyncExitStack
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Global session management
_exit_stack = AsyncExitStack()
_session = None

async def get_mcp_tools():
    """
    Connects to the Tavily MCP Server and returns a list of LangChain-compatible tools.
    Maintains a persistent connection using global state.
    """
    global _session, _exit_stack
    
    # Check API Key
    if not os.getenv("TAVILY_API_KEY"):
        # print("Warning: TAVILY_API_KEY not found. MCP tools disabled.")
        return []

    if _session:
        try:
            # Re-generate tools from existing session
            return await load_mcp_tools(_session)
        except Exception:
            # Session likely dead, reset
            _session = None
            # Ideally we should close stack, but let's just create new stack
            _exit_stack = AsyncExitStack()

    # Parameters for the server process
    python_exe = sys.executable
    server_params = StdioServerParameters(
        command=python_exe,
        args=["-m", "mcp_server_tavily"],
        env=os.environ.copy()
    )

    try:
        # Establish connection
        read, write = await _exit_stack.enter_async_context(stdio_client(server_params))
        session = await _exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        
        _session = session
        
        return await load_mcp_tools(_session)

    except Exception as e:
        # print(f"Error loading MCP tools: {e}")
        # Return empty list to avoid crashing the agent
        return []
