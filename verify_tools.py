
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Verify imports
try:
    from mcp import ClientSession
    print("✅ MCP library found.")
except ImportError:
    print("❌ MCP library NOT found (Agent will skip MCP tools).")

async def verify_tool_integration():
    from model_manager import model_manager
    model = model_manager.get_model("smart")
    
    print(f"🤖 Smart Model: {model_manager.smart_model_id}")
    if hasattr(model, "model_kwargs"):
        print(f"⚙️ Reasoning Effort: {model.model_kwargs.get('reasoning_effort', 'N/A')}")
        
    # Check if tools are bindable
    from tools import web_search, run_python
    tools = [web_search, run_python]
    
    try:
        bindable = model.bind_tools(tools)
        print("✅ Tools successfully bound to model (Local Tool Calling supported).")
    except Exception as e:
        print(f"❌ Tool binding failed: {e}")

if __name__ == "__main__":
    asyncio.run(verify_tool_integration())
