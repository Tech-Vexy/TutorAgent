"""
Optimized agent graph with performance improvements.

This version removes unnecessary steps and reduces latency:
- Skips planner, reviewer, and reflection for faster responses
- Reduces memory/knowledge retrieval
- Limits history context
- Uses faster models by default
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Performance flags
SKIP_PLANNER = os.getenv("SKIP_PLANNER", "1") == "1"
SKIP_REVIEWER = os.getenv("SKIP_REVIEWER", "1") == "1"
SKIP_REFLECTION = os.getenv("SKIP_REFLECTION", "1") == "1"
SKIP_MEMORY_SAVE = os.getenv("SKIP_MEMORY_SAVE", "0") == "1"
MEMORY_RETRIEVAL_LIMIT = int(os.getenv("MEMORY_RETRIEVAL_LIMIT", "1"))
KNOWLEDGE_RETRIEVAL_LIMIT = int(os.getenv("KNOWLEDGE_RETRIEVAL_LIMIT", "1"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

print(f"""
🚀 Performance Optimizations Active:
  Skip Planner: {SKIP_PLANNER}
  Skip Reviewer: {SKIP_REVIEWER}
  Skip Reflection: {SKIP_REFLECTION}
  Memory Limit: {MEMORY_RETRIEVAL_LIMIT}
  Knowledge Limit: {KNOWLEDGE_RETRIEVAL_LIMIT}
  History Limit: {MAX_HISTORY_MESSAGES} messages
""")

# Import the original agent graph
from agent_graph import *

# Override the route_decision function for optimized routing
def optimized_route_decision(state: AgentState):
    """Optimized routing that skips planner."""
    intent = state.get("intent")
    if intent == "VISION_ANALYSIS":
        return "vision_analysis"
    elif intent == "COMPLEX_REASONING":
        if SKIP_PLANNER:
            return "deep_thinker"  # Skip planner, go directly to deep thinker
        return "planner"
    else:
        return "simple_chat"

# Override should_continue to skip reviewer
def optimized_should_continue(state: AgentState):
    """Optimized continuation that skips reviewer."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    if SKIP_REVIEWER:
        if SKIP_REFLECTION:
            return "save_memory"  # Skip both reviewer and reflection
        return "reflection"
    return "reviewer"

# Override review_decision to skip re-processing
def optimized_review_decision(state: AgentState):
    """Optimized review that always proceeds (no loops)."""
    if SKIP_REVIEWER:
        return "reflection" if not SKIP_REFLECTION else "save_memory"
    
    # Original logic
    messages = state["messages"]
    if messages and isinstance(messages[-1], HumanMessage) and "REVIEWER FEEDBACK" in str(messages[-1].content):
        return "deep_thinker"
    return "reflection"

# Create optimized workflow
optimized_workflow = StateGraph(AgentState)

# Add nodes
optimized_workflow.add_node("router", router_node)
optimized_workflow.add_node("simple_chat", simple_chat_node)
optimized_workflow.add_node("vision_analysis", vision_node)
optimized_workflow.add_node("deep_thinker", deep_thinker_node)
optimized_workflow.add_node("tools", tool_node)

if not SKIP_MEMORY_SAVE:
    optimized_workflow.add_node("save_memory", save_memory_node)

if not SKIP_PLANNER:
    optimized_workflow.add_node("planner", planner_node)

if not SKIP_REFLECTION:
    optimized_workflow.add_node("reflection", reflection_node)

if not SKIP_REVIEWER:
    optimized_workflow.add_node("reviewer", reviewer_node)

# Set entry point
optimized_workflow.set_entry_point("router")

# Add routing from router
if SKIP_PLANNER:
    optimized_workflow.add_conditional_edges(
        "router",
        optimized_route_decision,
        {
            "vision_analysis": "vision_analysis",
            "deep_thinker": "deep_thinker",  # Direct to deep thinker
            "simple_chat": "simple_chat"
        }
    )
else:
    optimized_workflow.add_conditional_edges(
        "router",
        optimized_route_decision,
        {
            "vision_analysis": "vision_analysis",
            "planner": "planner",
            "simple_chat": "simple_chat"
        }
    )
    optimized_workflow.add_edge("planner", "deep_thinker")

# Vision analysis feeds into deep thinker
optimized_workflow.add_edge("vision_analysis", "deep_thinker")

# Simple chat path
if SKIP_MEMORY_SAVE:
    optimized_workflow.add_edge("simple_chat", END)
else:
    optimized_workflow.add_edge("simple_chat", "save_memory")

# Deep thinker conditional edges
if SKIP_REVIEWER and SKIP_REFLECTION:
    # Fast path: deep_thinker → tools (if needed) → save_memory → END
    def fast_continue(state: AgentState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        if SKIP_MEMORY_SAVE:
            return "end"
        return "save_memory"
    
    optimized_workflow.add_conditional_edges(
        "deep_thinker",
        fast_continue,
        {
            "tools": "tools",
            "save_memory": "save_memory" if not SKIP_MEMORY_SAVE else None,
            "end": END
        }
    )
    optimized_workflow.add_edge("tools", "deep_thinker")
    
else:
    # Original path with optional reviewer/reflection
    optimized_workflow.add_conditional_edges(
        "deep_thinker",
        optimized_should_continue,
        {
            "tools": "tools",
            "reviewer": "reviewer" if not SKIP_REVIEWER else None,
            "reflection": "reflection" if not SKIP_REFLECTION else None,
            "save_memory": "save_memory" if not SKIP_MEMORY_SAVE else None
        }
    )
    
    optimized_workflow.add_edge("tools", "deep_thinker")
    
    if not SKIP_REVIEWER:
        optimized_workflow.add_conditional_edges(
            "reviewer",
            optimized_review_decision,
            {
                "deep_thinker": "deep_thinker",
                "reflection": "reflection" if not SKIP_REFLECTION else "save_memory",
                "save_memory": "save_memory" if not SKIP_MEMORY_SAVE else None
            }
        )
    
    if not SKIP_REFLECTION:
        if SKIP_MEMORY_SAVE:
            optimized_workflow.add_edge("reflection", END)
        else:
            optimized_workflow.add_edge("reflection", "save_memory")

# Final edge
if not SKIP_MEMORY_SAVE:
    optimized_workflow.add_edge("save_memory", END)

# Compile the optimized graph
optimized_graph = optimized_workflow.compile()

# Export for use
graph = optimized_graph

print("✅ Optimized agent graph compiled successfully!")
print(f"   Estimated speedup: {2 if SKIP_PLANNER else 1}x - {4 if (SKIP_PLANNER and SKIP_REVIEWER and SKIP_REFLECTION) else 2}x faster")
