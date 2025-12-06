import os
from dotenv import load_dotenv

load_dotenv()

# Enable LangSmith tracing if API key is provided (allow disabling via DISABLE_TRACING=1)
if os.getenv("LANGCHAIN_API_KEY") and os.getenv("DISABLE_TRACING", "0") != "1":
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "TopScore-AI"))

from typing import TypedDict, List, Literal, Optional, Dict, Any, Annotated
import operator
import time
import random
from groq import RateLimitError, APIError
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from deepagents import create_deep_agent
from tools import generate_educational_plot, generate_kcse_quiz, learn_skill, search_skills, web_search, run_python, add_knowledge, add_knowledge_url, search_knowledge
from skills_db import EpisodicMemory, KnowledgeBase, SkillManager
from prompts import (
    ROUTER_SYSTEM_PROMPT,
    VISION_SYSTEM_PROMPT,
    SIMPLE_CHAT_SYSTEM_PROMPT,
    DEEP_THINKER_BASE_PROMPT,
    PEDAGOGY_SOCRATIC,
    PEDAGOGY_DIRECT,
    PLANNING_PROMPT,
    SKILL_SAVE_PROMPT
)

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Allow overriding the smart model via environment to avoid decommissioned defaults
SMART_MODEL = os.getenv("SMART_MODEL", "llama-3.3-70b-versatile")
# Allow overriding the fast model as well
FAST_MODEL = os.getenv("FAST_MODEL", "llama-3.1-8b-instant")
# Control fast model token cap for quicker responses
FAST_TOKENS = int(os.getenv("FAST_TOKENS", "512"))

# --- Models ---
# Fast model for routing and simple chats
fast_llm = ChatGroq(
    model=FAST_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
    max_retries=2,
    max_tokens=FAST_TOKENS
)

# Smart model for deep reasoning and tools (use a different model to avoid 8B rate limits)
smart_llm = ChatGroq(
    model=SMART_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.5,
    max_retries=3
)

# Mixtral model (sparse MoE)
mixtral_llm = ChatGroq(
    model="mixtral-8x7b-32768",
    api_key=GROQ_API_KEY,
    temperature=0.5
)

# Gemma model
gemma_llm = ChatGroq(
    model="gemma-2-9b-it",
    api_key=GROQ_API_KEY,
    temperature=0.5
)

# Llama 3.2 1B Instruct
llama32_1b_llm = ChatGroq(
    model="llama-3.2-1b-instruct",
    api_key=GROQ_API_KEY,
    temperature=0.5
)

# Llama 3.2 3B Instruct
llama32_3b_llm = ChatGroq(
    model="llama-3.2-3b-instruct",
    api_key=GROQ_API_KEY,
    temperature=0.5
)

# Llama 3.1 70B versatile
llama3_70b_llm = ChatGroq(
    model=SMART_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.3,
    max_retries=3
)

# Llama 3.1 8B instant
llama3_8b_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.5
)

# Vision model (Llama 3.2 11B Vision Instruct)
vision_llm = ChatGroq(
    model="llama-3.2-11b-vision-instruct",
    api_key=GROQ_API_KEY,
    temperature=0.2
)

# Initialize Episodic Memory
episodic_memory = EpisodicMemory()
# Initialize Knowledge Base for RAG
knowledge_base = KnowledgeBase()
# Initialize Skill Manager
skill_manager = SkillManager()

# --- State Definition ---
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "4"))


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_profile: Dict[str, Any]
    pedagogy_strategy: str  # DIRECT_ANSWER or SOCRATIC_GUIDE
    intent: str # SIMPLE_CHAT, COMPLEX_REASONING, VISION_ANALYSIS
    model_preference: Optional[str] # "fast", "smart", "vision"
    tool_invocations: Annotated[int, operator.add]
    plan: Optional[str]

# --- Router Schema ---
class RouterOutput(BaseModel):
    intent: Literal["SIMPLE_CHAT", "COMPLEX_REASONING"] = Field(
        ..., description="The type of interaction required. Choose COMPLEX_REASONING for any math, science, drawing, or complex tasks."
    )
    pedagogy_strategy: Literal["DIRECT_ANSWER", "SOCRATIC_GUIDE"] = Field(
        ..., description="The teaching strategy to apply."
    )

class SkillSaveOutput(BaseModel):
    save: bool = Field(..., description="Whether to save a new skill.")
    name: Optional[str] = Field(None, description="Short descriptive name of the skill.")
    description: Optional[str] = Field(None, description="Detailed description of when and how to apply this skill.")
    code: Optional[str] = Field(None, description="Optional Python code snippet demonstrating the skill.")

def clean_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Cleans messages by truncating large base64 image strings to prevent token limit errors.
    Also normalizes any unexpected content types gracefully.
    """
    cleaned = []
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            if "[IMAGE_GENERATED_BASE64_DATA:" in content:
                # Truncate the base64 part
                start = content.find("[IMAGE_GENERATED_BASE64_DATA:")
                end = content.find("]", start)
                if end != -1:
                    # Keep the marker but remove the data
                    content = content[:start] + "[IMAGE_DATA_HIDDEN]" + content[end+1:]
        
        # Reconstruct message with cleaned content
        if isinstance(msg, HumanMessage):
            cleaned.append(HumanMessage(content=content))
        elif isinstance(msg, AIMessage):
            cleaned.append(AIMessage(content=content))
        elif isinstance(msg, SystemMessage):
            cleaned.append(SystemMessage(content=content))
        elif isinstance(msg, ToolMessage):
            cleaned.append(ToolMessage(content=content, tool_call_id=getattr(msg, "tool_call_id", None)))
        else:
            cleaned.append(msg)
    return cleaned


def extract_text_content(message: BaseMessage) -> str:
    """Extract plain text from a HumanMessage that may be multimodal (list of parts)."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content) if content is not None else ""

# --- Nodes ---

async def router_node(state: AgentState):
    """
    Pedagogy Router:
    Classifies the user's intent and determines the best teaching strategy.
    - VISION_ANALYSIS: If the user provides an image.
    - COMPLEX_REASONING: For math, science, or deep questions requiring tools.
    - SIMPLE_CHAT: For greetings, quick facts, or clarifications.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check for image presence manually to force vision intent if needed
    has_image = False
    if isinstance(last_message, HumanMessage):
        if isinstance(last_message.content, list):
            for part in last_message.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    has_image = True
                    break
    
    if has_image:
        return {"intent": "VISION_ANALYSIS", "pedagogy_strategy": "SOCRATIC_GUIDE"}

    # Use structured output for text classification
    structured_llm = fast_llm.with_structured_output(RouterOutput)
    system_prompt = ROUTER_SYSTEM_PROMPT
    
    # We only send the last message to the router to save tokens/latency, or a summary
    # For better context, we might send the last few.
    # Clean messages to remove large base64 strings
    cleaned_history = clean_messages(messages[-3:])
    response = await structured_llm.ainvoke([SystemMessage(content=system_prompt)] + cleaned_history)
    
    return {
        "intent": response.intent,
        "pedagogy_strategy": response.pedagogy_strategy
    }

async def vision_node(state: AgentState):
    """
    Uses Llama 3.2 Vision to analyze images.
    """
    messages = state["messages"]
    # The vision model needs the image message. 
    # We assume the last message contains the image in the correct format.
    
    system_prompt = VISION_SYSTEM_PROMPT
    
    response = await vision_llm.ainvoke([SystemMessage(content=system_prompt)] + [messages[-1]])
    
    # We return the analysis as an AI message, which will then be passed to the deep thinker or returned
    return {"messages": [AIMessage(content=f"Image Analysis:\n{response.content}")]}

async def simple_chat_node(state: AgentState):
    """
    Handles simple interactions using the fast model.
    """
    messages = state["messages"]
    user_profile = state.get("user_profile", {})
    student_id = user_profile.get("student_id", "default_user")
    system_prompt = SIMPLE_CHAT_SYSTEM_PROMPT

    # Retrieve a small amount of relevant memory to personalize the reply
    memory_context = ""
    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        query_text = extract_text_content(last_message)
        memories = []
        if query_text.strip():
            memories = episodic_memory.retrieve_memory(student_id, query_text, n_results=2)
        if not memories:
            memories = episodic_memory.retrieve_recent_memory(student_id, n_results=2)
        if memories:
            formatted = "\n".join([f"- {m}" for m in memories])
            memory_context = f"\n\nRELEVANT PAST INTERACTIONS:\n{formatted}\n(Use this context to personalize your response, but do not explicitly mention 'memory' unless relevant.)"

    full_system_prompt = system_prompt + memory_context

    # Clean messages and limit history to last 5 to prevent token overflow
    cleaned_history = clean_messages(messages[-5:])
    response = await fast_llm.ainvoke([SystemMessage(content=full_system_prompt)] + cleaned_history)
    return {"messages": [response]}


def remove_code_blocks(text: str) -> str:
    """
    Removes markdown code blocks (e.g. ```python ... ```) from the text 
    to prevent showing code to students.
    """
    import re
    # Remove python blocks or generic blocks that might contain code
    # We replace them with a small placeholder or nothing to keep flow smooth
    # Removing strictly Python-labeled blocks and generic ones that might look like code
    pattern = r"```(?:python|py)?.*?\n[\s\S]*?```"
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()

async def deep_thinker_node(state: AgentState):
    """
    Handles complex reasoning using the smart model and tools.
    """
    messages = state["messages"]
    strategy = state.get("pedagogy_strategy", "DIRECT_ANSWER")
    model_pref = state.get("model_preference", "smart")
    user_profile = state.get("user_profile", {})
    student_id = user_profile.get("student_id", "default_user")
    plan = state.get("plan")
    
    # Retrieve Episodic Memory
    last_message = messages[-1]
    memory_context = ""
    knowledge_context = ""
    query_text = ""
    if isinstance(last_message, HumanMessage):
        query_text = extract_text_content(last_message)
        memories = []
        if isinstance(query_text, str) and query_text.strip():
            memories = episodic_memory.retrieve_memory(student_id, query_text, n_results=3)
        # Fallback to recent memories if the query yields nothing (or query empty)
        if not memories:
            memories = episodic_memory.retrieve_recent_memory(student_id, n_results=2)
        if memories:
            formatted_memories = "\n".join([f"- {m}" for m in memories])
            memory_context = f"\n\nRELEVANT PAST INTERACTIONS:\n{formatted_memories}\n(Use this context to personalize your response, but do not explicitly mention 'memory' unless relevant.)"
        # Retrieve Knowledge (RAG)
        try:
            if isinstance(query_text, str) and query_text.strip():
                items = knowledge_base.retrieve_knowledge(query_text, n_results=3)
                if items:
                    lines = []
                    for it in items:
                        meta = it.get("metadata", {}) if isinstance(it, dict) else {}
                        src = meta.get("source", "unknown")
                        snippet = it.get("text", "").strip() if isinstance(it, dict) else ""
                        if len(snippet) > 350:
                            snippet = snippet[:350] + "..."
                        lines.append(f"- [{src}] {snippet}")
                    knowledge_context = "\n\nRETRIEVED KNOWLEDGE (quotes/snippets):\n" + "\n".join(lines) + "\n(Use these snippets for grounding and cite briefly in parentheses.)"
        except Exception:
            knowledge_context = ""

    # Select model based on preference
    if model_pref == "fast":
        selected_llm = fast_llm
    elif model_pref in ("mixtral", "mixtral-8x7b-32768", "openai/gpt-oss"):
        selected_llm = mixtral_llm
    elif model_pref == "gemma":
        selected_llm = gemma_llm
    elif model_pref == "llama32_1b":
        selected_llm = llama32_1b_llm
    elif model_pref == "llama32_3b":
        selected_llm = llama32_3b_llm
    elif model_pref == "llama3_70b":
        selected_llm = llama3_70b_llm
    elif model_pref == "llama3_8b":
        selected_llm = llama3_8b_llm
    else:
        selected_llm = smart_llm
    
    # --- Deep Agent Integration ---
    
    # Bind tools (Added learn_skill and search_skills)
    tools = [
        generate_educational_plot,
        generate_kcse_quiz,
        learn_skill,
        search_skills,
        web_search,
        run_python,
        add_knowledge,
        add_knowledge_url,
        search_knowledge,
    ]
    
    # Dynamic System Prompt based on Strategy
    base_prompt = DEEP_THINKER_BASE_PROMPT.format(memory_context=memory_context, knowledge_context=knowledge_context)
    
    if plan:
        base_prompt += f"\n\nCURRENT PLAN:\n{plan}\n(Follow this plan to solve the problem.)"
    
    if strategy == "SOCRATIC_GUIDE":
        pedagogy_instruction = PEDAGOGY_SOCRATIC
    else:
        pedagogy_instruction = PEDAGOGY_DIRECT
        
    full_system_prompt = f"{base_prompt}\n\n{pedagogy_instruction}"
    
    # Create Deep Agent
    # We create a fresh one each time to inject the dynamic prompt and context
    # In a more persistent setup, we might cache it, but the prompt changes per turn.
    deep_agent = create_deep_agent(
        model=selected_llm,
        tools=tools,
        system_prompt=full_system_prompt
    )
    
    # Clean messages to remove large base64 strings from history to save tokens
    cleaned_messages = clean_messages(messages[-8:])
    
    try:
        # Invoke Deep Agent
        # We wrap the messages in a dict as expected by LangGraph agents
        result = await deep_agent.ainvoke({"messages": cleaned_messages})
        
        # Extract the last message from the result
        # The deep agent might return a list of messages including tool calls/results
        # We want the final answer or the tool call if it decided to stop there (though deep agent usually executes tools)
        result_messages = result.get("messages", [])
        if result_messages:
            last_response = result_messages[-1]
            if isinstance(last_response, AIMessage) and isinstance(last_response.content, str):
                # Filter out code blocks
                last_response.content = remove_code_blocks(last_response.content)
            
            return {"messages": [last_response]}
        else:
            return {"messages": [AIMessage(content="I'm sorry, I couldn't generate a response.")]}

    except (RateLimitError, APIError):
        # Fallback logic could be implemented here similar to before, 
        # but deep_agent encapsulates the execution loop.
        # For now, we return a polite error.
        return {"messages": [AIMessage(content="I'm temporarily rate-limited. Please try again.")]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"An error occurred in the deep agent: {e}")]}

def tool_node(state: AgentState):
    """
    Executes tools requested by the deep thinker.
    """
    from langgraph.prebuilt import ToolNode
    tools = [
        generate_educational_plot,
        generate_kcse_quiz,
        learn_skill,
        search_skills,
        web_search,
        run_python,
        add_knowledge,
        add_knowledge_url,
        search_knowledge,
    ]
    tool_executor = ToolNode(tools)
    return tool_executor.invoke(state)

def save_memory_node(state: AgentState):
    """
    Saves the interaction to episodic memory.
    - Extracts plain text from multimodal messages.
    - Truncates to a reasonable length to keep embeddings efficient.
    """
    messages = state["messages"]
    user_profile = state.get("user_profile", {})
    student_id = user_profile.get("student_id", "default_user")
    
    # Find the last human message and the last AI message
    # We look for the most recent pair
    last_human = None
    last_ai = None
    
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not last_ai:
            last_ai = m
        if isinstance(m, HumanMessage) and not last_human:
            last_human = m
        if last_human and last_ai:
            break
            
    if last_human and last_ai:
        human_text = extract_text_content(last_human)
        ai_text = last_ai.content if isinstance(last_ai.content, str) else str(last_ai.content)
        memory_content = f"User: {human_text}\nAI: {ai_text}"
        # Truncate to 1500 chars to keep vectors small; details still in checkpoints
        if len(memory_content) > 1500:
            memory_content = memory_content[:1500] + "..."
        episodic_memory.save_memory(student_id, memory_content, metadata={"source": "chat"})
        
    return state # Pass through

async def planner_node(state: AgentState):
    """
    Planner Node:
    Breaks down complex queries into a step-by-step plan.
    Retrieves relevant skills to inform the plan.
    """
    messages = state["messages"]
    last_message = messages[-1]
    query_text = extract_text_content(last_message)
    
    # Retrieve relevant skills
    skills = skill_manager.retrieve_skill(query_text, n_results=3)
    skills_text = ""
    if skills:
        skills_text = "\n".join([f"- {s.get('name')}: {s.get('description')}" for s in skills])
    else:
        skills_text = "No specific skills found."
        
    system_prompt = PLANNING_PROMPT.format(skills=skills_text, query=query_text)
    
    # Use fast model for planning to save time/cost
    response = await fast_llm.ainvoke([SystemMessage(content=system_prompt)])
    
    return {"plan": response.content}

async def reflection_node(state: AgentState):
    """
    Reflection Node:
    Analyzes the interaction to see if a new skill should be saved.
    """
    # Only reflect if tools were used or it was a complex task
    # We can check tool_invocations or just run it for COMPLEX_REASONING
    
    messages = state["messages"]
    # We need the full context to decide
    # Use structured output
    structured_llm = fast_llm.with_structured_output(SkillSaveOutput)
    
    # Clean messages
    cleaned_history = clean_messages(messages[-5:])
    
    try:
        response = await structured_llm.ainvoke([SystemMessage(content=SKILL_SAVE_PROMPT)] + cleaned_history)
        
        if response.save and response.name and response.description:
            skill_manager.save_skill(
                name=response.name,
                code=response.code or "",
                description=response.description
            )
            # We don't need to notify the user, just save it silently
    except Exception:
        # If reflection fails, just continue
        pass
        
    return state # Pass through

# --- Graph Construction ---

workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("simple_chat", simple_chat_node)
workflow.add_node("vision_analysis", vision_node)
workflow.add_node("deep_thinker", deep_thinker_node)
workflow.add_node("tools", tool_node)
workflow.add_node("save_memory", save_memory_node)
workflow.add_node("planner", planner_node)
workflow.add_node("reflection", reflection_node)

workflow.set_entry_point("router")

def route_decision(state: AgentState):
    intent = state.get("intent")
    if intent == "VISION_ANALYSIS":
        return "vision_analysis"
    elif intent == "COMPLEX_REASONING":
        return "planner"
    else:
        return "simple_chat"

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "vision_analysis": "vision_analysis",
        "planner": "planner",
        "simple_chat": "simple_chat"
    }
)

# Vision analysis feeds into deep thinker to solve the problem found
workflow.add_edge("vision_analysis", "deep_thinker")

# Planner feeds into deep thinker
workflow.add_edge("planner", "deep_thinker")

# Simple chat saves memory then ends
workflow.add_edge("simple_chat", "save_memory")

# Deep thinker conditional edge for tools
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "reflection"

workflow.add_conditional_edges(
    "deep_thinker",
    should_continue,
    {
        "tools": "tools",
        "reflection": "reflection"
    }
)

workflow.add_edge("tools", "deep_thinker")
workflow.add_edge("reflection", "save_memory")
workflow.add_edge("save_memory", END)

# Compile the graph (checkpointer will be passed at runtime)
graph = workflow.compile()
