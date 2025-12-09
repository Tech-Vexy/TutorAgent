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
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
except ImportError:
    ChatHuggingFace = None
    HuggingFaceEndpoint = None

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from deepagents import create_deep_agent
from tools import generate_educational_plot, generate_kcse_quiz, learn_skill, search_skills, web_search, run_python, add_knowledge, add_knowledge_url, search_knowledge, generate_image, recognize_text
from skills_db import EpisodicMemory, KnowledgeBase, SkillManager
from prompts import (
    ROUTER_SYSTEM_PROMPT,
    VISION_SYSTEM_PROMPT,
    SIMPLE_CHAT_SYSTEM_PROMPT,
    DEEP_THINKER_BASE_PROMPT,
    PEDAGOGY_SOCRATIC,
    PEDAGOGY_DIRECT,
    PLANNING_PROMPT,
    SKILL_SAVE_PROMPT,
    REVIEWER_PROMPT,
    LENS_MATH_PROMPT,
    LENS_TRANSLATE_PROMPT,
    LENS_EXPLAIN_PROMPT
)

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "google").lower() # "google" or "huggingface"

# --- Models ---
from model_manager import model_manager

# We access models dynamically via model_manager.get_model()
# to allow for runtime switching.

# Initialize Episodic Memory
episodic_memory = EpisodicMemory()
# Initialize Knowledge Base for RAG
knowledge_base = KnowledgeBase()
# Initialize Skill Manager
skill_manager = SkillManager()
# Initialize Conversation Summarizer for long context management
from conversation_summarizer import conversation_summarizer
# Initialize Learning Profile for adaptive tutoring
from learning_profile import get_learning_profile
# Initialize Language Support for multilingual responses
from language_support import detect_language, get_language_instruction, get_localized_prompt, LanguageCode

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
    review_count: int
    response_language: str  # "sw" for Kiswahili, "en" for English
    lens_mode: Optional[str] # "solve", "translate", "explain", "vision"

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

class ReviewOutput(BaseModel):
    approved: bool = Field(..., description="Whether the response is approved.")
    feedback: str = Field(..., description="Feedback if rejected, or empty if approved.")

def clean_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Cleans messages to prevent 413 Payload Too Large errors:
    1. Limits history to last N messages
    2. Removes image data from multimodal messages
    3. Truncates very long text
    """
    # Limit history to prevent huge payloads
    max_history = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
    if len(messages) > max_history:
        messages = messages[-max_history:]
    
    cleaned = []
    for msg in messages:
        content = msg.content
        
        # Handle multimodal content (list with images/text)
        if isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image_url":
                        # Replace image with text marker
                        new_content.append({"type": "text", "text": "[Image in history]"})
                    elif item.get("type") == "text":
                        # Keep text but truncate if very long
                        text = item.get("text", "")
                        if len(text) > 25000:
                            text = text[:25000] + "...[truncated]"
                        new_content.append({"type": "text", "text": text})
            content = new_content
        
        # Handle string content
        elif isinstance(content, str):
            # Remove generated image data
            if "[IMAGE_GENERATED_BASE64_DATA:" in content:
                start = content.find("[IMAGE_GENERATED_BASE64_DATA:")
                end = content.find("]", start)
                if end != -1:
                    content = content[:start] + "[IMAGE_DATA_HIDDEN]" + content[end+1:]
            # Truncate very long strings
            if len(content) > 25000:
                content = content[:25000] + "...[truncated]"
        
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
    
    Also detects language - uses Kiswahili for Swahili subject, English otherwise.
    Respects user's preferred_language setting if set.
    """
    messages = state["messages"]
    last_message = messages[-1]
    user_profile = state.get("user_profile", {})
    student_id = user_profile.get("student_id", "default_user")
    
    # Extract text for language detection
    query_text = extract_text_content(last_message) if isinstance(last_message, HumanMessage) else ""
    
    # Check user's profile for language preference first
    detected_lang = "en"  # Default
    lang_reason = "Default English"
    
    try:
        learner_profile = get_learning_profile(student_id)
        user_lang_pref = learner_profile.data.get("preferred_language", "auto")
        
        if user_lang_pref == "sw":
            # User explicitly wants Kiswahili
            detected_lang = "sw"
            lang_reason = "User preference set to Kiswahili"
        elif user_lang_pref == "en":
            # User explicitly wants English
            detected_lang = "en"
            lang_reason = "User preference set to English"
        else:
            # Auto-detect based on content
            detected_lang, lang_reason = detect_language(query_text)
    except Exception:
        # Fall back to auto-detection if profile access fails
        detected_lang, lang_reason = detect_language(query_text)
    
    if detected_lang == "sw":
        print(f"\n🌍 Language Detection: KISWAHILI - {lang_reason}\n")
    
    # Check for image presence manually to force vision intent if needed
    has_image = False
    if isinstance(last_message, HumanMessage):
        if isinstance(last_message.content, list):
            for part in last_message.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    has_image = True
                    break
    
    if has_image:
        # Check for explicit lens mode in message metadata (if passed via LangChain message additional_kwargs or content)
        # However, since we receive clean messages, we might need to rely on the server.py to have injected it into state,
        # OR we check if the user text explicitly triggers it.
        
        lens_mode = "vision"
        
        # Simple heuristic if frontend sends text like "Please solve this image"
        lower_query = query_text.lower()
        if "solve" in lower_query or "calculate" in lower_query or "math" in lower_query:
            lens_mode = "solve"
        elif "translate" in lower_query:
            lens_mode = "translate"
        elif "explain" in lower_query or "diagram" in lower_query:
            lens_mode = "explain"
            
        return {
            "intent": "VISION_ANALYSIS", 
            "pedagogy_strategy": "SOCRATIC_GUIDE",
            "response_language": detected_lang,
            "lens_mode": lens_mode
        }

    # Use structured output for text classification
    structured_llm = model_manager.get_model("fast").with_structured_output(RouterOutput)
    system_prompt = ROUTER_SYSTEM_PROMPT
    
    # We only send the last message to the router to save tokens/latency, or a summary
    # For better context, we might send the last few.
    # Clean messages to remove large base64 strings
    # We pass more context to Router now to better understand intent
    cleaned_history = clean_messages(messages[-5:])
    response = await structured_llm.ainvoke([SystemMessage(content=system_prompt)] + cleaned_history)
    
    updates = {
        "intent": response.intent,
        "pedagogy_strategy": response.pedagogy_strategy,
        "response_language": detected_lang
    }
    
    # Smart Model Switching:
    # If the task requires complex reasoning, automatically upgrade to the smart model
    if response.intent == "COMPLEX_REASONING":
        print("\n🚀 Smart Switching: Upgrading to SMART model for complex reasoning task.\n")
        updates["model_preference"] = "smart"
        
    return updates

async def vision_node(state: AgentState):
    """
    Uses Llama 3.2 Vision to analyze images.
    Applies language context if analyzing Swahili subject content.
    """
    messages = state["messages"]
    response_language = state.get("response_language", "en")
    # The vision model needs the image message. 
    # We assume the last message contains the image in the correct format.
    
    # Determine which prompt to use based on lens_mode
    lens_mode = state.get("lens_mode", "vision")
    
    if lens_mode == "solve":
        base_prompt = LENS_MATH_PROMPT
    elif lens_mode == "translate":
        base_prompt = LENS_TRANSLATE_PROMPT.format(target_language=response_language)
    elif lens_mode == "explain":
        base_prompt = LENS_EXPLAIN_PROMPT
    else:
        base_prompt = VISION_SYSTEM_PROMPT

    # Apply language instruction to vision prompt
    system_prompt = get_localized_prompt(base_prompt, response_language)
    
    try:
        response = await model_manager.get_model("vision").ainvoke([SystemMessage(content=system_prompt)] + [messages[-1]])
        content = f"Image Analysis:\n{response.content}"
    except Exception as e:
        # Handle 413 or other API errors
        if "413" in str(e) or "Payload Too Large" in str(e):
             content = "Error: The image provided is too large for the vision model. Please try resizing it or using a smaller file."
        else:
             content = f"Error analyzing image: {str(e)}"
    
    # We return the analysis as an AI message, which will then be passed to the deep thinker or returned
    return {"messages": [AIMessage(content=content)]}

async def simple_chat_node(state: AgentState):
    """
    Handles simple interactions using the fast model.
    Responds in Kiswahili if the subject is Swahili, English otherwise.
    """
    messages = state["messages"]
    user_profile = state.get("user_profile", {})
    student_id = user_profile.get("student_id", "default_user")
    response_language = state.get("response_language", "en")
    
    # Apply language-specific prompt
    system_prompt = get_localized_prompt(SIMPLE_CHAT_SYSTEM_PROMPT, response_language)

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

    # Clean messages (let clean_messages handle the limit)
    cleaned_history = clean_messages(messages)
    response = await model_manager.get_model("fast").ainvoke([SystemMessage(content=full_system_prompt)] + cleaned_history)
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
    Responds in Kiswahili if the subject is Swahili, English otherwise.
    """
    messages = state["messages"]
    strategy = state.get("pedagogy_strategy", "DIRECT_ANSWER")
    model_pref = state.get("model_preference", "smart")
    user_profile = state.get("user_profile", {})
    student_id = user_profile.get("student_id", "default_user")
    plan = state.get("plan")
    response_language = state.get("response_language", "en")
    
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
    # Defaulting to Gemini 1.5 Pro for smart and Flash for everything else
    # Select model based on preference
    # Defaulting to Gemini 1.5 Pro for smart and Flash for everything else
    if model_pref == "fast":
        selected_llm = model_manager.get_model("fast")
    else:
        selected_llm = model_manager.get_model("smart")
    
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
        generate_image,
        recognize_text,
    ]
    
    # --- Integration: Tavily MCP Tools ---
    try:
        from mcp_tool_adapter import get_mcp_tools
        mcp_tools = await get_mcp_tools()
        if mcp_tools:
            # Avoid duplicate tool names if possible, or let LLM decide
            # MCP tools might be named 'tavily_web_search' vs existing 'web_search'
            tools.extend(mcp_tools)
    except Exception as e:
        # Log error but don't crash if MCP fails (e.g. if subprocess fails)
        pass # In production, use a logger
    
    # Dynamic System Prompt based on Strategy
    base_prompt = DEEP_THINKER_BASE_PROMPT.format(memory_context=memory_context, knowledge_context=knowledge_context)
    
    # Add Language Instruction (Kiswahili for Swahili subject, English otherwise)
    language_instruction = get_language_instruction(response_language)
    base_prompt += f"\n{language_instruction}"
    
    # Add Adaptive Learning Context
    try:
        learner_profile = get_learning_profile(student_id)
        learner_context = learner_profile.get_learning_context()
        if learner_context:
            base_prompt += f"\n\nLEARNER PROFILE:\n{learner_context}\n(Adapt your explanation based on this learner's profile.)"
    except Exception:
        pass
    
    if plan:
        base_prompt += f"\n\nCURRENT PLAN:\n{plan}\n(Follow this plan to solve the problem.)"
    
    if strategy == "SOCRATIC_GUIDE":
        pedagogy_instruction = PEDAGOGY_SOCRATIC
    else:
        pedagogy_instruction = PEDAGOGY_DIRECT
        
    full_system_prompt = f"{base_prompt}\n\n{pedagogy_instruction}"
    
    # Create Deep Agent
    # We use create_react_agent for optimized tool calling (parallel execution support)
    deep_agent = create_react_agent(
        model=selected_llm,
        tools=tools,
        state_modifier=full_system_prompt
    )
    
    # Clean messages to remove large base64 strings from history to save tokens
    cleaned_messages = clean_messages(messages)
    
    # Apply conversation summarization for very long sessions
    # This compacts old messages into a summary while keeping recent ones intact
    thread_id = user_profile.get("student_id", "default")
    if conversation_summarizer.should_summarize(cleaned_messages):
        try:
            summarizer_llm = model_manager.get_model("fast")
            cleaned_messages, summary = await conversation_summarizer.summarize_and_compact(
                cleaned_messages, thread_id, summarizer_llm
            )
            print(f"\n📝 Conversation summarized. Summary length: {len(summary)} chars\n")
        except Exception as e:
            print(f"Summarization skipped: {e}")
    
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
        generate_image,
        recognize_text,
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
    # Gemini requires at least one non-system message, so include the user's query
    response = await model_manager.get_model("fast").ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query_text)
    ])
    
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
    structured_llm = model_manager.get_model("fast").with_structured_output(SkillSaveOutput)
    
    # Clean messages
    cleaned_history = clean_messages(messages)
    
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

async def reviewer_node(state: AgentState):
    """
    Reviewer Node:
    Critiques the Deep Thinker's response.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If last message is not AI, something is wrong, just approve to avoid getting stuck
    if not isinstance(last_message, AIMessage):
        return {"review_count": 0} # No op
        
    structured_llm = model_manager.get_model("fast").with_structured_output(ReviewOutput)
    
    # We send the last few messages for context, plus the candidate response
    cleaned_history = clean_messages(messages[-10:])
    
    try:
        review = await structured_llm.ainvoke([SystemMessage(content=REVIEWER_PROMPT)] + cleaned_history)
        
        current_count = state.get("review_count", 0)
        
        if review.approved or current_count >= 2:
            # Approved or max retries hit
            return {"review_count": 0} # Reset? Or just pass.
        else:
            # Rejected
            # We add a HumanMessage with feedback to guide the agent
            feedback_msg = HumanMessage(content=f"REVIEWER FEEDBACK (Internal): {review.feedback}\nPlease rewrite your response to address this.")
            return {
                "messages": [feedback_msg],
                "review_count": current_count + 1
            }
    except Exception:
        # If review fails, pass through
        return {"review_count": 0}

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
workflow.add_node("reviewer", reviewer_node)

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
        "reflection": "reviewer"
    }
)

def review_decision(state: AgentState):
    # If the last message is Human (feedback), loop back to deep_thinker
    # If it's AI (original response unchanged) or review_count reset, proceed
    messages = state["messages"]
    if messages and isinstance(messages[-1], HumanMessage) and "REVIEWER FEEDBACK" in str(messages[-1].content):
        return "deep_thinker"
    return "reflection"

workflow.add_conditional_edges(
    "reviewer",
    review_decision,
    {
        "deep_thinker": "deep_thinker",
        "reflection": "reflection"
    }
)

workflow.add_edge("tools", "deep_thinker")
workflow.add_edge("reflection", "save_memory")
workflow.add_edge("save_memory", END)

# Compile the graph (checkpointer will be passed at runtime)
graph = workflow.compile()
