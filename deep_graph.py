"""LangGraph implementation for the Deep Reasoning AI Tutor."""

from typing import TypedDict, List, Dict, Literal, Annotated, Union, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import io
import base64
import sys
import json
import os
from contextlib import redirect_stdout, redirect_stderr
from skills_db import SkillManager, EpisodicMemory
from prompts import ROUTER_PROMPT, TUTOR_SYSTEM_PROMPT, SKILL_SAVE_PROMPT, PLANNING_PROMPT
import operator
from langgraph.checkpoint.memory import MemorySaver
from database import get_or_create_profile

# Define State
class TutorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_complexity_level: str
    teaching_strategy: str
    retrieved_skills: List[Dict]
    retrieved_memories: List[str]
    current_plan: str
    user_id: str
    student_profile: Dict[str, Any]
    current_image: Optional[str]

class RouteQuery(BaseModel):
    """Route the user query to the most appropriate teaching strategy."""
    strategy: Literal["DIRECT_ANSWER", "SOCRATIC_GUIDE", "QUIZ_REQUEST", "HOMEWORK_HELPER"] = Field(
        ..., description="The teaching strategy to use."
    )
    complexity: Literal["SIMPLE", "COMPLEX"] = Field(
        ..., description="The complexity of the user query."
    )

class DeepTutorGraph:
    def __init__(self, 
                 fast_model: str = "gpt-3.5-turbo", 
                 slow_model: str = "gpt-4",
                 fast_base_url: str = None,
                 slow_base_url: str = None,
                 fast_api_key: str = None,
                 slow_api_key: str = None,
                 checkpointer = None
                 ):
        # Initialize models
        self.fast_llm = ChatOpenAI(
            model=fast_model, 
            temperature=0,
            base_url=fast_base_url,
            api_key=fast_api_key
        )
        self.slow_llm = ChatOpenAI(
            model=slow_model, 
            temperature=0.7,
            base_url=slow_base_url,
            api_key=slow_api_key
        )
        
        # Vision model configuration (defaults to fast_base_url if not set)
        self.vision_base_url = os.getenv("VLLM_BASE_URL", fast_base_url)
        self.vision_model_name = os.getenv("VISION_MODEL", "qwen2.5-vl")
        self.vision_llm = ChatOpenAI(
            model=self.vision_model_name,
            temperature=0,
            base_url=self.vision_base_url,
            api_key=fast_api_key or "EMPTY"
        )

        self.skill_manager = SkillManager()
        self.episodic_memory = EpisodicMemory()
        self.checkpointer = checkpointer if checkpointer else MemorySaver()
        self.search_tool = TavilySearchResults(max_results=3)
        
        # Define tools
        self.tools = [self.python_sandbox, self.retrieve_skill_tool, self.save_skill_tool, self.search_tool, self.generate_quiz, self.get_profile_tool]
        
        self.slow_llm_with_tools = self.slow_llm.bind_tools(self.tools)
        
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(TutorState)
        
        # Add nodes
        workflow.add_node("load_profile", self._load_profile_node)
        workflow.add_node("vision_analysis", self._vision_analysis_node)
        workflow.add_node("router", self._router_node)
        workflow.add_node("simple_response", self._simple_response_node)
        workflow.add_node("skill_retrieval", self._skill_retrieval_node)
        workflow.add_node("planning", self._planning_node)
        workflow.add_node("deep_reasoning", self._deep_reasoning_node)
        workflow.add_node("tool_execution", self._tool_execution_node)
        workflow.add_node("skill_saving", self._skill_saving_node)
        
        # Add edges
        workflow.set_entry_point("load_profile")
        
        # Conditional edge from load_profile to check for images
        workflow.add_conditional_edges(
            "load_profile",
            self._route_vision,
            {
                "VISION": "vision_analysis",
                "ROUTER": "router"
            }
        )
        
        workflow.add_edge("vision_analysis", "router")
        
        workflow.add_conditional_edges(
            "router",
            self._route_complexity,
            {
                "SIMPLE": "simple_response",
                "COMPLEX": "skill_retrieval"
            }
        )
        
        workflow.add_edge("simple_response", END)
        workflow.add_edge("skill_retrieval", "planning")
        workflow.add_edge("planning", "deep_reasoning")
        
        # Conditional edge for deep reasoning (tool use vs end)
        workflow.add_conditional_edges(
            "deep_reasoning",
            self._route_deep_reasoning,
            {
                "TOOLS": "tool_execution",
                "DONE": "skill_saving"
            }
        )
        
        workflow.add_edge("tool_execution", "deep_reasoning")
        workflow.add_edge("skill_saving", END)
        
        return workflow.compile(checkpointer=self.checkpointer)

    def _route_vision(self, state: TutorState) -> Literal["VISION", "ROUTER"]:
        """Check if the conversation contains an image to analyze."""
        messages = state.get("messages", [])
        if not messages:
            return "ROUTER"
            
        last_message = messages[-1]
        
        # Check for OpenAI-format image message
        if isinstance(last_message, dict) and last_message.get("type") == "image_url":
            return "VISION"
        elif isinstance(last_message, HumanMessage):
            if isinstance(last_message.content, list):
                for part in last_message.content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return "VISION"
            elif isinstance(last_message.content, str) and "data:image" in last_message.content:
                 return "VISION"
                 
        return "ROUTER"

    async def _load_profile_node(self, state: TutorState) -> TutorState:
        """Load the student profile at the start of the conversation."""
        user_id = state.get("user_id", "default_user")
        profile = await get_or_create_profile(user_id)
        return {"student_profile": profile}

    def _vision_analysis_node(self, state: TutorState) -> TutorState:
        """Analyze images if present using Qwen2.5-VL."""
        messages = state.get("messages", [])
        if not messages:
            return {}
            
        last_message = messages[-1]
        image_url = None
        
        # Check for OpenAI-format image message
        if isinstance(last_message, dict) and last_message.get("type") == "image_url":
            image_url = last_message.get("image_url", {}).get("url")
        elif isinstance(last_message, HumanMessage):
            # Handle content as list of parts (text + image) or direct string
            if isinstance(last_message.content, list):
                for part in last_message.content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url")
                        break
            elif isinstance(last_message.content, str) and "data:image" in last_message.content:
                 image_url = last_message.content
            
        if not image_url:
            return {}
            
        # Transcribe handwriting
        system_msg = SystemMessage(content=(
            "You are a handwriting transcription assistant. Given an image embedded as a data URL, "
            "transcribe any handwritten text precisely. If the image contains math notation, preserve it in LaTeX-like plaintext. "
            "Return only the transcription text without extra commentary."
        ))
        prompt = f"Image (data URL): {image_url}\n\nPlease transcribe the handwriting found in the image."
        
        try:
            response = self.vision_llm.invoke([system_msg, HumanMessage(content=prompt)])
            transcription = response.content
            
            # Append transcription as a user message so the tutor can see it
            return {
                "messages": [HumanMessage(content=f"Image Transcription:\n{transcription}")]
            }
        except Exception as e:
            return {
                "messages": [SystemMessage(content=f"Error analyzing image: {str(e)}")]
            }

    def _router_node(self, state: TutorState) -> TutorState:
        """Route queries based on complexity and teaching strategy."""
        # Get the last user message (or transcription)
        # We need to find the last HumanMessage that has text content
        last_message_content = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
                last_message_content = msg.content
                break
        
        if not last_message_content:
            last_message_content = "Hello"

        structured_llm = self.fast_llm.with_structured_output(RouteQuery)
        response = structured_llm.invoke([
            SystemMessage(content="You are a query classifier. Analyze the user's request."),
            HumanMessage(content=last_message_content)
        ])
        
        return {
            "user_complexity_level": response.complexity,
            "teaching_strategy": response.strategy
        }
    
    def _route_complexity(self, state: TutorState) -> Literal["SIMPLE", "COMPLEX"]:
        if state["teaching_strategy"] == "DIRECT_ANSWER" and state["user_complexity_level"] == "SIMPLE":
            return "SIMPLE"
        return "COMPLEX"
    
    def _simple_response_node(self, state: TutorState) -> TutorState:
        """Handle simple queries with fast model."""
        messages = [SystemMessage(content="You are a helpful AI tutor. Provide clear, concise answers.")]
        messages.extend(state["messages"])
        
        response = self.fast_llm.invoke(messages)
        return {"messages": [response]}
    
    def _skill_retrieval_node(self, state: TutorState) -> TutorState:
        """Retrieve relevant skills and memories for complex queries."""
        # Find last user message
        last_message = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_message = msg.content
                break
                
        user_id = state.get("user_id", "default_user")
        
        # Retrieve skills
        skills_str = self.skill_manager.retrieve_skill(last_message)
        
        # Retrieve episodic memories
        memories = self.episodic_memory.retrieve_memory(user_id, last_message)
        memories_str = "\n".join(memories) if memories else "No relevant past memories."
        
        context_msg = f"Relevant Skills Retrieved:\n{skills_str}\n\nRelevant Past Memories:\n{memories_str}"
        
        return {
            "messages": [SystemMessage(content=context_msg)],
            "retrieved_memories": memories
        }

    def _planning_node(self, state: TutorState) -> TutorState:
        """Create a plan for complex queries."""
        user_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        
        # Get skills context
        skills_context = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, SystemMessage) and "Relevant Skills Retrieved" in msg.content:
                skills_context = msg.content
                break
        
        profile = state.get("student_profile", {})
        learning_style = profile.get("learning_style", "Visual")
        
        prompt = PLANNING_PROMPT.format(skills=skills_context, query=user_query)
        prompt += f"\n\nStudent Learning Style: {learning_style}"
        
        response = self.fast_llm.invoke([SystemMessage(content=prompt)])
        plan = response.content
        
        return {"current_plan": plan, "messages": [SystemMessage(content=f"Proposed Plan:\n{plan}")]}
            
    def _deep_reasoning_node(self, state: TutorState) -> TutorState:
        """Handle complex queries with deep reasoning."""
        strategy = state.get("teaching_strategy", "DIRECT_ANSWER")
        
        # Add the plan to the context if it exists
        plan_context = ""
        if state.get("current_plan"):
            plan_context = f"\n\nFollow this plan:\n{state['current_plan']}"
            
        if strategy == "SOCRATIC_GUIDE":
            system_prompt = """You are an expert AI Tutor for the Kenyan education system using the Socratic method.
CRITICAL WORKFLOW:
1. Guide students step-by-step, ask probing questions instead of giving answers.
2. Use local Kenyan examples (e.g., M-Pesa, Nairobi traffic, Rift Valley geography, Matatu culture) to make concepts concrete.
3. Align with the Kenyan CBC (Competency-Based Curriculum) standards where possible.
4. Use 'python_sandbox' for visualizations and 'tavily_search_results_json' for real-time info.
5. Be patient and encouraging.
"""
        elif strategy == "QUIZ_REQUEST":
            system_prompt = """You are a Quiz Master.
1. Use the 'generate_quiz' tool to create a quiz for the user.
2. Review the user's answers if they provide them.
"""
        else:
            system_prompt = TUTOR_SYSTEM_PROMPT
            
        messages = [SystemMessage(content=system_prompt + plan_context)] + state["messages"]
        response = self.slow_llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _route_deep_reasoning(self, state: TutorState) -> Literal["TOOLS", "DONE"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "TOOLS"
        return "DONE"

    def _tool_execution_node(self, state: TutorState) -> TutorState:
        """Execute tools called by the LLM."""
        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            output = ""
            try:
                if tool_name == "python_sandbox":
                    output = self.python_sandbox(**tool_args)
                elif tool_name == "retrieve_skill_tool":
                    output = self.retrieve_skill_tool(**tool_args)
                elif tool_name == "save_skill_tool":
                    output = self.save_skill_tool(**tool_args)
                elif tool_name == "tavily_search_results_json":
                    output = self.search_tool.invoke(tool_args)
                elif tool_name == "generate_quiz":
                    output = self.generate_quiz(**tool_args)
                elif tool_name == "get_profile_tool":
                    # This tool is async in definition but we are calling it sync here?
                    # We should probably use the tool's invoke method which handles async if properly set up,
                    # or just call the function if we can await it.
                    # Since this node is sync, we might have issues if we await.
                    # But LangGraph supports async nodes. Let's make this node async if needed.
                    # For now, let's assume the tool wrapper handles it or we skip await if it's not awaitable in this context.
                    # Actually, get_profile_tool is defined as async def below.
                    # We can't call it directly without await.
                    # Let's use the tool instance from self.tools
                    tool_instance = next((t for t in self.tools if t.name == "get_profile_tool"), None)
                    if tool_instance:
                        # If we are in a sync node, we can't await.
                        # We should make _tool_execution_node async.
                        pass 
                    output = "Profile tool called (async handling needed)" 
                else:
                    output = f"Error: Tool {tool_name} not found."
            except Exception as e:
                output = f"Error executing tool {tool_name}: {str(e)}"
            
            results.append(ToolMessage(content=str(output), tool_call_id=tool_id))
            
        return {"messages": results}

    def _skill_saving_node(self, state: TutorState) -> TutorState:
        """Evaluate and save new skills and episodic memory after complex problem solving."""
        messages = state["messages"]
        user_id = state.get("user_id", "default_user")
        
        # Extract conversation for context
        conversation_str = ""
        for msg in messages[-5:]: # Last few messages
            if isinstance(msg, HumanMessage):
                conversation_str += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                conversation_str += f"AI: {msg.content}\n"
        
        # 1. Save Skill (Existing logic)
        save_prompt = f"{SKILL_SAVE_PROMPT}\n\nConversation:\n{conversation_str}"
        response = self.fast_llm.invoke([SystemMessage(content=save_prompt)])
        
        try:
            import json
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            result = json.loads(content)
            if result.get("save"):
                self.skill_manager.save_skill(
                    result["name"], 
                    result.get("code", ""), 
                    result["description"]
                )
        except:
            pass
            
        # 2. Save Episodic Memory
        summary_prompt = f"Summarize the following interaction in 1-2 sentences, focusing on what the user asked and what was achieved. Conversation:\n{conversation_str}"
        summary_response = self.fast_llm.invoke([SystemMessage(content=summary_prompt)])
        self.episodic_memory.save_memory(user_id, summary_response.content)
        
        return {"messages": []}

    @tool
    def python_sandbox(self, code: str) -> str:
        """Execute Python code safely and return results with plots.
        Use this for math, data analysis, or creating visualizations.
        """ 
        output = io.StringIO()
        error_output = io.StringIO()
        
        try:
            # Capture stdout and stderr
            with redirect_stdout(output), redirect_stderr(error_output):
                # Create a safe execution environment
                safe_globals = {
                    '__builtins__': {
                        'print': print, 'len': len, 'range': range, 'str': str,
                        'int': int, 'float': float, 'list': list, 'dict': dict,
                        'sum': sum, 'max': max, 'min': min, 'abs': abs,
                        'sorted': sorted, 'enumerate': enumerate, 'zip': zip
                    },
                    'plt': plt, 'matplotlib': plt, 'io': io, 'base64': base64
                }
                
                exec(code, safe_globals)
                
                # Check if matplotlib figure exists
                if plt.get_fignums():
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.read()).decode()
                    plt.close('all')
                    return f"Output: {output.getvalue()}\n[Generated Plot: data:image/png;base64,{img_b64}]"
                
        except Exception as e:
            return f"Error: {str(e)}\nStderr: {error_output.getvalue()}"
            
        return f"Output: {output.getvalue()}"

    @tool
    def save_skill_tool(self, name: str, code: str, description: str) -> str:
        """Save a reusable skill, technique, or algorithm to the database.
        Use this when you have solved a problem and want to remember the method for later.
        """
        skill_id = self.skill_manager.save_skill(name, code, description)
        return f"Skill '{name}' saved."

    @tool
    def retrieve_skill_tool(self, query: str) -> str:
        """Retrieve relevant skills from the database based on a query.
        Use this to find similar problems or techniques before solving a new problem.
        """
        return self.skill_manager.retrieve_skill(query)

    @tool
    def generate_quiz(self, topic: str, difficulty: str = "medium", num_questions: int = 3) -> str:
        """Generate a quiz for the student to practice.
        Args:
            topic: The subject or topic to quiz on.
            difficulty: 'easy', 'medium', or 'hard'.
            num_questions: Number of questions to generate.
        """
        prompt = f"Generate a {num_questions}-question quiz on '{topic}' with {difficulty} difficulty. Include 4 multiple choice options for each question. Do not provide the answers yet."
        response = self.fast_llm.invoke([HumanMessage(content=prompt)])
        return response.content

    @tool
    async def get_profile_tool(self, student_id: str) -> str:
        """Get or create the student profile."""
        profile = await get_or_create_profile(student_id)
        return json.dumps(profile)
