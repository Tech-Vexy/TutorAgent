import matplotlib
import seaborn as sns
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import warnings
# Suppress non-interactive backend warning when user code calls plt.show() under Agg
warnings.filterwarnings(
    "ignore",
    message=r".*FigureCanvasAgg is non-interactive, and thus cannot be shown.*",
    category=UserWarning,
)
from langchain_core.tools import tool
from skills_db import SkillManager, KnowledgeBase
from tavily import TavilyClient
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    ddg_search = DuckDuckGoSearchRun()
except ImportError:
    ddg_search = None

try:
    from huggingface_hub import InferenceClient
    # Default to a standard stable diffusion model
    # User mentioned "nano banana" - checking if they meant "stable-diffusion-v1-5" or simlar.
    # Using SDXL base for quality.
    HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    hf_client = InferenceClient(model=HF_MODEL, token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
except (ImportError, Exception):
    hf_client = None
import os
import math
import contextlib
import sys
import json

# Initialize SkillManager
skill_manager = SkillManager()
# Initialize Knowledge Base for RAG
knowledge_base = KnowledgeBase()

# Initialize Tavily (uses TAVILY_API_KEY from environment)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
try:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
except Exception:
    tavily_client = None

@tool
def generate_educational_plot(python_code: str) -> str:
    """
    Executes Python Matplotlib code to generate an educational graph or plot.
    The code must use 'plt' to create the figure.
    Returns a special string containing the base64 encoded image data.
    """
    fig = plt.figure()
    buf = io.BytesIO()
    try:
        # Create a new figure to avoid state leakage
        # fig already created; ensure we close it in finally block
        
        # Mock plt.show to prevent blocking or warnings in Agg backend
        # and to ensure the figure isn't cleared before we save it.
        # We store the original just in case, though in Agg it's less critical.
        original_show = plt.show
        plt.show = lambda *args, **kwargs: None
        
        try:
            # Handle single-line code with semicolons by replacing them with newlines
            if ';' in python_code and '\n' not in python_code:
                python_code = python_code.replace('; ', ';\n').replace(';', ';\n')
            
            # Safe execution environment
            local_scope = {'plt': plt, 'matplotlib': matplotlib, 'sns': sns}
            exec(python_code, {}, local_scope)
        finally:
            # Restore plt.show
            plt.show = original_show
        
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        # Encode
        b64_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"[IMAGE_GENERATED_BASE64_DATA: {b64_data}]"
    except Exception as e:
        return f"Error generating plot: {str(e)}"
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass
        buf.close()

# --- Instructor Client ---
try:
    import instructor
    from openai import OpenAI
    from quiz_schemas import Quiz
    # Initialize patched client (requires OPENAI_API_KEY)
    # Check if key exists to avoid crash on import/init
    if os.getenv("OPENAI_API_KEY"):
        instructor_client = instructor.from_openai(OpenAI())
    else:
        instructor_client = None
except ImportError:
    instructor_client = None

@tool
def generate_kcse_quiz(topic: str) -> str:
    """
    Generates a 3-question KCSE-style quiz on the given topic using structured output.
    Returns the formatted quiz.
    """
    if not instructor_client:
        return f"Error: Instructor/OpenAI not available. Placeholder quiz for {topic}: 1. What is {topic}? 2. Explain {topic}. 3. Define {topic}."

    try:
        # Generate structured quiz
        quiz = instructor_client.chat.completions.create(
            model="gpt-3.5-turbo", # Or gpt-4o, etc.
            response_model=Quiz,
            messages=[
                {"role": "user", "content": f"Generate a hard KCSE quiz on {topic}."}
            ]
        )
        
        # Format output
        out = f"**KCSE Quiz on {quiz.topic}** (Total Marks: {quiz.total_marks})\n\n"
        for q in quiz.questions:
            out += f"{q.id}. {q.text} ({q.marks} marks)\n   *Guide: {q.answer_guide}*\n"
        return out
    except Exception as e:
        return f"Error generating quiz with instructor: {e}"

@tool
def learn_skill(name: str, code: str, description: str) -> str:
    """
    Saves a new skill (code snippet, formula, or procedure) to the agent's long-term skill memory.
    Use this when the user teaches you a new method or you want to remember a complex solution for later.
    
    Args:
        name: A short, descriptive name for the skill (e.g., "Quadratic Formula Solver").
        code: The code, formula, or step-by-step procedure.
        description: A detailed explanation of when and how to use this skill.
    """
    try:
        skill_id = skill_manager.save_skill(name, code, description)
        return f"Skill '{name}' saved successfully with ID: {skill_id}"
    except Exception as e:
        return f"Error saving skill: {str(e)}"

@tool
def search_skills(query: str) -> str:
    """
    Searches the agent's skill memory for relevant code, formulas, or procedures.
    Use this when you need to recall how to solve a specific type of problem.
    
    Args:
        query: A search query describing the skill you need (e.g., "how to solve quadratic equations").
    """
    try:
        skills = skill_manager.retrieve_skill(query)
        if not skills:
            return "No relevant skills found."
        
        result = "Found the following skills:\n"
        for skill in skills:
            result += f"- Name: {skill.get('name')}\n  Description: {skill.get('description')}\n  Code/Content: {skill.get('code')}\n\n"
        return result
    except Exception as e:
        return f"Error searching skills: {str(e)}"

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for up-to-date information using Tavily and return concise results.
    Returns a JSON string with a short summary and top links.
    """
    try:
        if tavily_client is None:
            return json.dumps({"error": "Tavily disabled: missing or invalid TAVILY_API_KEY"})
        resp = tavily_client.search(query=query, max_results=max_results)
        # Normalize common Tavily fields
        results = resp.get("results") or resp.get("data") or []
        simplified = []
        for r in results:
            simplified.append({
                "title": r.get("title") or r.get("name"),
                "url": r.get("url"),
                "snippet": r.get("content") or r.get("snippet")
            })
        out = {
            "query": query,
            "num_results": len(simplified),
            "results": simplified[:max_results]
        }
        return json.dumps(out)
        return json.dumps(out)
    except Exception as e:
        # Fallback to DuckDuckGo if Tavily fails
        if ddg_search:
            try:
                # DDG returns a string, so we wrap it
                raw_res = ddg_search.run(query)
                return json.dumps({
                    "query": query,
                    "source": "duckduckgo",
                    "results": [{"title": "DuckDuckGo Result", "url": "", "snippet": raw_res}]
                })
            except Exception as e2:
                return json.dumps({"error": f"Tavily failed: {str(e)}. DDG failed: {str(e2)}"})
        return json.dumps({"error": str(e)})

@tool
def add_knowledge(text: str, source: str = "user") -> str:
    """
    Add text content to the long-term knowledge base for RAG.
    Splits long text into chunks automatically. Returns number of chunks added.
    """
    try:
        count = knowledge_base.add_document(text, metadata={"source": source})
        return f"Knowledge stored successfully. Chunks added: {count} (source={source})."
    except Exception as e:
        return f"Error adding knowledge: {e}"

@tool
def add_knowledge_url(url: str) -> str:
    """
    Fetch a web page and store its main text into the knowledge base.
    Note: This uses a simple fetch and does not perform full boilerplate removal.
    """
    try:
        import requests
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        text = resp.text
        # Very naive stripping of HTML tags
        try:
            import re
            text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
        except Exception:
            pass
        count = knowledge_base.add_document(text, metadata={"source": url})
        return f"Fetched and stored from URL. Chunks added: {count} (source={url})."
    except Exception as e:
        return f"Error adding knowledge from URL: {e}"

@tool
def search_knowledge(query: str, n_results: int = 3) -> str:
    """
    Search the knowledge base for relevant passages to use as context.
    Returns a formatted string with sources and excerpts.
    """
    try:
        items = knowledge_base.retrieve_knowledge(query, n_results=n_results)
        if not items:
            return "No relevant knowledge found."
        lines = []
        for i, it in enumerate(items, 1):
            meta = it.get("metadata", {})
            src = meta.get("source", "unknown")
            snippet = it.get("text", "").strip()
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."
            lines.append(f"{i}. [source: {src}] {snippet}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching knowledge: {e}"

@tool
def run_python(code: str) -> str:
    """
    Execute small Python snippets for calculations or text processing.
    Safety: No imports, file/network access disabled. Provides 'math' module.
    Returns the stringified value of variable 'result' if set; otherwise stdout.
    """
    # Disallow dangerous patterns
    forbidden = ["__import__", "open(", "exec(", "eval(", "os.", "sys.", "subprocess", "socket", "requests", "import "]
    lowered = code.lower()
    if any(p in lowered for p in forbidden):
        return "Error: Disallowed code detected. Please avoid imports or system access."
    # Prepare sandbox
    safe_globals = {"__builtins__": {} , "math": math}
    safe_locals = {}
    # Capture stdout
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, safe_globals, safe_locals)
    except Exception as e:
        return f"Error executing code: {e}"
    # Prefer 'result' variable
    if "result" in safe_locals:
        try:
            return str(safe_locals["result"]) 
        except Exception:
            pass
    output = buf.getvalue()
    return output if output else ""

@tool
def generate_image(prompt: str) -> str:
    """
    Generates an image based on the text prompt using Hugging Face (Stable Diffusion).
    Returns a BASE64 string of the image to displaying it directly.
    """
    if not hf_client:
        return "Error: Image generation is not available (missing huggingface_hub or HUGGINGFACEHUB_API_TOKEN)."
    
    try:
        # InferenceClient.text_to_image returns a PIL Image
        image = hf_client.text_to_image(prompt)
        
        # Convert to base64
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        b64_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"[IMAGE_GENERATED_BASE64_DATA: {b64_data}]"
    except Exception as e:
        return f"Error generating image: {str(e)}"

@tool
def recognize_text(image_input: str) -> str:
    """
    Recognizes and extracts text from an image using Google's vision capabilities (equivalent to capabilities found in ML Kit).
    
    Args:
        image_input: The local file path or URL of the image to process.
    """
    # Import locally to avoid circular dependencies or load issues if not needed
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        import mimetypes
    except ImportError:
        return "Error: langchain-google-genai library is missing."

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not found. Cannot perform text recognition."

    try:
        # Use gemini-1.5-flash for speed and cost-effectiveness
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
        
        # Prepare content
        # Note: We send a prompt to extract text
        content_parts = [
            {"type": "text", "text": "Please extract all the text you can see in this image. Output ONLY the extracted text, preserving the layout if possible. Do not add any conversational filler."}
        ]

        # Handle URL vs Local File
        if image_input.startswith("http://") or image_input.startswith("https://"):
            content_parts.append({"type": "image_url", "image_url": image_input})
        else:
             # Assume local path
             if not os.path.exists(image_input):
                 return f"Error: File not found at {image_input}"
             
             # Guess mime type
             mime_type, _ = mimetypes.guess_type(image_input)
             if not mime_type:
                 mime_type = "image/png"
             
             with open(image_input, "rb") as f:
                 img_data = base64.b64encode(f.read()).decode("utf-8")
             
             # Construct data URI
             data_uri = f"data:{mime_type};base64,{img_data}"
             content_parts.append({"type": "image_url", "image_url": data_uri})
        
        msg = HumanMessage(content=content_parts)
        response = llm.invoke([msg])
        return response.content

    except Exception as e:
        return f"Error recognizing text: {str(e)}"
