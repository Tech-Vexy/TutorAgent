import os
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv
import logging

load_dotenv()

# Monkey patch pydantic.BaseSettings for older/incompatible libraries (like specific chromadb versions)
# Must be before any library imports chromadb
try:
    import pydantic
    # Try to use V1 BaseSettings shim provided by Pydantic V2
    if hasattr(pydantic, 'v1') and hasattr(pydantic.v1, 'BaseSettings'):
        pydantic.BaseSettings = pydantic.v1.BaseSettings
    # If not available (e.g. truly old pydantic?), try pydantic_settings (V2 behavior, caused validation errors)
    # But for compatibility, v1 is preferred.
    elif not hasattr(pydantic, 'BaseSettings'):
         try:
             from pydantic_settings import BaseSettings
             pydantic.BaseSettings = BaseSettings
         except ImportError:
             pass
except ImportError:
    pass


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TopScoreServer")

# Enable LangSmith tracing if API key is provided (allow disabling via DISABLE_TRACING=1)
if os.getenv("LANGCHAIN_API_KEY") and os.getenv("DISABLE_TRACING", "0") != "1":
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "TopScore-AI"))

import json
import base64
import io
import asyncio
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from groq import Groq

try:
    # Try to import optimized agent first for better performance
    from agent_graph_optimized import graph, knowledge_base
    logger.info("✅ Using OPTIMIZED agent graph (2-4x faster)")
except ImportError:
    # Fallback to regular agent
    from agent_graph import graph, knowledge_base
    logger.warning("Using standard agent graph. For faster responses, create agent_graph_optimized.py")
from database import get_or_create_profile, init_db, upsert_thread_metadata
from model_manager import model_manager

# Optional Groq Compound Systems adapter
from groq_system import systems_supported, stream_compound_response

# --- Simple in-memory LRU cache for responses ---
class _ResponseCache:
    def __init__(self, max_entries: int, ttl_seconds: int):
        self._store = OrderedDict()
        self.max_entries = max_entries
        self.ttl = ttl_seconds

    def _now(self) -> float:
        return time.time()

    def get(self, key: str):
        if key in self._store:
            entry = self._store[key]
            # TTL check
            if self._now() - entry["ts"] > self.ttl:
                try:
                    del self._store[key]
                except Exception:
                    pass
                return None
            # Update recency
            self._store.move_to_end(key)
            return entry
        return None

    def set(self, key: str, value: dict):
        self._store[key] = {
            **value,
            "ts": self._now()
        }
        self._store.move_to_end(key)
        # Enforce max size
        while len(self._store) > self.max_entries:
            try:
                self._store.popitem(last=False)
            except Exception:
                break

# Initialize global cache instance (can be disabled via env)
_response_cache = _ResponseCache(
    max_entries=int(os.getenv("CACHE_MAX_ENTRIES", "256")),
    ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600"))
)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _make_cache_key(message_text: str, image_data: Optional[str]) -> str:
    # If image is provided by user, include its sha1 digest in key; otherwise text-only key
    if image_data:
        try:
            import hashlib
            payload = image_data.split(",", 1)[1] if "," in image_data else image_data
            digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
            return f"img:{digest}|text:{_normalize_text(message_text)}"
        except Exception:
            return f"img:unknown|text:{_normalize_text(message_text)}"
    return f"text:{_normalize_text(message_text)}"

# --- Configuration ---
DB_URI = os.getenv("DB_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ENABLE_TTS = os.getenv("ENABLE_TTS", "0")  # Set to "1" to enable TTS if supported
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "100"))
# Response Cache configuration (in-memory LRU)
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "1") == "1"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour default
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "256"))
# Groq Compound Systems toggle
USE_GROQ_SYSTEM = os.getenv("USE_GROQ_SYSTEM", "0") == "1"
GROQ_SYSTEM_ID = os.getenv("GROQ_SYSTEM_ID")
# Graph recursion handling controls
MAX_GRAPH_RETRIES = int(os.getenv("MAX_GRAPH_RETRIES", "1"))
RECURSION_LIMIT_STEP = int(os.getenv("RECURSION_LIMIT_STEP", "50"))
# Wall-clock safeguard for a single turn (seconds); 0 disables timeout
GRAPH_MAX_SECONDS = int(os.getenv("GRAPH_MAX_SECONDS", "20"))

# DB Pool Configuration
DB_POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "2"))
DB_POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "10"))

if not DB_URI:
    logger.warning("DB_URI not set. Persistence will fail if not configured.")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set. STT/TTS features will be disabled.")
# Notify if systems are enabled
if USE_GROQ_SYSTEM:
    logger.info("USE_GROQ_SYSTEM=1; will attempt to use Groq Compound Systems (groq/compound).")

# Initialize Groq Client for Audio (guarded)
try:
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception as _e:
    logger.warning(f"Failed to initialize Groq client: {_e}")
    groq_client = None

# --- OpenTelemetry Integration ---
try:
    from otel_setup import setup_opentelemetry
except ImportError:
    setup_opentelemetry = None

# --- Lifespan Manager for DB Pool ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create DB pool
    # Add connection recycling to prevent SSL timeouts
    pool = AsyncConnectionPool(
        conninfo=DB_URI, 
        max_size=DB_POOL_MAX_SIZE, 
        min_size=DB_POOL_MIN_SIZE,
        open=False,
        kwargs={
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
            "connect_timeout": 60
        }
    )
    await pool.open()
    app.state.pool = pool
    
    # Initialize SQLAlchemy Tables
    await init_db()
    
    # Initialize Checkpointer tables once
    # We use a dedicated connection for setup to ensure autocommit for CREATE INDEX CONCURRENTLY
    async with pool.connection() as conn:
        await conn.set_autocommit(True)
        checkpointer = AsyncPostgresSaver(conn)
        await checkpointer.setup()
        
        # Create user_threads table for history tracking
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_threads (
                thread_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_threads_user_id ON user_threads (user_id);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_threads_updated_at ON user_threads (updated_at DESC);")
    
    yield
    # Shutdown: Close DB pool
    await pool.close()

# --- API Documentation ---
API_TITLE = "TopScore AI Tutor API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
# TopScore AI Tutor API 🎓

A powerful AI-powered tutoring system designed for Kenyan students (KCSE/CBC curriculum).

## Features

- 🤖 **Intelligent Tutoring**: Multi-model AI with deep reasoning capabilities
- 🌍 **Multilingual Support**: Responds in Kiswahili for Swahili subjects, English otherwise
- 📚 **Knowledge Ingestion**: Upload PDFs, URLs, or connect Google Drive
- 💾 **Persistent Memory**: Remembers past interactions and learning progress
- 📊 **Adaptive Learning**: Tracks strengths, weaknesses, and preferred learning style
- 🎯 **KCSE/CBC Focused**: Tailored for Kenyan education system

## Authentication

Currently, the API uses user IDs for identification. No authentication token required.

## WebSocket

For real-time chat, connect to `/ws/chat` with JSON messages.

## Language Support

The API automatically detects and responds in:
- **Kiswahili** 🇰🇪: When the subject is Swahili/Kiswahili
- **English** 🇬🇧: For all other subjects

Users can override this in their profile settings.
"""

API_TAGS = [
    {
        "name": "Chat",
        "description": "Real-time chat endpoints for tutoring interactions"
    },
    {
        "name": "Threads",
        "description": "Conversation thread management and message history"
    },
    {
        "name": "Knowledge",
        "description": "Knowledge base ingestion - upload documents, URLs, or connect Google Drive"
    },
    {
        "name": "Profile",
        "description": "User learning profile management - preferences, strengths, weaknesses"
    },
    {
        "name": "Models",
        "description": "AI model configuration and management"
    },
    {
        "name": "System",
        "description": "System health, info, and configuration endpoints"
    }
]

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    openapi_tags=API_TAGS,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "TopScore AI Support",
        "url": "https://github.com/Tech-Vexy/TutorAgent",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Setup OpenTelemetry
if setup_opentelemetry:
    setup_opentelemetry(app)
    logger.info("OpenTelemetry setup called.")
else:
    logger.warning("OpenTelemetry setup skipped (missing dependencies).")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "null"  # For file:// protocol
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional

# --- API Request/Response Models ---

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="The user's message/question", example="What is photosynthesis?")
    user_id: str = Field("default_user", description="Unique identifier for the user", example="student_001")
    thread_id: str = Field("default_thread", description="Conversation thread ID for context continuity", example="thread_abc123")
    model_preference: str = Field("fast", description="Model to use: 'fast' (quick responses) or 'smart' (deep reasoning)", example="smart")
    stream: bool = Field(True, description="Whether to stream the response (SSE format)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Explain the water cycle in simple terms",
                "user_id": "student_001",
                "thread_id": "science_lesson_1",
                "model_preference": "smart",
                "stream": True
            }
        }

class MessageResponse(BaseModel):
    """A single message in a conversation"""
    type: str = Field(..., description="Message type: 'user' or 'ai'", example="ai")
    content: str = Field(..., description="Message content", example="Photosynthesis is the process...")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of the message")

class ThreadInfo(BaseModel):
    """Information about a conversation thread"""
    thread_id: str = Field(..., description="Unique thread identifier")
    title: str = Field(..., description="Thread title (first message preview)")
    updated_at: str = Field(..., description="Last update timestamp")

class ProfileUpdate(BaseModel):
    """Request to update a user preference"""
    key: str = Field(..., description="Preference key to update", example="preferred_style")
    value: str = Field(..., description="New value for the preference", example="visual")
    
    class Config:
        json_schema_extra = {
            "example": {
                "key": "preferred_language",
                "value": "sw"
            }
        }

class TopicRecord(BaseModel):
    """Record of a learning interaction on a topic"""
    topic: str = Field(..., description="Topic name", example="Algebra")
    was_correct: bool = Field(True, description="Whether the user answered correctly")

class UrlRequest(BaseModel):
    """Request to ingest content from a URL"""
    url: str = Field(..., description="URL to scrape and ingest into knowledge base", example="https://example.com/biology-notes")

class DriveIngestRequest(BaseModel):
    """Request to ingest a file from Google Drive"""
    file_id: str = Field(..., description="Google Drive file ID")
    file_name: str = Field(..., description="Name of the file for reference")

class ModelUpdate(BaseModel):
    """Request to update AI model configuration"""
    type: str = Field(..., description="Model type: 'fast', 'smart', or 'vision'", example="fast")
    model_id: str = Field(..., description="New model identifier", example="llama-3.1-8b-instant")
    persist: bool = Field(False, description="Whether to save to .env file")

class StatusResponse(BaseModel):
    """Standard API status response"""
    status: str = Field(..., description="Response status: 'success' or 'error'")
    message: Optional[str] = Field(None, description="Additional message or error details")

class KnowledgeUploadResponse(BaseModel):
    """Response after knowledge ingestion"""
    status: str = Field(..., description="Response status")
    chunks_added: Optional[int] = Field(None, description="Number of chunks added to knowledge base")
    message: str = Field(..., description="Result message")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Server status", example="healthy")
    database: str = Field(..., description="Database connection status")
    langsmith_tracing: Optional[bool] = Field(None, description="Whether LangSmith tracing is enabled")

@app.post("/chat", tags=["Chat"], summary="Send a message and get AI response")
async def chat_endpoint(request: ChatRequest):
    """
    Send a message to the AI tutor and receive a streaming response.
    
    The AI automatically:
    - Detects if the subject is Swahili and responds in Kiswahili
    - Uses appropriate teaching strategies (Socratic or Direct)
    - Retrieves relevant knowledge from the knowledge base
    - Adapts to the user's learning profile
    
    **Response Format (SSE):**
    - `{"type": "message", "content": "..."}`  - Text chunks
    - `{"type": "image", "content": "base64..."}` - Generated images
    - `data: [DONE]` - End of response
    """
    async def event_generator():
        # Setup similar to WebSocket
        user_id = request.user_id
        thread_id = request.thread_id
        message_text = request.message
        
        # Profile
        user_profile = await get_or_create_profile(user_id)
        
        # Message
        human_message = HumanMessage(content=message_text)
        
        # Config
        run_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": GRAPH_RECURSION_LIMIT}
        
        # Run Graph
        from agent_graph import workflow
        app_graph = workflow.compile()
        
        async for event in app_graph.astream_events(
            {
                "messages": [human_message],
                "model_preference": request.model_preference,
                "user_profile": user_profile,
                "tool_invocations": 0
            },
            config=run_config,
            version="v1"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                 # Filter internal nodes
                node_name = event.get("metadata", {}).get("langgraph_node")
                if node_name in ["planner", "reflection"]:
                    continue
                
                content = event["data"]["chunk"].content
                if content:
                    # Stream structured chunks for frontend
                    yield f"data: {json.dumps({'type': 'message', 'content': content})}\n\n"
            
            elif kind == "on_tool_end":
                tool_output = event["data"].get("output")
                if tool_output and isinstance(tool_output, str) and "[IMAGE_GENERATED_BASE64_DATA:" in tool_output:
                     yield f"data: {json.dumps({'type': 'token', 'content': tool_output})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/memory/{user_id}", tags=["Profile"], summary="Get user memory state")
async def get_memory(user_id: str):
    """
    Retrieve the user's profile and episodic memory state.
    
    Returns information about past interactions and user preferences.
    """
    profile = await get_or_create_profile(user_id)
    return profile

# --- WebSocket Endpoint ---
@app.websocket("/ws/chat")

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Get the pool from app state
    pool = app.state.pool
    
    # Initialize Checkpointer with the pool
    checkpointer = AsyncPostgresSaver(pool)
    
    # Setup is handled in lifespan
    # await checkpointer.setup()

    user_id = "unknown"
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            user_id = payload.get("user_id", "default_user")
            thread_id = payload.get("thread_id", "default_thread")
            message_text = payload.get("message", "")
            image_data = payload.get("image_data") # Base64 string or URL
            audio_data = payload.get("audio_data") # Base64 string of audio
            model_preference = payload.get("model_preference", "fast") # "fast", "smart", or "adk"
            
            logger.info(f"Received message from user={user_id}, thread={thread_id}, model={model_preference}")
            
            # --- ADK Integration ---
            if model_preference == "adk":
                try:
                    from adk_module import get_adk_response
                    logger.info("Routing to Google ADK Agent...")
                    # ADK implementation in this module is synchronous/blocking for now
                    response_text = await asyncio.to_thread(get_adk_response, message_text)
                    
                    await websocket.send_json({"type": "token", "content": response_text})
                    await websocket.send_json({"type": "end_turn"})
                    continue
                except Exception as e:
                    logger.error(f"ADK Error: {e}")
                    await websocket.send_json({"type": "error", "content": f"ADK Error: {str(e)}"})
                    continue
            
            # --- Audio Processing (STT) ---
            if audio_data and groq_client is not None:
                try:
                    logger.info("Processing audio input...")
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(audio_data.split(",")[1] if "," in audio_data else audio_data)
                    # Create a file-like object
                    audio_file = io.BytesIO(audio_bytes)
                    audio_file.name = "input.wav" # Groq needs a filename
                    
                    # Transcribe using Whisper Large V3
                    transcription = groq_client.audio.transcriptions.create(
                        file=(audio_file.name, audio_file.read()),
                        model="whisper-large-v3",
                        response_format="json",
                        language="en",
                        temperature=0.0
                    )
                    message_text = transcription.text
                    logger.info(f"STT Transcription: {message_text}")
                    # Send transcription back to client
                    await websocket.send_json({"type": "transcription", "content": message_text})
                except Exception as e:
                    logger.error(f"STT Error: {e}")
                    await websocket.send_json({"type": "error", "content": f"STT Error: {str(e)}"})
            elif audio_data and groq_client is None:
                logger.warning("STT disabled: missing GROQ_API_KEY")
                await websocket.send_json({"type": "error", "content": "STT disabled: missing GROQ_API_KEY"})
            
            # Update thread metadata in DB using SQLAlchemy engine helper to avoid stale pool connections
            try:
                title = message_text[:50] + "..." if len(message_text) > 50 else message_text
                if not title:
                    title = "New Chat"
                await upsert_thread_metadata(thread_id, user_id, title)
            except Exception as e:
                logger.error(f"DB Error updating thread: {e}")

            # Fetch User Profile
            user_profile = await get_or_create_profile(user_id)

            # Construct LangChain Message
            content = [{"type": "text", "text": message_text}]
            
            if image_data:
                # If it's a base64 string, format it for Llama 3.2 Vision
                # Assuming standard data URI format: "data:image/jpeg;base64,..."
                image_url = {"url": image_data}
                content.append({"type": "image_url", "image_url": image_url})
            
            human_message = HumanMessage(content=content)
            
            # Config for run settings
            run_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": GRAPH_RECURSION_LIMIT}
            
            # Run the graph WITHOUT checkpointer to avoid SSL issues
            from agent_graph import workflow
            app_graph = workflow.compile()  # No checkpointer
            
            full_response_text = ""
            events_to_cache = []
            cache_key = _make_cache_key(message_text, image_data)

            # Cache lookup and fast replay
            if CACHE_ENABLED:
                try:
                    cached = _response_cache.get(cache_key)
                except Exception:
                    cached = None
                if cached and cached.get("events"):
                    logger.info("Cache hit! Replaying response.")
                    # Replay cached events
                    for ev in cached["events"]:
                        try:
                            await websocket.send_json(ev)
                        except Exception:
                            pass
                    full_response_text = cached.get("full_text", "")
                    # Finish this turn immediately
                    await websocket.send_json({"type": "end_turn"})
                    continue

            # Try Groq Compound System first (text-only), then fall back to LangGraph
            used_system = False
            if USE_GROQ_SYSTEM and systems_supported() and not image_data:
                try:
                    logger.info("Using Groq Compound System...")
                    for token in stream_compound_response(message_text, session_id=thread_id, system_id=GROQ_SYSTEM_ID):
                        if token:
                            full_response_text += token
                            ev = {"type": "token", "content": token}
                            events_to_cache.append(ev)
                            await websocket.send_json(ev)
                    used_system = True
                except Exception as e:
                    logger.error(f"Groq System error, falling back to LangGraph: {e}")
                    used_system = False
            
            if not used_system:
                logger.info("Using LangGraph workflow...")
                # Enhanced recursion handling: retries with increased limit and an optional wall-clock timeout
                attempts = 0
                current_limit = GRAPH_RECURSION_LIMIT

                async def _consume_events(limit: int):
                    local_config = {**run_config, "recursion_limit": limit}
                    async for event in app_graph.astream_events(
                        {
                            "messages": [human_message],
                            "model_preference": model_preference,
                            "user_profile": user_profile,
                            "tool_invocations": 0
                        },
                        config=local_config,
                        version="v1"
                    ):
                        kind = event["event"]
                        # logger.debug(f"Event: {kind}") # Debug logging
                        
                        # Handle Status Updates based on Node Entry
                        if kind == "on_chain_start":
                            node_name = event.get("metadata", {}).get("langgraph_node")
                            if node_name:
                                status_text = None
                                if node_name == "planner":
                                    status_text = "Planning solution..."
                                elif node_name == "vision_analysis":
                                    status_text = "Analyzing image..."
                                elif node_name == "tools":
                                    status_text = "Using tools..."
                                elif node_name == "reflection":
                                    status_text = "Reflecting on interaction..."
                                elif node_name == "deep_thinker":
                                    status_text = "Thinking..."
                                
                                if status_text:
                                    await websocket.send_json({"type": "status", "content": status_text})

                        if kind == "on_chat_model_stream":
                            # Filter out tokens from internal nodes (planner, reflection)
                            node_name = event.get("metadata", {}).get("langgraph_node")
                            if node_name in ["planner", "reflection"]:
                                continue

                            content = event["data"]["chunk"].content
                            if content:
                                nonlocal full_response_text
                                full_response_text += content
                                ev = {"type": "token", "content": content}
                                events_to_cache.append(ev)
                                await websocket.send_json(ev)

                        elif kind == "on_tool_end":
                            # Send tool output (e.g., plots)
                            tool_output = event["data"].get("output")
                            if tool_output:
                                logger.info(f"Tool execution finished: {event['name']}")
                                # Check for base64 image data in the output string
                                if isinstance(tool_output, str) and "[IMAGE_GENERATED_BASE64_DATA:" in tool_output:
                                    try:
                                        # Extract base64 data
                                        start_marker = "[IMAGE_GENERATED_BASE64_DATA:"
                                        start = tool_output.find(start_marker) + len(start_marker)
                                        end = tool_output.find("]", start)
                                        if end != -1:
                                            b64_data = tool_output[start:end].strip()
                                            ev_img = {"type": "image", "content": b64_data}
                                            events_to_cache.append(ev_img)
                                            await websocket.send_json(ev_img)
                                    except Exception as e:
                                        logger.error(f"Error extracting image data: {e}")

                while True:
                    try:
                        if GRAPH_MAX_SECONDS > 0:
                            await asyncio.wait_for(_consume_events(current_limit), timeout=GRAPH_MAX_SECONDS)
                        else:
                            await _consume_events(current_limit)
                        break  # success
                    except GraphRecursionError as e:
                        if attempts < MAX_GRAPH_RETRIES:
                            attempts += 1
                            current_limit += RECURSION_LIMIT_STEP
                            try:
                                logger.info(f"Graph recursion limit hit, retrying ({attempts}/{MAX_GRAPH_RETRIES})...")
                                await websocket.send_json({
                                    "type": "info",
                                    "content": f"This seems complex; expanding reasoning depth and retrying ({attempts}/{MAX_GRAPH_RETRIES})..."
                                })
                            except Exception:
                                pass
                            continue
                        # Final failure after retries
                        logger.error(f"Graph recursion limit exceeded: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "content": f"The task became too recursive and hit the safety limit ({current_limit}). Try rephrasing or simplifying your request."
                        })
                        break
                    except asyncio.TimeoutError:
                        logger.error("Graph execution timed out.")
                        await websocket.send_json({
                            "type": "error",
                            "content": "The request took too long and was stopped to keep the app responsive. Please try a shorter or simpler query."
                        })
                        break

            # Cache store (on miss) before generating audio
            if CACHE_ENABLED and events_to_cache:
                try:
                    _response_cache.set(cache_key, {"events": events_to_cache, "full_text": full_response_text})
                except Exception:
                    pass

            # --- Audio Generation (TTS) ---
            # Generate audio only if explicitly enabled and client is available
            if full_response_text and ENABLE_TTS == "1" and groq_client is not None:
                try:
                    logger.info("Generating TTS audio...")
                    if len(full_response_text) < 1000 and hasattr(groq_client, "audio") and hasattr(groq_client.audio, "speech"):
                        response = groq_client.audio.speech.create(
                            model="playai-tts",
                            voice="Angelo-PlayAI",
                            input=full_response_text
                        )
                        # Extract bytes
                        if hasattr(response, "content"):
                            audio_bytes = response.content
                        elif hasattr(response, "read"):
                            audio_bytes = response.read()
                        else:
                            audio_bytes = bytes(response)
                        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                        await websocket.send_json({"type": "audio_response", "content": f"data:audio/mp3;base64,{audio_b64}"})
                except Exception as e:
                    logger.error(f"TTS Error (PlayAI/Groq): {e}")
            elif full_response_text and ENABLE_TTS == "1" and groq_client is None:
                logger.warning("TTS disabled: missing GROQ_API_KEY")
                await websocket.send_json({"type": "error", "content": "TTS disabled: missing GROQ_API_KEY"})

            await websocket.send_json({"type": "end_turn"})
            
    except WebSocketDisconnect:
        logger.info(f"Client {user_id} disconnected")
    except Exception as e:
        logger.error(f"Error: {e}")
        # Debugging for Tool Call Errors
        if hasattr(e, 'response'):
             logger.error(f"Error Response: {e.response}")
        # LangChain often attaches 'failed_generation' to OutputParserExceptions
        if hasattr(e, 'failed_generation'):
             logger.error(f"Failed Generation: {e.failed_generation}")
        
        import traceback
        logger.error(traceback.format_exc())
        
        await websocket.close()

@app.get("/", tags=["System"], summary="API Root")
async def root():
    """Root endpoint - confirms the API is running."""
    return {"message": "TopScore AI Backend is Running"}

@app.get("/developer", tags=["System"], summary="Developer Guide")
def developer_guide():
    """
    Returns the HTML Developer Guide with integration details.
    """
    with open("api_documentation.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/info", tags=["System"], summary="Get server information")
async def get_server_info():
    """
    Get detailed server information including:
    - Current status and version
    - Available features
    - Configured AI models
    - Available API endpoints
    """
    return {
        "status": "running",
        "version": "1.0.0",
        "features": {
            "threads": True,
            "websocket": True,
            "message_persistence": True,
            "groq_models": True,
            "knowledge_ingestion": True,
            "multilingual": True,
            "adaptive_learning": True
        },
        "models": model_manager.get_current_config(),
        "endpoints": {
            "websocket": "/ws/chat",
            "threads": "/threads/{user_id}",
            "messages": "/threads/{thread_id}/messages",
            "save_message": "/threads/save_message",
            "direct_messages": "/threads/{thread_id}/messages_direct",
            "update_model": "/models/update",
            "upload_knowledge": "/knowledge/upload",
            "add_url": "/knowledge/url",
            "profile": "/profile/{user_id}",
            "drive_files": "/knowledge/drive/list",
            "developer_guide": "/developer"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }
    }

class ModelUpdate(BaseModel):
    type: str  # fast, smart, vision
    model_id: str
    persist: bool = False

@app.post("/models/update", tags=["Models"], summary="Update AI model configuration")
async def update_model_endpoint(update: ModelUpdate):
    """
    Dynamically update the AI model for a specific type.
    
    - **fast**: Quick responses, used for simple queries
    - **smart**: Deep reasoning, used for complex problems
    - **vision**: Image analysis capabilities
    
    Set `persist=true` to save the change to `.env` file.
    """
    try:
        model_manager.update_model(update.type, update.model_id, update.persist)
        return {"status": "success", "config": model_manager.get_current_config()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/threads/{user_id}", tags=["Threads"], summary="Get user's conversation threads")
async def get_user_threads(user_id: str):
    """
    Retrieve all conversation threads for a user, ordered by most recent.
    
    Returns a list of threads with their IDs, titles, and last update times.
    """
    pool = app.state.pool
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT thread_id, title, updated_at FROM user_threads WHERE user_id = %s ORDER BY updated_at DESC",
            (user_id,)
        )
        rows = await cursor.fetchall()
        return [{"thread_id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]

@app.get("/threads/{thread_id}/messages", tags=["Threads"], summary="Get messages in a thread")
async def get_thread_messages(thread_id: str):
    """
    Retrieve all messages in a conversation thread.
    
    Returns messages in chronological order with their type (user/ai) and content.
    """
    try:
        pool = app.state.pool
        checkpointer = AsyncPostgresSaver(pool)
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = await checkpointer.aget(config)  # ← Changed from .get() to .aget()
        
        if not checkpoint:
            return []
        
        # Extract messages from checkpoint
        # The state structure depends on agent_graph.py
        # Usually state['messages']
        messages = checkpoint.get("channel_values", {}).get("messages", [])
        
        # Format for client
        formatted_messages = []
        for msg in messages:
            try:
                msg_type = msg.type
                content = msg.content
                # Handle list content (multimodal)
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
                    content = " ".join(text_parts) if text_parts else str(content)
                
                formatted_messages.append({
                    "type": "user" if msg_type == "human" else "ai",
                    "content": content,
                })
            except Exception as e:
                logger.warning(f"Error formatting message: {e}")
                continue
            
        return formatted_messages
    except Exception as e:
        logger.error(f"Error getting thread messages: {e}")
        return []

@app.post("/threads/save_message")
async def save_message_to_thread(request: dict):
    """
    Save a single message to a thread incrementally.
    Prevents data loss if WebSocket disconnects.
    """
    try:
        thread_id = request.get("thread_id")
        user_id = request.get("user_id")
        message = request.get("message", "")
        message_type = request.get("message_type", "user")
        
        pool = app.state.pool
        
        # Update thread metadata
        if message_type == "user" and message:
            title = message[:50] + ("..." if len(message) > 50 else "")
            await upsert_thread_metadata(thread_id, user_id, title)
        
        # Save to messages table
        async with pool.connection() as conn:
            # Create table if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS thread_messages (
                    id SERIAL PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_thread_messages_thread_id 
                ON thread_messages(thread_id, created_at)
            """)
            
            # Insert message
            await conn.execute("""
                INSERT INTO thread_messages (thread_id, user_id, message_type, content)
                VALUES (%s, %s, %s, %s)
            """, (thread_id, user_id, message_type, message))
            
        return {"status": "saved"}
    except Exception as e:
        logger.error(f"Error saving message: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/threads/{thread_id}/messages_direct")
async def get_thread_messages_direct(thread_id: str):
    """
    Get messages from thread_messages table (faster, more reliable).
    Falls back to checkpoint if no messages found.
    """
    try:
        pool = app.state.pool
        
        async with pool.connection() as conn:
            # Check if table exists
            try:
                cursor = await conn.execute("""
                    SELECT message_type, content, created_at
                    FROM thread_messages
                    WHERE thread_id = %s
                    ORDER BY created_at ASC
                """, (thread_id,))
                
                rows = await cursor.fetchall()
                
                if rows:
                    return [{
                        "type": row[0],
                        "content": row[1],
                        "timestamp": row[2].isoformat() if row[2] else None
                    } for row in rows]
            except:
                pass
        
        # Fallback to checkpoint method
        return await get_thread_messages(thread_id)
        
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        return []

@app.post("/knowledge/upload", tags=["Knowledge"], summary="Upload a document to knowledge base")
async def upload_knowledge(file: UploadFile = File(...)):
    """
    Upload a PDF or text file to the knowledge base.
    
    The content is automatically:
    - Extracted (text from PDF or plain text)
    - Chunked for efficient retrieval
    - Indexed for semantic search
    
    Supported formats: `.pdf`, `.txt`, `.md`
    """
    try:
        content = ""
        filename = file.filename.lower()
        
        if filename.endswith(".pdf"):
            import pypdf
            reader = pypdf.PdfReader(file.file)
            for page in reader.pages:
                content += page.extract_text() + "\n"
        else:
            # Assume text
            content_bytes = await file.read()
            content = content_bytes.decode("utf-8", errors="ignore")
            
        if not content.strip():
            return {"status": "error", "message": "File is empty or could not be read."}
            
        count = knowledge_base.add_document(content, metadata={"source": file.filename})
        return {"status": "success", "chunks_added": count, "message": f"Successfully ingested {file.filename}"}
        
    except ImportError:
        return {"status": "error", "message": "pypdf not installed. Server cannot process PDFs."}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return {"status": "error", "message": str(e)}

class UrlRequest(BaseModel):
    url: str

@app.post("/knowledge/url", tags=["Knowledge"], summary="Ingest content from URL")
async def add_knowledge_url_endpoint(request: UrlRequest):
    """
    Scrape and ingest content from a web URL into the knowledge base.
    
    The URL content is:
    - Fetched and HTML tags stripped
    - Cleaned and chunked
    - Indexed for semantic search
    """

    try:
        import requests as req
        from bs4 import BeautifulSoup
        
        resp = req.get(request.url, timeout=10)
        resp.raise_for_status()
        
        # Simple HTML stripping
        try:
            soup = BeautifulSoup(resp.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator=" ", strip=True)
        except ImportError:
            import re
            text = resp.text
            text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = " ".join(text.split())

        count = knowledge_base.add_document(text, metadata={"source": request.url})
        return {"status": "success", "chunks_added": count, "message": f"Successfully ingested {request.url}"}
        
    except Exception as e:
        logger.error(f"URL ingestion error: {e}")
        return {"status": "error", "message": str(e)}


# --- GOOGLE DRIVE ENDPOINTS ---

@app.get("/knowledge/drive/list", tags=["Knowledge"], summary="List Google Drive files")
async def list_drive_files(q: Optional[str] = None):
    """
    List files from connected Google Drive.
    
    Requires `drive_credentials.json` or `service_account.json` in the server root.
    
    - **q**: Optional search query to filter files by name
    """
    try:
        from google_drive_connector import drive_connector
        files = drive_connector.list_files(query=q)
        return {"status": "success", "files": files}
    except FileNotFoundError:
        return {"status": "error", "message": "Missing drive_credentials.json"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class DriveIngestRequest(BaseModel):
    file_id: str
    file_name: str

@app.post("/knowledge/drive/ingest", tags=["Knowledge"], summary="Ingest file from Google Drive")
async def ingest_drive_file(request: DriveIngestRequest):
    """
    Download and ingest a file from Google Drive into the knowledge base.
    
    Supports large files with streaming download.
    Automatically processes PDFs and text files.
    """
    try:
        from google_drive_connector import drive_connector
        
        # Use new streaming method
        # We pass the knowledge_base instance directly
        count, message = drive_connector.download_and_ingest_streaming(
            request.file_id, 
            request.file_name, 
            knowledge_base
        )
        
        if count == 0 and "Error" in message:
            return {"status": "error", "message": message}
            
        return {"status": "success", "chunks_added": count, "message": message}
    except Exception as e:
        logger.error(f"Drive ingestion error: {e}")
        return {"status": "error", "message": str(e)}


# --- LEARNING PROFILE ENDPOINTS ---

@app.get("/profile/{user_id}", tags=["Profile"], summary="Get user's learning profile")
async def get_user_profile(user_id: str):
    """
    Get the complete learning profile for a user.
    
    Returns:
    - Topics studied with accuracy statistics
    - Identified strengths and weaknesses
    - Learning preferences (style, difficulty, language)
    - Session statistics
    """
    try:
        from learning_profile import get_learning_profile
        profile = get_learning_profile(user_id)
        return {"status": "success", "profile": profile.data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class ProfileUpdate(BaseModel):
    key: str
    value: str

@app.post("/profile/{user_id}/preference", tags=["Profile"], summary="Update user preference")
async def update_user_preference(user_id: str, update: ProfileUpdate):
    """
    Update a user's learning preference.
    
    Available preferences:
    - **preferred_style**: 'balanced', 'visual', 'textual', 'interactive'
    - **difficulty_level**: 'easy', 'medium', 'hard'
    - **preferred_language**: 'auto', 'en' (English), 'sw' (Kiswahili)
    """
    try:
        from learning_profile import get_learning_profile
        profile = get_learning_profile(user_id)
        profile.set_preference(update.key, update.value)
        return {"status": "success", "profile": profile.data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class TopicRecord(BaseModel):
    topic: str
    was_correct: bool = True

@app.post("/profile/{user_id}/record", tags=["Profile"], summary="Record learning interaction")
async def record_topic_interaction(user_id: str, record: TopicRecord):
    """
    Record a learning interaction for a specific topic.
    
    This data is used to:
    - Track topic mastery over time
    - Identify strengths and weaknesses
    - Personalize future tutoring responses
    """
    try:
        from learning_profile import get_learning_profile
        profile = get_learning_profile(user_id)
        profile.record_interaction(record.topic, record.was_correct)
        return {"status": "success", "profile": profile.data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    import sys
    import asyncio

    # Force SelectorEventLoop on Windows for psycopg compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    config = uvicorn.Config(app, host="0.0.0.0", port=8080)
    
    # Monkeypatch config.setup_event_loop (not server.setup_event_loop)
    original_setup = config.setup_event_loop
    def _setup_event_loop_override():
        if sys.platform == "win32":
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        else:
             original_setup()
             
    config.setup_event_loop = _setup_event_loop_override
    
    server = uvicorn.Server(config)
    server.run()


# --- Health and Readiness Probes ---
@app.get("/health", tags=["System"], summary="Health check")
async def health():
    """
    Basic health check endpoint.
    
    Returns configuration status for:
    - LangSmith tracing
    - Database connection
    - Groq API
    - TTS feature
    """
    return {
        "status": "ok",
        "langsmith_tracing": bool(os.getenv("LANGCHAIN_API_KEY")),
        "db_uri_configured": bool(os.getenv("DB_URI")),
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "enable_tts": os.getenv("ENABLE_TTS", "0") == "1",
    }

@app.get("/ready", tags=["System"], summary="Readiness probe")
async def ready():
    """
    Kubernetes-style readiness probe.
    
    Checks database connectivity to determine if the server
    is ready to accept traffic.
    """
    # Check DB connectivity
    pool = getattr(app.state, "pool", None)
    db_ok = False
    try:
        if pool is not None:
            async with pool.connection() as conn:
                cursor = await conn.execute("SELECT 1")
                _ = await cursor.fetchone()
                db_ok = True
    except Exception as e:
        db_ok = False
    return {
        "ready": db_ok,
        "db": "ok" if db_ok else "error",
    }
