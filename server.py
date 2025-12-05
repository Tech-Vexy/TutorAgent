import os
from dotenv import load_dotenv
import logging

load_dotenv()

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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from groq import Groq

from agent_graph import graph
from database import get_or_create_profile, init_db, upsert_thread_metadata

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
USE_GROQ_SYSTEM = os.getenv("USE_GROQ_SYSTEM", "1") == "1"
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

app = FastAPI(lifespan=lifespan)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            model_preference = payload.get("model_preference", "fast") # "fast" or "smart"
            
            logger.info(f"Received message from user={user_id}, thread={thread_id}, model={model_preference}")
            
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

@app.get("/")
async def root():
    return {"message": "TopScore AI Backend is Running"}

@app.get("/threads/{user_id}")
async def get_user_threads(user_id: str):
    pool = app.state.pool
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT thread_id, title, updated_at FROM user_threads WHERE user_id = %s ORDER BY updated_at DESC",
            (user_id,)
        )
        rows = await cursor.fetchall()
        return [{"thread_id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]

@app.get("/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str):
    pool = app.state.pool
    checkpointer = AsyncPostgresSaver(pool)
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = await checkpointer.get(config)
    
    if not checkpoint:
        return []
    
    # Extract messages from checkpoint
    # The state structure depends on agent_graph.py
    # Usually state['messages']
    messages = checkpoint.get("channel_values", {}).get("messages", [])
    
    # Format for client
    formatted_messages = []
    for msg in messages:
        msg_type = msg.type
        content = msg.content
        # Handle list content (multimodal)
        if isinstance(content, list):
            text_parts = [c["text"] for c in content if c["type"] == "text"]
            content = " ".join(text_parts)
            
        formatted_messages.append({
            "type": "user" if msg_type == "human" else "ai",
            "content": content,
            # "timestamp": ... (LangChain messages don't always have timestamps unless added)
        })
        
    return formatted_messages

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


# --- Health and Readiness Probes ---
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "langsmith_tracing": bool(os.getenv("LANGCHAIN_API_KEY")),
        "db_uri_configured": bool(os.getenv("DB_URI")),
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "enable_tts": os.getenv("ENABLE_TTS", "0") == "1",
    }

@app.get("/ready")
async def ready():
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
