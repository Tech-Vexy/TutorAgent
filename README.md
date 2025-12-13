# TopScore AI - Kenyan Agentic Tutor

TopScore AI is an advanced, multimodal AI tutor designed specifically for the Kenyan education context (KCSE/CBC). It acts as a patient, caring, and knowledgeable "Senior Teacher," using local analogies (e.g., M-Pesa, Matatus, Farming) to explain complex concepts.

## 🚀 Features

*   **Kenyan Persona:** Uses local context and analogies to make learning relatable. Adopts a patient, Socratic teaching style.
*   **Multimodal Capabilities:**
    *   **Text:** Deep reasoning and simple chat.
    *   **Vision:** Analyzes images (homework, diagrams) using Llama 3.2 Vision.
    *   **Voice:** Speech-to-Text (Whisper) for natural interaction.
*   **Intelligent Routing:** Automatically routes queries to the best model (Vision, Complex Reasoning, or Simple Chat).
*   **Model Switching:** Users can toggle between "Smart" (Llama 3.3 70B), "Fast" (Llama 3.1 8B), Mixtral, and Gemma models.
*   **Tools:** Generates educational plots and KCSE-style quizzes.
*   **Persistence:** Saves conversation history and state using PostgreSQL (Supabase/Neon).
*   **Real-time Streaming:** WebSocket-based architecture for low-latency responses.

## 🛠️ Tech Stack

*   **Framework:** FastAPI (Python)
*   **Orchestration:** LangGraph & LangChain
*   **Inference Engine:** [Groq](https://groq.com/) (LPU Inference)
*   **Models:**
    *   *Reasoning:* Llama 3.3 70B Versatile, Llama 3.1 8B Instant, Mixtral 8x7B, Gemma 2 9B
    *   *Vision:* Llama 3.2 11B Vision Preview
    *   *Audio:* Whisper Large V3 (STT)
*   **Database:** PostgreSQL (via `langgraph-checkpoint-postgres`)
*   **Frontend:** HTML/JS WebSocket Client (for testing)

## 📋 Prerequisites

*   Python 3.10+
*   A Groq API Key
*   A PostgreSQL Database (Supabase, Neon, or local)

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd TutorAgent
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory:
    ```dotenv
    # Server Configuration
    HOST=0.0.0.0
    PORT=8080

    # TopScore AI Configuration
    GROQ_API_KEY=your_groq_api_key_here
    # Connection string for Supabase/Neon (ensure ?sslmode=require is appended if needed)
    DB_URI=postgresql://user:password@host:5432/dbname?sslmode=require
    ```

## 🏃‍♂️ Usage

1.  **Start the Backend Server:**
    ```bash
    python server.py
    ```
    The server will start on `http://0.0.0.0:8080`.

2.  **Open a Client:**
    
    **Option A: Streamlit Dashboard (Recommended)**
    *   Run: `run_dashboard.bat` (Windows)
    *   Provides: Chat, History, Settings, Status
    
    **Option B: Web Client**
    *   Open: `chat_persistent.html`
    *   Provides: Reliable chat with auto-reconnect

    **Option C: Terminal**
    *   Run: `python test_client.py`

3.  **Interact:**
    *   Type messages to chat.
    *   Upload images for analysis.
    *   Use the "Record" button (Web Client) to speak to the tutor.
    *   Select different models using the dropdown.

## 📂 Project Structure

*   `server.py`: FastAPI application, WebSocket endpoints, and database connection management.
*   `agent_graph.py`: Defines the LangGraph workflow, nodes (Router, Vision, Deep Thinker), and agent state.
*   `tools.py`: Custom tools for the agent (Plotting, Quiz Generation).
*   `test_client.html`: Simple web interface for testing text, image, and audio features.
*   `requirements.txt`: Python dependencies.

## 🧠 Agent Logic

The agent uses a **StateGraph** to manage the conversation flow:
1.  **Router Node:** Analyzes the user's input (text or image) to determine intent.
2.  **Vision Node:** If an image is present, Llama 3.2 Vision analyzes it.
3.  **Deep Thinker Node:** The core reasoning engine (Llama 3.3/Mixtral) processes the query, using tools if necessary, and formulates a response using the Kenyan Teacher persona.
4.  **Tools Node:** Executes Python code for plots or quizzes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Performance tuning (latency)
To reduce time-to-first-token and overall response latency, this build includes several toggles:

- Default model preference: now defaults to the fast model for new chats if the client does not specify
  - Server default: `model_preference = "fast"`
  - Client can still explicitly choose `"smart"` when needed

- Disable LangSmith tracing to remove network overhead (optional)
  - Set `DISABLE_TRACING=1` to skip enabling LangSmith tracing even if `LANGCHAIN_API_KEY` is present

- Limit fast model output length
  - `FAST_TOKENS` (default `512`) caps tokens for the fast model to encourage quicker, concise answers
  - Override via `.env`, e.g. `FAST_TOKENS=256`

- Model selection via environment
  - `FAST_MODEL` (default `llama-3.1-8b-instant`)
  - `SMART_MODEL` (default `llama-3.3-70b-versatile`)

Operational tips:
- For impatient users, prefer `fast` for general queries and switch to `smart` only for complex reasoning or tool-heavy tasks.
- If you see latency spikes and you are not actively using LangSmith, set `DISABLE_TRACING=1`.

## Groq Compound Systems (default)
Groq's Compound Systems are now the default path for text-only chats. They integrate GPT-OSS 120B, Llama 4, and external tools (e.g., web search, code execution) via a unified interface managed in the Groq Console.

Setup:
1. Create a System in the Groq Console and note its System ID.
2. Add the following to your `.env`:
   - `GROQ_SYSTEM_ID=<your_system_id>`
   - Optional: `GROQ_SYSTEM_VERSION=<version_id>` to pin a specific version
   - Optional: set `USE_GROQ_SYSTEM=0` to disable and use the built-in LangGraph agent by default instead
3. Ensure `GROQ_API_KEY` is set.

Behavior:
- By default (`USE_GROQ_SYSTEM=1`), the server will try to stream a response from your Groq System when the incoming message is text-only.
- If an image is attached, or if the Systems API is unavailable/throws an error, the server gracefully falls back to the built-in LangGraph agent pipeline.
- Streaming is preserved: tokens are forwarded over the existing WebSocket protocol with `type: "token"`.

Limitations (current):
- Systems path handles text messages. Image analysis and Python tools continue to run via the existing LangGraph flow.
- If you need tools within Groq Systems, configure them inside your System in the Console. The server does not yet forward tool calls from Systems back to the client.

## OpenTelemetry & LangSmith Tracing

TopScore AI includes **automatic OpenTelemetry tracing** via LangSmith's native OTEL integration. This provides deep visibility into your agent's behavior, LLM calls, tool usage, and performance metrics.

### 🎯 What Gets Traced

The `langsmith[otel]` package automatically instruments:
- ✅ **LangChain chains and agents** - Full execution flows
- ✅ **LLM calls** - Groq, OpenAI, Google Gemini, etc.
- ✅ **Tools** - Plot generation, quiz creation, web search, etc.
- ✅ **Prompts** - Input prompts and responses
- ✅ **Token usage** - Track costs and performance
- ✅ **Latency** - Measure response times
- ✅ **Errors** - Automatic error capture with stack traces

### 📋 Setup Instructions

1. **Get a LangSmith API key**  
   Sign up at [smith.langchain.com](https://smith.langchain.com/) and create an API key

2. **Configure environment variables** in your `.env` file:
   ```bash
   # Enable OpenTelemetry tracing
   LANGSMITH_OTEL_ENABLED=true
   
   # Enable LangSmith tracing (required)
   LANGSMITH_TRACING=true
   
   # Your LangSmith API key
   LANGSMITH_API_KEY=your_api_key_here
   
   # Optional: Project name for organizing traces
   LANGSMITH_PROJECT=TopScore-AI
   
   # Optional: LangSmith endpoint (default shown)
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   ```

3. **Start the server**
   ```bash
   python server.py
   ```
   
   You'll see a confirmation message:
   ```
   ============================================================
   OpenTelemetry Configuration (LangSmith Native)
   ============================================================
   OTEL Enabled: True
   Tracing Enabled: True
   Endpoint: https://api.smith.langchain.com
   Project: TopScore-AI
   
   📊 Automatic instrumentation active for:
     ✓ LangChain chains and agents
     ✓ LLM calls (Groq, OpenAI, Google, etc.)
     ✓ Tools and function calls
     ✓ Retrievers and vector stores
     ✓ Prompts and prompt templates
   
   🔍 View traces at: https://smith.langchain.com/
   ============================================================
   ```

4. **View traces**
   - Go to [smith.langchain.com](https://smith.langchain.com/)
   - Select your project (e.g., "TopScore-AI")
   - Browse traces to see detailed execution flows
   - Click on any trace to see:
     - Full conversation history
     - LLM prompts and responses
     - Tool calls and results
     - Token usage and costs
     - Latency breakdown
     - Error details (if any)

### 🔧 Advanced Configuration

**Disable tracing entirely:**
```bash
DISABLE_TRACING=1
```

**Multiple workspaces:**
If your API key is linked to multiple workspaces, specify which one:
```bash
LANGSMITH_WORKSPACE_ID=your_workspace_id
```

**Performance impact:**
- Tracing adds minimal overhead (<50ms per request)
- All tracing is async and non-blocking
- Failed trace uploads don't affect app functionality
- To reduce latency further, disable tracing for production

### 📊 Example Trace View

When you chat with the tutor, each interaction creates a trace showing:

```
User Message: "Draw a graph of y = x^2"
  ├─ Router Node (50ms)
  │   └─ Intent: COMPLEX_REASONING
  ├─ Deep Thinker Node (1.2s)
  │   ├─ LLM Call: llama-3.3-70b (800ms)
  │   │   ├─ Prompt: [DEEP_THINKER_BASE_PROMPT + user message]
  │   │   ├─ Response: [Explanation + tool call]
  │   │   └─ Tokens: 450 input, 120 output
  │   └─ Tool: generate_educational_plot (400ms)
  │       ├─ Input: Python code for plotting
  │       └─ Output: Base64 image
  └─ Response: Explanation + plot image
```

### 🆚 Legacy vs New Tracing

**Old way (deprecated):**
- Manual OpenTelemetry setup
- Separate instrumentation
- Complex configuration

**New way (current):**
- Automatic with `langsmith[otel]`
- Zero configuration needed
- Just set environment variables

### 📚 Learn More

- [LangSmith OpenTelemetry Docs](https://docs.langchain.com/langsmith/trace-with-opentelemetry)
- [OpenTelemetry Concepts](https://opentelemetry.io/docs/concepts/)
- [LangSmith Dashboard](https://smith.langchain.com/)

## Response caching (in-memory)
To speed up repeated questions, the server caches recent responses and replays them instantly on identical prompts.

- Enabled by default.
- Works for both plain text answers and tool outputs (e.g., generated plots sent as base64 images).
- Cache key: normalized text of the user message; if the user attached an image, the key also includes a hash of that image.

Environment toggles:
- `CACHE_ENABLED` (default `1`) — set to `0` to disable caching.
- `CACHE_TTL_SECONDS` (default `3600`) — how long an entry is valid.
- `CACHE_MAX_ENTRIES` (default `256`) — maximum number of entries kept (LRU eviction).

Notes:
- A cache hit replays previously streamed events (tokens/images) and ends the turn immediately.
- If text differs slightly (punctuation/spacing/case are ignored), it will still match; semantic variations are not matched.
