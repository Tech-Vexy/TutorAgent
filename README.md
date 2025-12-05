# TopScore AI - Kenyan Agentic Tutor

TopScore AI is an advanced, multimodal AI tutor designed specifically for the Kenyan education context (KCSE/CBC). It acts as a patient, caring, and knowledgeable "Senior Teacher," using local analogies (e.g., M-Pesa, Matatus, Farming) to explain complex concepts.

## 🚀 Features

*   **Kenyan Persona:** Uses local context and analogies to make learning relatable. Adopts a patient, Socratic teaching style.
*   **Multimodal Capabilities:**
    *   **Text:** Deep reasoning and simple chat.
    *   **Vision:** Analyzes images (homework, diagrams) using Llama 3.2 Vision.
    *   **Voice:** Speech-to-Text (Whisper) and Text-to-Speech (PlayAI) for natural interaction.
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
    *   *Audio:* Whisper Large V3 (STT), PlayAI-TTS (TTS)
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

2.  **Run the Test Client:**
    *   **Web Client:** Open `test_client.html` in your web browser.
    *   **Terminal Client:** Run `python test_client.py`.

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
