import os
import streamlit as st
import requests
import json
import base64
import pandas as pd

# Configuration
API_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
st.set_page_config(page_title="Deep Reasoning AI Control Center", layout="wide", page_icon="🧠")

# Session State & Styling
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain--v1.png", width=80)
    st.title("TopScore Control")
    
    st.markdown(f"**User ID:** `{st.session_state.user_id}`")
    new_user = st.text_input("Change User ID", value=st.session_state.user_id)
    if new_user != st.session_state.user_id:
        st.session_state.user_id = new_user
        st.experimental_rerun()
    
    # Status
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"Backend: {health['status']}")
        if health.get("langsmith_tracing"):
            st.caption("✅ Tracing Active")
    except:
        st.error("Backend Offline")

# Tabs
tab_chat, tab_skills, tab_knowledge, tab_memory, tab_config = st.tabs(["💬 Chat", "📚 Skills", "📖 Knowledge", "🧠 Memory", "⚙️ Config"])

# --- CHAT TAB ---
with tab_chat:
    st.subheader("Interactive Reasoning")
    
    # Model Selection
    model_pref = st.radio("Agent Mode", ["fast", "smart"], horizontal=True, format_func=lambda x: x.upper())
    
    # Display Chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            # Check for images
            if "[IMAGE_GENERATED_BASE64_DATA:" in content:
                parts = content.split("[IMAGE_GENERATED_BASE64_DATA:")
                st.markdown(parts[0])
                try:
                    img_b64 = parts[1].split("]")[0].strip()
                    st.image(base64.b64decode(img_b64))
                except:
                    st.error("Image render failed")
            else:
                st.markdown(content)
    
    # Input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            
            try:
                # Use new POST streaming endpoint
                payload = {
                    "message": prompt,
                    "user_id": st.session_state.user_id,
                    "model_preference": model_pref,
                    "stream": True
                }
                
                with requests.post(f"{API_URL}/chat", json=payload, stream=True) as response:
                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith("data: "):
                                    data_str = line[6:]
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data = json.loads(data_str)
                                        if data["type"] == "message":
                                            chunk = data["content"]
                                            full_response += chunk
                                            placeholder.markdown(full_response + "▌")
                                        elif data["type"] == "token":
                                            # Tool outputs or full tokens
                                            if "[IMAGE_GENERATED_BASE64_DATA:" in data["content"]:
                                                 full_response += "\n" + data["content"]
                                    except:
                                        pass
                                        
                        placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

# --- SKILLS TAB ---
with tab_skills:
    st.subheader("Agent Capabilities")
    if st.button("Refresh Skills Registry"):
        try:
            resp = requests.get(f"{API_URL}/skills")
            if resp.status_code == 200:
                skills = resp.json()
                if skills:
                    for s in skills:
                        with st.expander(f"🧩 {s.get('name')}"):
                            st.write(s.get('description'))
                            st.code(s.get('code'), language='python')
                else:
                    st.info("No learned skills yet.")
            else:
                st.warning("Skills registry endpoint not available (server might not support it).")
        except Exception as e:
             st.warning(f"Could not fetch skills: {e}")

# --- KNOWLEDGE TAB ---
with tab_knowledge:
    st.subheader("Ingest Knowledge (RAG)")
    
    st.markdown("Upload documents, provide URLs, or connect Google Drive to expand knowledge.")
    
    col_up, col_url, col_drive = st.columns(3)
    
    with col_up:
        st.markdown("#### 📄 File Upload")
        uploaded_file = st.file_uploader("Upload PDF or Text", type=["pdf", "txt", "md"])
        if uploaded_file and st.button("Ingest File"):
            with st.spinner("Uploading and processing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    r = requests.post(f"{API_URL}/knowledge/upload", files=files)
                    if r.status_code == 200:
                        st.success(f"Response: {r.json().get('message')}")
                    else:
                        st.error(f"Error: {r.text}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    
    with col_url:
        st.markdown("#### 🌐 URL Ingestion")
        url_input = st.text_input("Enter URL to scrape")
        if url_input and st.button("Ingest URL"):
            with st.spinner("Scraping..."):
                try:
                    r = requests.post(f"{API_URL}/knowledge/url", json={"url": url_input})
                    if r.status_code == 200:
                        st.success(f"Response: {r.json().get('message')}")
                    else:
                        st.error(f"Error: {r.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

    with col_drive:
        st.markdown("#### ☁️ Google Drive")
        st.info("Requires `drive_credentials.json` in root.")
        if st.button("List Drive Files"):
            try:
                r = requests.get(f"{API_URL}/knowledge/drive/list")
                data = r.json()
                if data.get("status") == "success":
                    st.session_state["drive_files"] = data.get("files", [])
                else:
                    st.error(data.get("message"))
            except Exception as e:
                st.error(f"Connection error: {e}")
                
        if "drive_files" in st.session_state and st.session_state["drive_files"]:
            drive_file = st.selectbox("Select File", st.session_state["drive_files"], format_func=lambda x: x['name'])
            if st.button("Ingest from Drive"):
                 with st.spinner("Downloading from Drive..."):
                    try:
                        r = requests.post(f"{API_URL}/knowledge/drive/ingest", json={"file_id": drive_file['id'], "file_name": drive_file['name']})
                        if r.status_code == 200:
                            st.success(f"Response: {r.json().get('message')}")
                        else:
                            st.error(f"Error: {r.text}")
                    except Exception as e:
                        st.error(f"Ingest failed: {e}")
                    
    st.info("Ingested content is automatically indexed and retrieved during complex reasoning tasks.")

# --- MEMORY TAB ---
with tab_memory:
    st.subheader("🧠 Memory & Learning Profile")
    
    col_mem, col_profile = st.columns(2)
    
    with col_mem:
        st.markdown("#### 📚 Episodic Memory")
        if st.button("Fetch Memory"):
            try:
                mem = requests.get(f"{API_URL}/memory/{st.session_state.user_id}").json()
                st.json(mem)
            except Exception as e:
                st.error(f"Failed to fetch memory: {e}")
    
    with col_profile:
        st.markdown("#### 📊 Learning Profile")
        if st.button("Load Profile"):
            try:
                r = requests.get(f"{API_URL}/profile/{st.session_state.user_id}")
                data = r.json()
                if data.get("status") == "success":
                    profile = data.get("profile", {})
                    st.session_state["learning_profile"] = profile
                else:
                    st.error(data.get("message"))
            except Exception as e:
                st.error(f"Failed to load profile: {e}")
        
        if "learning_profile" in st.session_state:
            profile = st.session_state["learning_profile"]
            
            st.markdown(f"**Total Questions:** {profile.get('total_questions', 0)}")
            
            if profile.get("strengths"):
                st.success(f"💪 Strengths: {', '.join(profile['strengths'])}")
            if profile.get("weaknesses"):
                st.warning(f"📈 Areas to Improve: {', '.join(profile['weaknesses'])}")
            
            st.divider()
            st.markdown("**Preferences:**")
            
            style = st.selectbox("Learning Style", ["balanced", "visual", "textual", "interactive"], 
                                 index=["balanced", "visual", "textual", "interactive"].index(profile.get("preferred_style", "balanced")))
            difficulty = st.selectbox("Difficulty Level", ["easy", "medium", "hard"],
                                      index=["easy", "medium", "hard"].index(profile.get("difficulty_level", "medium")))
            
            if st.button("Save Preferences"):
                try:
                    requests.post(f"{API_URL}/profile/{st.session_state.user_id}/preference", json={"key": "preferred_style", "value": style})
                    requests.post(f"{API_URL}/profile/{st.session_state.user_id}/preference", json={"key": "difficulty_level", "value": difficulty})
                    st.success("Preferences saved!")
                except Exception as e:
                    st.error(f"Failed to save: {e}")

# --- CONFIG TAB ---
with tab_config:
    st.header("⚙️ Configuration")
    
    st.subheader("🤖 Model Manager")
    
    # helper for updates
    def update_model_config(m_type, m_id, save_default):
        try:
            r = requests.post(f"{API_URL}/models/update", json={
                "type": m_type, 
                "model_id": m_id,
                "persist": save_default
            })
            if r.status_code == 200:
                st.toast(f"✅ {m_type.upper()} model updated!")
                if save_default:
                     st.toast("💾 Saved to .env!")
                return True
            else:
                st.error(f"Update failed: {r.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")
        return False
    
    try:
        # Fetch current config
        info = requests.get(f"{API_URL}/info", timeout=2).json()
        current_models = info.get("models", {})
        
        # Persistence Option
        save_default = st.checkbox("💾 Save changes as default (updates .env)", value=False)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### ⚡ Fast Model")
            curr_fast = current_models.get("fast_model", "")
            new_fast = st.text_input("Model ID", value=curr_fast, key="fast_in")
            if new_fast != curr_fast and st.button("Update Fast Model"):
                update_model_config("fast", new_fast, save_default)
                
        with col2:
            st.markdown("#### 🧠 Smart Model")
            curr_smart = current_models.get("smart_model", "")
            new_smart = st.text_input("Model ID", value=curr_smart, key="smart_in")
            if new_smart != curr_smart and st.button("Update Smart Model"):
                update_model_config("smart", new_smart, save_default)

        with col3:
            st.markdown("#### 👁️ Vision Model")
            curr_vis = current_models.get("vision_model", "")
            new_vis = st.text_input("Model ID", value=curr_vis, key="vis_in")
            if new_vis != curr_vis and st.button("Update Vision Model"):
                update_model_config("vision", new_vis, save_default)
                
        st.info("💡 Note: Updating models applies immediately to new requests.")
        
    except Exception as e:
        st.warning(f"Could not load model configuration: {e}")
    
    st.divider()
    st.write("Edit `.env` to change other detailed settings.")
