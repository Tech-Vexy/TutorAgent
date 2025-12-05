import streamlit as st
import requests
import json
import pandas as pd
import base64
from io import BytesIO
from PIL import Image

# Configuration
API_URL = "http://localhost:8080"
st.set_page_config(page_title="Deep Reasoning AI Tutor", layout="wide")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Sidebar - Skills & Status
with st.sidebar:
    st.title("🧠 Knowledge Base")
    
    # Connection Check
    try:
        health = requests.get(f"{API_URL}/health").json()
        st.success(f"System Status: {health['status']}")
    except:
        st.error("Backend Offline")
    
    st.divider()
    
    # Skills List
    st.subheader("Acquired Skills")
    if st.button("Refresh Skills"):
        try:
            response = requests.get(f"{API_URL}/skills")
            if response.status_code == 200:
                skills = response.json()
                if skills:
                    for skill in skills:
                        with st.expander(f"📚 {skill.get('name', 'Unnamed Skill')}"):
                            st.write(f"**Description:** {skill.get('description')}")
                            if skill.get('code'):
                                st.code(skill.get('code'), language='python')
                else:
                    st.info("No skills learned yet.")
        except Exception as e:
            st.error(f"Error fetching skills: {e}")

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

# Main Chat Interface
st.title("🎓 Deep Reasoning AI Tutor")
st.caption("Dual-Model Architecture: Fast Router + Deep Reasoner with Skill Acquisition")

# Display Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        
        # Check for images (plots) in the content
        # The backend returns plots as "[Generated Plot: data:image/png;base64,...]"
        if "[Generated Plot: data:image/png;base64," in content:
            parts = content.split("[Generated Plot: data:image/png;base64,")
            text_part = parts[0]
            img_part = parts[1].split("]")[0]
            
            st.markdown(text_part)
            try:
                image_data = base64.b64decode(img_part)
                st.image(image_data, caption="Generated Visualization")
            except:
                st.error("Failed to render plot")
        else:
            st.markdown(content)

# Chat Input
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Prepare request
            payload = {"message": prompt}
            if st.session_state.session_id:
                payload["user_id"] = st.session_state.session_id
            
            # Stream response
            with requests.post(f"{API_URL}/chat", json=payload, stream=True) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith("data: "):
                                data_str = line[6:]
                                try:
                                    data = json.loads(data_str)
                                    
                                    if data["type"] == "message":
                                        # Accumulate response
                                        # Note: The backend sends full messages or chunks? 
                                        # Based on server.py: yield f"data: {json.dumps(response_data)}\n\n"
                                        # And deep_graph.py: response = self.slow_llm.invoke(messages) -> returns full content usually
                                        # But we are using .astream on the graph.
                                        # Graph .astream yields updates to state.
                                        # If the node returns a full message, we get a full message.
                                        # If we want token streaming, we need to stream the LLM call inside the node.
                                        # For now, let's assume we get chunks or full messages and just append/replace.
                                        
                                        # Actually, looking at server.py:
                                        # latest_message = chunk["messages"][-1]
                                        # This is likely the FULL message from the node, not a token.
                                        # So we will receive the full message at once when a node finishes.
                                        
                                        new_content = data["content"]
                                        # If it's a new message (different from what we have), append it.
                                        # But since we are streaming the graph execution, we might get intermediate messages?
                                        # Let's just show the latest content.
                                        
                                        # Simple logic: just show what we get.
                                        # If we get multiple messages (e.g. from tool use then final answer), we might want to handle that.
                                        # For now, let's just overwrite the placeholder with the latest content.
                                        full_response = new_content
                                        message_placeholder.markdown(full_response + "▌")
                                        
                                        # Update session ID if provided
                                        if "session_id" in data:
                                            st.session_state.session_id = data["session_id"]
                                            
                                    elif data["type"] == "error":
                                        st.error(f"Error: {data['error']}")
                                        
                                except json.JSONDecodeError:
                                    pass
                    
                    message_placeholder.markdown(full_response)
                    
                    # Check for images in the final response
                    if "[Generated Plot: data:image/png;base64," in full_response:
                        parts = full_response.split("[Generated Plot: data:image/png;base64,")
                        text_part = parts[0]
                        img_part = parts[1].split("]")[0]
                        message_placeholder.markdown(text_part)
                        try:
                            image_data = base64.b64decode(img_part)
                            st.image(image_data, caption="Generated Visualization")
                        except:
                            pass

                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"Server Error: {response.status_code}")
        except Exception as e:
            st.error(f"Connection Error: {e}")
