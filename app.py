# ============================================================================
# Streamlit GUI for PDF Chatbot - Offline RAG System
# Uses chatbot.py backend with Ollama for offline model support
# ============================================================================

import streamlit as st
import os
import sys
import time
import threading
from pathlib import Path

# Suppress chatbot prints when loading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="PDF Chatbot - Offline RAG",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'backend_ready' not in st.session_state:
    st.session_state.backend_ready = False
if 'chatbot_module' not in st.session_state:
    st.session_state.chatbot_module = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'kb_mode' not in st.session_state:
    st.session_state.kb_mode = "full"
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Auto"
if 'detailed_mode' not in st.session_state:
    st.session_state.detailed_mode = False
if 'target_file' not in st.session_state:
    st.session_state.target_file = "All files"
if 'available_files' not in st.session_state:
    st.session_state.available_files = ["All files"]

# Load chatbot module
@st.cache_resource
def load_chatbot():
    """Load the chatbot module"""
    try:
        # Suppress prints during import
        import chatbot
        chatbot.SUPPRESS_PRINTS = True
        return chatbot
    except Exception as e:
        st.error(f"Error loading chatbot module: {e}")
        return None

# Load chatbot
if st.session_state.chatbot_module is None:
    with st.spinner("Initializing backend (loading knowledge base and models)..."):
        chatbot = load_chatbot()
        if chatbot:
            st.session_state.chatbot_module = chatbot
            chatbot.SUPPRESS_PRINTS = True
            st.session_state.backend_ready = True
            st.session_state.kb_mode = chatbot.KB_MODE
            # Get available files
            try:
                files = sorted(set(c.get("source_file", "") for c in chatbot.chunks if c.get("source_file")))
                st.session_state.available_files = ["All files"] + files
            except:
                pass

chatbot = st.session_state.chatbot_module

# Sidebar
with st.sidebar:
    st.header("📚 Knowledge Base & Options")
    
    if not st.session_state.backend_ready:
        st.warning("Backend is initializing...")
    else:
        st.success("✅ Backend Ready")
    
    st.markdown("---")
    
    # KB Mode
    st.subheader("KB Mode")
    kb_mode = st.radio(
        "Knowledge Base Mode:",
        ["Base (original docs only)", "Full (with uploaded docs)"],
        index=0 if st.session_state.kb_mode == "base" else 1,
        key="kb_mode_radio"
    )
    if chatbot:
        chatbot.KB_MODE = "base" if kb_mode.startswith("Base") else "full"
        st.session_state.kb_mode = chatbot.KB_MODE
    
    st.markdown("---")
    
    # Model Selection
    st.subheader("Model Settings")
    if chatbot and chatbot.check_ollama():
        available_models = chatbot.get_available_models()
        model_options = ["Auto"] + [m for m in available_models if m in chatbot.MODEL_REGISTRY and chatbot.MODEL_REGISTRY[m]["min_ram_gb"] < 100]
        if not model_options:
            model_options = ["Auto"] + available_models[:5]  # Fallback
        
        selected_model = st.selectbox(
            "Select Model:",
            model_options,
            index=0 if st.session_state.selected_model == "Auto" else (model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0)
        )
        st.session_state.selected_model = selected_model
        
        if selected_model != "Auto" and selected_model in chatbot.MODEL_REGISTRY:
            cfg = chatbot.MODEL_REGISTRY[selected_model]
            st.caption(f"Quality: {cfg['quality']} | Speed: {cfg['speed']}")
    else:
        st.warning("⚠️ Ollama not detected")
        st.info("Install Ollama from https://ollama.ai and pull a model (e.g., `ollama pull llama3.2:1b`)")
        selected_model = "Auto"
    
    # Detailed answer
    detailed_mode = st.checkbox("Detailed answer", value=st.session_state.detailed_mode)
    st.session_state.detailed_mode = detailed_mode
    
    st.markdown("---")
    
    # File Management
    st.subheader("File Management")
    
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "txt", "docx", "xlsx", "xls"],
        help="Upload PDF, TXT, DOCX, or Excel files"
    )
    
    if uploaded_file is not None:
        if st.button("📄 Process Uploaded File"):
            with st.spinner("Processing file..."):
                # Save uploaded file temporarily
                temp_path = os.path.join(os.path.dirname(__file__), "temp_upload", uploaded_file.name)
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    result = chatbot.process_file(temp_path, persist=True)
                    if result:
                        st.success(f"✅ Added {len(result)} chunks from {uploaded_file.name}")
                        # Refresh file list
                        files = sorted(set(c.get("source_file", "") for c in chatbot.chunks if c.get("source_file")))
                        st.session_state.available_files = ["All files"] + files
                        st.rerun()
                    else:
                        st.error("Failed to process file")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    # Load databank folder
    st.markdown("---")
    databank_path = st.text_input("Load Databank Folder:", placeholder="Enter folder path")
    if st.button("📁 Load Databank"):
        if databank_path and os.path.exists(databank_path):
            with st.spinner("Loading databank (this may take a while)..."):
                try:
                    chatbot.load_databank(databank_path, persist=True)
                    st.success("✅ Databank loaded")
                    # Refresh file list
                    files = sorted(set(c.get("source_file", "") for c in chatbot.chunks if c.get("source_file")))
                    st.session_state.available_files = ["All files"] + files
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.error("Invalid folder path")
    
    st.markdown("---")
    
    # Target file selector
    st.subheader("Query Options")
    if st.session_state.available_files:
        target_file = st.selectbox(
            "Target File (optional):",
            st.session_state.available_files,
            index=0 if st.session_state.target_file == "All files" else (st.session_state.available_files.index(st.session_state.target_file) if st.session_state.target_file in st.session_state.available_files else 0)
        )
        st.session_state.target_file = target_file
    
    # Refresh file list
    if st.button("🔄 Refresh File List"):
        if chatbot:
            files = sorted(set(c.get("source_file", "") for c in chatbot.chunks if c.get("source_file")))
            st.session_state.available_files = ["All files"] + files
            st.success("File list refreshed")
            st.rerun()
    
    st.markdown("---")
    
    # Status
    st.subheader("Status")
    if chatbot:
        try:
            files_count = len(set(c.get("source_file", "") for c in chatbot.chunks if c.get("source_file")))
            chunks_count = len(chatbot.chunks)
            st.metric("Files Indexed", files_count)
            st.metric("Total Chunks", f"{chunks_count:,}")
            st.metric("KB Mode", chatbot.KB_MODE.upper())
            
            ollama_status = "🟢 Online" if chatbot.check_ollama() else "🔴 Offline"
            st.metric("Ollama", ollama_status)
        except:
            pass

# Main chat interface
st.title("📚 PDF Chatbot - Offline RAG System")
st.markdown("Ask questions about your documents. Everything runs locally with Ollama!")

if not st.session_state.backend_ready:
    st.info("⏳ Initializing backend... Please wait.")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get parameters
                target = st.session_state.target_file
                target_file = None if target == "All files" else target
                
                model_sel = st.session_state.selected_model
                force_model = None if model_sel == "Auto" else model_sel
                
                detailed = st.session_state.detailed_mode
                
                # Query RAG
                answer = chatbot.query_rag(
                    prompt,
                    detailed=detailed,
                    target_file=target_file,
                    force_model=force_model
                )
                
                if answer:
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = "I couldn't find an answer. Please try rephrasing your question or check if Ollama is running."
                    st.warning(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")
st.caption("💡 **Tip**: Make sure Ollama is running and you have models installed (e.g., `ollama pull llama3.2:1b`)")
