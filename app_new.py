"""
Streamlit UI for PDF QA System with LSTM
Query-only interface - Training should be done using train_model.py
"""

import streamlit as st
import os
from pathlib import Path
import time

# Import our QA system
from qa_system import QASystem

# Page configuration
st.set_page_config(
    page_title="PDF QA System - Query Interface",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Navy Blue Theme CSS
st.markdown("""
<style>
    /* Main navy blue background */
    .stApp {
        background-color: #001f3f !important;
        color: #fafafa !important;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #001f3f !important;
        padding-top: 2rem;
    }
    
    /* Remove white backgrounds from top and bottom */
    header,
    footer,
    .stHeader,
    .stFooter,
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        background-color: #001f3f !important;
    }
    
    /* Top area */
    .stApp > header {
        background-color: #001f3f !important;
    }
    
    /* Bottom area */
    .stApp > footer {
        background-color: #001f3f !important;
    }
    
    /* Any white backgrounds in main area */
    .main > div,
    .main > div > div,
    .main > div > div > div {
        background-color: #001f3f !important;
    }
    
    /* Remove white from all elements except chat input */
    div:not([data-testid="stChatInput"]):not([data-testid="stChatInput"] > div):not([data-testid="stChatInput"] input),
    section:not([data-testid="stChatInput"]),
    article:not([data-testid="stChatInput"]) {
        background-color: #001f3f !important;
    }
    
    /* Exception: Only chat input box inner elements should be white */
    [data-testid="stChatInput"] > div {
        background-color: white !important;
    }
    
    [data-testid="stChatInput"] input {
        background-color: white !important;
    }
    
    /* Sidebar navy blue */
    [data-testid="stSidebar"] {
        background-color: #0a1929 !important;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background-color: #0a1929 !important;
    }
    
    /* Chat messages - curved boxes with blue backgrounds */
    [data-testid="stChatMessage"] {
        background-color: #0d47a1 !important;
        padding: 1.2rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stChatMessage"][data-testid="user"] {
        background-color: #1565c0 !important;
        border-radius: 20px;
    }
    
    [data-testid="stChatMessage"][data-testid="assistant"] {
        background-color: #0d47a1 !important;
        border-radius: 20px;
    }
    
    /* Ensure all text areas within chat messages use the same blue */
    [data-testid="stChatMessage"] *,
    [data-testid="stChatMessage"] > *,
    [data-testid="stChatMessage"] > * > *,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span {
        background-color: transparent !important;
    }
    
    [data-testid="stChatMessage"][data-testid="user"] *,
    [data-testid="stChatMessage"][data-testid="user"] > *,
    [data-testid="stChatMessage"][data-testid="user"] > * > *,
    [data-testid="stChatMessage"][data-testid="user"] div,
    [data-testid="stChatMessage"][data-testid="user"] p,
    [data-testid="stChatMessage"][data-testid="user"] span {
        background-color: transparent !important;
    }
    
    [data-testid="stChatMessage"][data-testid="assistant"] *,
    [data-testid="stChatMessage"][data-testid="assistant"] > *,
    [data-testid="stChatMessage"][data-testid="assistant"] > * > *,
    [data-testid="stChatMessage"][data-testid="assistant"] div,
    [data-testid="stChatMessage"][data-testid="assistant"] p,
    [data-testid="stChatMessage"][data-testid="assistant"] span {
        background-color: transparent !important;
    }
    
    /* Markdown elements in chat messages */
    [data-testid="stChatMessage"] .stMarkdown,
    [data-testid="stChatMessage"] .stMarkdown > div,
    [data-testid="stChatMessage"] .stMarkdown > div > div,
    [data-testid="stChatMessage"] .stMarkdown p,
    [data-testid="stChatMessage"] .element-container,
    [data-testid="stChatMessage"] .stMarkdownContainer {
        background-color: transparent !important;
    }
    
    /* Override any navy blue backgrounds in chat messages */
    [data-testid="stChatMessage"] [style*="background-color: #001f3f"],
    [data-testid="stChatMessage"] [style*="background-color:#001f3f"] {
        background-color: transparent !important;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stMarkdown p {
        color: #fafafa !important;
    }
    
    /* Input box */
    .stTextInput input {
        background-color: #0a1929 !important;
        color: #fafafa !important;
        border: 1px solid #1565c0 !important;
    }
    
    /* Number input */
    .stNumberInput input {
        background-color: #0a1929 !important;
        color: #fafafa !important;
        border: 1px solid #1565c0 !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }
    
    .stNumberInput input:focus {
        border-color: #1976d2 !important;
        box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.2) !important;
        outline: none !important;
    }
    
    .stNumberInput label {
        color: #fafafa !important;
    }
    
    /* Select boxes */
    .stSelectbox select {
        background-color: #0a1929 !important;
        color: #fafafa !important;
    }
    
    /* Multiselect */
    .stMultiSelect label {
        color: #fafafa !important;
    }
    
    .stMultiSelect > div > div {
        background-color: #0a1929 !important;
    }
    
    /* Slider */
    .stSlider {
        color: #64b5f6 !important;
    }
    
    .stSlider label {
        color: #fafafa !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1976d2 !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #1565c0 !important;
    }
    
    /* Chat input - Entirely white box with black text */
    [data-testid="stChatInput"] {
        background-color: white !important;
        padding: 1rem 1rem 1rem 1rem !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
        border-radius: 25px !important;
    }
    
    [data-testid="stChatInput"] > div {
        background-color: white !important;
        border-radius: 25px !important;
        padding: 0.4rem 0.8rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        margin: 0 !important;
    }
    
    [data-testid="stChatInput"] input {
        background-color: white !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.5rem 0.8rem !important;
        font-size: 0.95rem !important;
    }
    
    [data-testid="stChatInput"] input::placeholder {
        color: #666 !important;
    }
    
    /* Ensure all elements inside chat input are white */
    [data-testid="stChatInput"] * {
        background-color: white !important;
    }
    
    /* Text color for user input */
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] textarea {
        color: #000000 !important;
    }
    
    /* Equal spacing container */
    [data-testid="stChatInputContainer"] {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    /* Ensure equal spacing above and below */
    .stApp > div:last-child {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Main content bottom spacing */
    .main .block-container {
        padding-bottom: 1rem !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #90caf9 !important;
    }
    
    /* Remove spinners and status messages */
    .stSpinner {
        display: none !important;
    }
    
    /* Error and warning messages */
    .stAlert {
        background-color: #001f3f !important;
        border-left: 4px solid #ef4444 !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #fafafa !important;
    }
    
    /* Scrollbar - Navy blue */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a1929;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1565c0;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #1976d2;
    }
    
    /* Additional elements */
    .element-container {
        background-color: #001f3f !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #0a1929 !important;
        color: #fafafa !important;
    }
    
    /* Reduce spacing in chat messages */
    [data-testid="stChatMessage"] .stMarkdown {
        margin-bottom: 0.5rem !important;
        background-color: transparent !important;
    }
    
    [data-testid="stChatMessage"] .stMarkdown p {
        margin-bottom: 0.3rem !important;
        background-color: transparent !important;
    }
    
    [data-testid="stChatMessage"] hr {
        margin: 0.5rem 0 !important;
    }
    
    /* Force all containers within chat messages to be transparent */
    [data-testid="stChatMessage"] .element-container,
    [data-testid="stChatMessage"] .block-container,
    [data-testid="stChatMessage"] [class*="container"],
    [data-testid="stChatMessage"] [class*="markdown"] {
        background-color: transparent !important;
    }
    
    /* Hide any status indicators */
    [data-testid="stStatusWidget"] {
        display: none !important;
    }
    
    /* Sidebar settings boxes - Curved shape */
    .sidebar-box {
        background-color: #0d47a1 !important;
        border-radius: 20px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.4) !important;
        min-height: auto !important;
        display: flex !important;
        flex-direction: column !important;
        visibility: visible !important;
        align-items: stretch !important;
    }
    
    /* Ensure all content is inside the box */
    .sidebar-box > * {
        margin: 0 !important;
        width: 100% !important;
    }
    
    .sidebar-box .stMarkdown {
        margin-bottom: 0.8rem !important;
        text-align: left !important;
    }
    
    /* Ensure sidebar box doesn't affect button color */
    .sidebar-box button {
        background-color: #0d47a1 !important;
        background: #0d47a1 !important;
    }
    
    /* Align multiselect and number input properly */
    .sidebar-box .stMultiSelect,
    .sidebar-box .stNumberInput {
        width: 100% !important;
        margin-top: 0.5rem !important;
    }
    
    /* Align button in center */
    .sidebar-box .stButton {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        margin-top: 0.5rem !important;
    }
    
    /* Selected documents - navy blue background */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #001f3f !important;
        color: #fafafa !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    .stMultiSelect [data-baseweb="tag"] span {
        color: #fafafa !important;
    }
    
    /* Slider - Simple design with white drag handle */
    .stSlider {
        background-color: transparent !important;
    }
    
    /* Slider track - white and thick - multiple selectors to ensure visibility */
    .stSlider > div > div,
    .stSlider [data-baseweb="slider"],
    .stSlider [data-baseweb="slider"] > div,
    .stSlider [data-baseweb="slider"] > div > div,
    .stSlider [data-baseweb="slider-track"],
    .stSlider [data-baseweb="slider-track"] > div {
        background-color: white !important;
        background: white !important;
        height: 10px !important;
        min-height: 10px !important;
        border-radius: 5px !important;
        border: none !important;
    }
    
    /* Slider track inner element */
    .stSlider [data-baseweb="slider"] [data-baseweb="slider-track"] {
        background-color: white !important;
        background: white !important;
        height: 10px !important;
    }
    
    /* Slider thumb/drag handle - Simple and functional */
    .stSlider [role="slider"],
    .stSlider [data-baseweb="thumb"],
    .stSlider button[role="slider"] {
        background-color: #0d47a1 !important;
        background: #0d47a1 !important;
        width: 22px !important;
        height: 22px !important;
        min-width: 22px !important;
        min-height: 22px !important;
        border: 2px solid white !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4) !important;
        cursor: grab !important;
        touch-action: none !important;
        -webkit-tap-highlight-color: transparent !important;
        pointer-events: auto !important;
        z-index: 10 !important;
        visibility: visible !important;
        opacity: 1 !important;
        display: block !important;
        position: relative !important;
    }
    
    .stSlider [role="slider"]:hover,
    .stSlider [data-baseweb="thumb"]:hover,
    .stSlider button[role="slider"]:hover {
        background-color: #1565c0 !important;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.5) !important;
        transform: scale(1.1) !important;
    }
    
    .stSlider [role="slider"]:active,
    .stSlider [data-baseweb="thumb"]:active,
    .stSlider button[role="slider"]:active {
        background-color: #1976d2 !important;
        cursor: grabbing !important;
        transform: scale(1.05) !important;
    }
    
    /* Remove any transforms or complex styles that might interfere */
    .stSlider [role="slider"],
    .stSlider [data-baseweb="thumb"],
    .stSlider button[role="slider"] {
        transition: background-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    /* Ensure slider track is clickable */
    .stSlider [data-baseweb="slider"],
    .stSlider [data-baseweb="slider-track"] {
        cursor: pointer !important;
        pointer-events: auto !important;
        position: relative !important;
    }
    
    /* Hide any drag indicators or overlays */
    .stSlider [class*="drag"],
    .stSlider [class*="overlay"],
    .stSlider [style*="visibility: hidden"],
    .stSlider [style*="opacity: 0"] {
        visibility: hidden !important;
        opacity: 0 !important;
        display: none !important;
    }
    
    /* Slider value display */
    .stSlider label {
        color: #fafafa !important;
    }
    
    /* Clear Chat button - Navy blue with white border - More specific selectors */
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] .sidebar-box button,
    [data-testid="stSidebar"] .sidebar-box .stButton > button,
    [data-testid="stSidebar"] button[data-testid="baseButton-secondary"],
    [data-testid="stSidebar"] [class*="button"] {
        background-color: #001f3f !important;
        background: #001f3f !important;
        color: white !important;
        border: 2px solid white !important;
        border-color: white !important;
        padding: 0.6rem 1.2rem !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        width: 100% !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] button:hover,
    [data-testid="stSidebar"] .sidebar-box button:hover,
    [data-testid="stSidebar"] .sidebar-box .stButton > button:hover,
    [data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover,
    [data-testid="stSidebar"] [class*="button"]:hover {
        background-color: #0a1929 !important;
        background: #0a1929 !important;
        border-color: white !important;
        box-shadow: 0 4px 8px rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:active,
    [data-testid="stSidebar"] button:active,
    [data-testid="stSidebar"] .sidebar-box button:active,
    [data-testid="stSidebar"] .sidebar-box .stButton > button:active,
    [data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:active,
    [data-testid="stSidebar"] [class*="button"]:active {
        background-color: #0d47a1 !important;
        background: #0d47a1 !important;
        border-color: white !important;
        transform: translateY(0) !important;
        box-shadow: 0 2px 4px rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Ensure button text and icon are white */
    [data-testid="stSidebar"] .stButton > button *,
    [data-testid="stSidebar"] button *,
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] button {
        color: white !important;
    }
    
    /* Remove extra boxes - only keep sidebar-box class */
    [data-testid="stSidebar"] .element-container:not(:has(.sidebar-box)) {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Hide any drag overlay or indicator elements */
    .stSlider [class*="drag-indicator"],
    .stSlider [class*="drag-overlay"],
    .stSlider [aria-label*="drag"],
    .stSlider [title*="drag"] {
        visibility: hidden !important;
        opacity: 0 !important;
        display: none !important;
    }
    
    /* Ensure slider container has proper alignment */
    .stSlider > div {
        display: flex !important;
        flex-direction: column !important;
        align-items: stretch !important;
        width: 100% !important;
    }
    
    /* Ensure multiselect has proper alignment */
    .stMultiSelect > div {
        width: 100% !important;
    }
    
    /* Sidebar content alignment */
    [data-testid="stSidebar"] .element-container {
        width: 100% !important;
    }
    
    /* Ensure sidebar boxes align properly */
    [data-testid="stSidebar"] {
        display: flex !important;
        flex-direction: column !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'is_ready' not in st.session_state:
    st.session_state.is_ready = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

def stream_text(text, speed=0.02):
    """Stream text word by word for realistic typing effect"""
    words = text.split()
    placeholder = st.empty()
    full_text = ""
    
    for word in words:
        full_text += word + " "
        placeholder.markdown(full_text)
        time.sleep(speed)
    
    # Final update to ensure complete text
    placeholder.markdown(full_text.strip())
    return full_text.strip()

# Initialize QA system and load trained models
if st.session_state.qa_system is None:
    st.session_state.qa_system = QASystem()
    # Try to load saved data (trained models)
    try:
        st.session_state.qa_system.load_saved_data()
        if (st.session_state.qa_system.chunks and 
            st.session_state.qa_system.chunk_embeddings is not None):
            st.session_state.is_ready = True
    except Exception as e:
        st.session_state.is_ready = False

# Sidebar - Settings with boxes
with st.sidebar:
    st.header("⚙️ Settings")
    
    # PDF Selection Box
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    st.markdown("**Select PDF(s):**")
    if st.session_state.is_ready and st.session_state.qa_system.chunks:
        available_files = sorted(list(set([chunk.get('source_file', 'Unknown') for chunk in st.session_state.qa_system.chunks])))
        
        # Initialize selected files in session state
        if 'selected_files' not in st.session_state:
            st.session_state.selected_files = available_files  # Default: all files selected
        
        selected_files = st.multiselect(
            "Choose PDF files",
            options=available_files,
            default=st.session_state.selected_files,
            help="Select one or more PDFs to search. Leave empty to search all PDFs.",
            key="pdf_selector",
            label_visibility="collapsed"
        )
        
        # Update session state (empty list means search all)
        st.session_state.selected_files = selected_files
    else:
        st.session_state.selected_files = []
        st.info("No PDFs available")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chunks to Retrieve Box
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    st.markdown("**Chunks to Retrieve:**")
    top_k = st.number_input(
        "", 
        min_value=1, 
        max_value=20, 
        value=5,
        step=1,
        help="Number of document chunks to retrieve for each query",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clear Chat Button Box (smaller)
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_btn"):
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Main interface - Clean Layout
# Header with title and image
col_title, col_image = st.columns([4, 1])
with col_title:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px;">
        <span style="font-size: 48px;">⚓</span>
        <h1 style="font-family: 'Segoe UI', Arial, sans-serif; font-weight: 400; font-size: 2.5rem; margin: 0; letter-spacing: 2px;">25th ITMC CHATBOT</h1>
    </div>
    """, unsafe_allow_html=True)
with col_image:
    # Get the image path relative to the script location
    image_path = os.path.join(os.path.dirname(__file__), "Naval_Ensign_of_India.svg.webp")
    if os.path.exists(image_path):
        st.image(image_path, width=200)
    else:
        # Fallback: try current directory
        if os.path.exists("Naval_Ensign_of_India.svg.webp"):
            st.image("Naval_Ensign_of_India.svg.webp", width=200)

# Check if system is ready (silent check)
if not st.session_state.is_ready:
    st.error("⚠️ System not ready. Please train the model first using `train_model.py`")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    # Set custom avatar based on role
    avatar = "❓" if message["role"] == "user" else "💡"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        # Show sources if available (minimal)
        if message["role"] == "assistant" and "sources" in message:
            sources = message.get("sources", [])
            if sources:
                st.caption(f"📄 Sources: {', '.join(sources[:3])}{'...' if len(sources) > 3 else ''}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.is_ready:
        st.warning("⚠️ System not ready. Please train the model first using train_model.py")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="❓"):
            st.markdown(prompt)
        
        # Generate response with word-by-word streaming
        with st.chat_message("assistant", avatar="💡"):
            try:
                # Get selected files (empty list means search all)
                selected_files = st.session_state.get('selected_files', [])
                
                # Query the system (backend logic unchanged)
                result = st.session_state.qa_system.query(
                    prompt, 
                    top_k=top_k,
                    selected_files=selected_files if selected_files else None
                )
                
                chunks = result.get('chunks', [])
                sources = result.get('sources', [])
                
                # Display each chunk separately with clear visual separation
                if chunks:
                    # Group chunks by source file
                    chunks_by_file = {}
                    for chunk in chunks:
                        file_name = chunk.get('source_file', 'Unknown')
                        if file_name not in chunks_by_file:
                            chunks_by_file[file_name] = []
                        chunks_by_file[file_name].append(chunk)
                    
                    # Display chunks grouped by file
                    for file_name, file_chunks in chunks_by_file.items():
                        st.markdown(f"**From PDF: {file_name}**")
                        
                        for idx, chunk in enumerate(file_chunks, 1):
                            chunk_text = chunk.get('text', '')
                            if chunk_text:
                                # Stream each chunk word by word
                                st.markdown(f"**Response {idx}:**")
                                stream_text(chunk_text, speed=0.02)
                                # Add minimal spacing between responses
                                if idx < len(file_chunks):
                                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Build full answer for message history
                    answer_parts = []
                    for file_name, file_chunks in chunks_by_file.items():
                        answer_parts.append(f"**📄 {file_name}**\n")
                        for idx, chunk in enumerate(file_chunks, 1):
                            answer_parts.append(f"**Chunk {idx}:**\n{chunk.get('text', '')}\n\n---\n")
                    answer = "\n".join(answer_parts)
                else:
                    answer = "I couldn't find relevant information to answer your question."
                    st.markdown(answer)
                
                # Show sources (minimal)
                if sources:
                    st.caption(f"📄 Sources: {', '.join(sources[:3])}{'...' if len(sources) > 3 else ''}")
                
                # Add to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

