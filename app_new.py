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
    layout="wide"
)

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'is_ready' not in st.session_state:
    st.session_state.is_ready = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

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

# Sidebar
with st.sidebar:
    st.header("📚 PDF QA System")
    st.markdown("**Query Interface**")
    st.caption("Training: Use `train_model.py` in VS Code")
    st.markdown("---")
    
    # System Status
    st.subheader("System Status")
    
    if st.session_state.is_ready:
        st.success("✅ System Ready")
        
        chunks_count = len(st.session_state.qa_system.chunks) if st.session_state.qa_system.chunks else 0
        st.metric("Chunks", chunks_count)
        
        # Show loaded files
        if st.session_state.qa_system.chunks:
            files = list(set([chunk.get('source_file', 'Unknown') for chunk in st.session_state.qa_system.chunks]))
            with st.expander(f"📄 Trained Files ({len(files)})"):
                for f in files:
                    st.text(f"• {f}")
        
        if st.session_state.qa_system.chunk_embeddings is not None:
            emb_shape = st.session_state.qa_system.chunk_embeddings.shape
            st.metric("Embeddings", f"{emb_shape[0]} x {emb_shape[1]}")
            st.success("✅ Ready for queries using similarity search")
    else:
        st.error("❌ System Not Ready")
        st.warning("""
        **No trained model found!**
        
        Please train the model first:
        1. Run `train_model.py` in VS Code
        2. Upload PDFs and train the model
        3. Then refresh this page
        """)
    
    st.markdown("---")
    
    st.subheader("Query Settings")
    top_k = st.slider("Top K Results", min_value=1, max_value=20, value=5, 
                     help="Number of document chunks to retrieve")
    
    st.markdown("---")
    
    st.subheader("About")
    st.info("""
    **How it works:**
    - Load PDFs and create embeddings
    - Query using cosine similarity search
    - Returns actual PDF content in sequence
    - No model training required!
    """)

# Main interface
st.title("📚 PDF QA System - Query Interface")
st.markdown("Ask questions about your trained PDF documents")

# Check if system is ready
if not st.session_state.is_ready:
    st.error("⚠️ **System Not Ready**")
    st.warning("""
    **Please train the model first:**
    
    1. Open `train_model.py` in VS Code
    2. Run the script: `python train_model.py`
    3. Upload your PDFs and train the model
    4. Once training is complete, refresh this page
    
    The trained model will be automatically loaded.
    """)
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 Sources"):
                for source in message.get("sources", []):
                    st.text(source)

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.is_ready:
        st.warning("⚠️ System not ready. Please train the model first using train_model.py")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    result = st.session_state.qa_system.query(
                        prompt, 
                        top_k=top_k
                    )
                    
                    answer = result['answer']
                    sources = result.get('sources', [])
                    
                    st.markdown(answer)
                    
                    # Show similarity scores
                    if result.get('scores'):
                        with st.expander("📊 Similarity Scores"):
                            for i, (chunk, score) in enumerate(zip(result['chunks'], result['scores'])):
                                st.metric(
                                    f"Chunk {i+1} ({chunk.get('source_file', 'Unknown')})",
                                    f"{score:.3f}"
                                )
                    
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

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")
st.caption("💡 **Tip**: Train the model using `train_model.py` in VS Code, then query here!")

