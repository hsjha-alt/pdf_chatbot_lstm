"""
PDF Data Loader and Chunking Module
Loads PDFs from a folder and creates text chunks
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import re

# Try multiple import paths for different LangChain versions
LANGCHAIN_AVAILABLE = False
RecursiveCharacterTextSplitter = None

try:
    # Try newer version (langchain-text-splitters package)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        # Try older version (langchain package)
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        print("Warning: LangChain not available. Install with: pip install langchain langchain-text-splitters")


class PDFDataLoader:
    """Loads PDFs from a folder and creates chunks"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the data loader
        
        Args:
            chunk_size: Maximum size of each chunk in characters (default: 1000)
            overlap: Number of characters to overlap between chunks (default: 200)
                    Note: overlap should be less than chunk_size
        """
        self.chunk_size = chunk_size
        # Ensure overlap is reasonable
        self.overlap = min(overlap, chunk_size - 100) if chunk_size > 100 else min(overlap, chunk_size // 2)
        self.chunks = []
        
        # Initialize LangChain text splitter for paragraph-based chunking
        if LANGCHAIN_AVAILABLE and RecursiveCharacterTextSplitter is not None:
            try:
                # Paragraph-based splitter: prioritize paragraphs, only split if paragraph is too large
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=self.overlap,
                    length_function=len,
                    separators=[
                        "\n\n\n",  # Multiple paragraphs (triple newline)
                        "\n\n",    # Paragraphs (double newline) - PRIMARY SEPARATOR
                        "\n",      # Single newline (only if paragraph is too large)
                        ". ",      # Sentences (only if paragraph is still too large)
                        "! ",      # Exclamations
                        "? ",      # Questions
                        "; ",      # Semicolons
                        ", ",      # Commas
                        " ",       # Words
                        ""         # Characters (last resort)
                    ],
                    is_separator_regex=False,
                    keep_separator=True  # Keep paragraph separators in chunks
                )
            except Exception as e:
                print(f"Warning: Failed to initialize LangChain text splitter: {e}")
                self.text_splitter = None
        else:
            self.text_splitter = None
    
    def load_pdf(self, pdf_path: str, max_pages: Optional[int] = None, max_text_per_page: int = 50000) -> str:
        """
        Extract text from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            max_pages: Maximum number of pages to process (default: None = all pages)
            max_text_per_page: Maximum text length per page (default: 50000)
            
        Returns:
            Extracted text as string
        """
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            
            # Determine how many pages to process
            if max_pages is None:
                pages_to_process = total_pages
            else:
                if total_pages > max_pages:
                    print(f"  Warning: PDF has {total_pages} pages. Processing first {max_pages} pages only.")
                pages_to_process = min(total_pages, max_pages)
            
            text = ""
            for i, page in enumerate(reader.pages):
                if i >= pages_to_process:
                    break
                try:
                    page_text = page.extract_text()
                    if page_text:
                        # Limit text per page to prevent memory issues
                        if len(page_text) > max_text_per_page:
                            page_text = page_text[:max_text_per_page]
                            print(f"  Warning: Page {i+1} text truncated (too long)")
                        if len(page_text.strip()) > 10:
                            text += page_text + "\n"
                except Exception as e:
                    print(f"  Warning: Error extracting page {i+1}: {e}")
                    continue
            return text
        except ImportError:
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    
                    # Determine how many pages to process
                    if max_pages is None:
                        pages_to_process = total_pages
                    else:
                        if total_pages > max_pages:
                            print(f"  Warning: PDF has {total_pages} pages. Processing first {max_pages} pages only.")
                        pages_to_process = min(total_pages, max_pages)
                    
                    text = ""
                    for i in range(pages_to_process):
                        try:
                            page = pdf_reader.pages[i]
                            page_text = page.extract_text()
                            if page_text:
                                # Limit text per page
                                if len(page_text) > max_text_per_page:
                                    page_text = page_text[:max_text_per_page]
                                    print(f"  Warning: Page {i+1} text truncated (too long)")
                                if len(page_text.strip()) > 10:
                                    text += page_text + "\n"
                        except Exception as e:
                            print(f"  Warning: Error extracting page {i+1}: {e}")
                            continue
                    return text
            except Exception as e:
                print(f"Error reading PDF {pdf_path}: {e}")
                return ""
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        return text.strip()
    
    def create_chunks(self, text: str, source_file: str, max_text_size: int = 5000000, max_chunks: int = 10000) -> List[Dict]:
        """
        Split text into chunks based on paragraphs
        
        Args:
            text: Text to chunk
            source_file: Source file name
            max_text_size: Maximum text size to process (default: 5MB)
            max_chunks: Maximum number of chunks to create (default: 10000)
            
        Returns:
            List of chunk dictionaries
        """
        # Basic validation
        if len(text) < 50:
            return []
        
        # Limit text size to prevent memory issues
        if len(text) > max_text_size:
            print(f"  Warning: Text is very large ({len(text)} chars). Truncating to {max_text_size} chars.")
            text = text[:max_text_size]
        
        chunks = []
        
        # Split text into paragraphs (preserve paragraph structure)
        # Paragraphs are separated by double newlines or more
        paragraphs = re.split(r'\n\s*\n+', text)
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        print(f"  Found {len(paragraphs)} paragraphs")
        
        chunk_id = 0
        current_chunk = ""
        
        for para in paragraphs:
            if chunk_id >= max_chunks:
                print(f"  Warning: Reached maximum chunks limit ({max_chunks}). Some text may not be chunked.")
                break
            
            # If adding this paragraph would exceed chunk_size, save current chunk and start new one
            if current_chunk and len(current_chunk) + len(para) + 2 > self.chunk_size:
                # Save current chunk
                if len(current_chunk.strip()) > 50:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'source_file': source_file,
                        'chunk_id': chunk_id,
                        'start_pos': 0,
                        'end_pos': len(current_chunk)
                    })
                    chunk_id += 1
                
                # Start new chunk with current paragraph
                current_chunk = para
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            
            # If a single paragraph is too large, split it by sentences
            if len(current_chunk) > self.chunk_size:
                # Split large paragraph by sentences
                sentences = re.split(r'([.!?]\s+)', current_chunk)
                temp_chunk = ""
                
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
                    
                    if temp_chunk and len(temp_chunk) + len(sentence) > self.chunk_size:
                        # Save temp_chunk
                        if len(temp_chunk.strip()) > 50:
                            chunks.append({
                                'text': temp_chunk.strip(),
                                'source_file': source_file,
                                'chunk_id': chunk_id,
                                'start_pos': 0,
                                'end_pos': len(temp_chunk)
                            })
                            chunk_id += 1
                        temp_chunk = sentence
                    else:
                        temp_chunk += sentence
                
                current_chunk = temp_chunk
        
        # Add the last chunk if it exists
        if current_chunk and len(current_chunk.strip()) > 50:
            chunks.append({
                'text': current_chunk.strip(),
                'source_file': source_file,
                'chunk_id': chunk_id,
                'start_pos': 0,
                'end_pos': len(current_chunk)
            })
        
        print(f"  Created {len(chunks)} paragraph-based chunks")
        return chunks
    
    def load_folder(self, folder_path: str) -> List[Dict]:
        """
        Load all PDFs from a folder and create chunks
        
        Args:
            folder_path: Path to folder containing PDFs
            
        Returns:
            List of all chunks from all PDFs
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            print(f"Folder not found: {folder_path}")
            return []
        
        all_chunks = []
        pdf_files = list(folder_path.glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files in {folder_path}")
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            try:
                text = self.load_pdf(str(pdf_file))
                if text:
                    print(f"  Extracted {len(text)} characters")
                    chunks = self.create_chunks(text, pdf_file.name)
                    all_chunks.extend(chunks)
                    print(f"  Created {len(chunks)} chunks from {pdf_file.name}")
                else:
                    print(f"  Warning: Could not extract text from {pdf_file.name}")
            except MemoryError:
                print(f"  Error: Out of memory processing {pdf_file.name}")
                print(f"  Try processing this file separately or reduce chunk_size/overlap")
                continue
            except Exception as e:
                print(f"  Error processing {pdf_file.name}: {e}")
                continue
        
        self.chunks = all_chunks
        print(f"\nTotal chunks created: {len(all_chunks)}")
        return all_chunks
    
    def save_chunks(self, filepath: str):
        """Save chunks to a pickle file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"Saved {len(self.chunks)} chunks to {filepath}")
    
    def load_chunks(self, filepath: str) -> List[Dict]:
        """Load chunks from a pickle file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"Loaded {len(self.chunks)} chunks from {filepath}")
            return self.chunks
        return []

