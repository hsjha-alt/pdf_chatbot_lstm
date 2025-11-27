"""
PDF Data Loader and Chunking Module
Loads PDFs from a folder and creates text chunks
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import re


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
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            source_file: Source file name
            max_text_size: Maximum text size to process (default: 5MB)
            max_chunks: Maximum number of chunks to create (default: 10000)
            
        Returns:
            List of chunk dictionaries
        """
        text = self.clean_text(text)
        if len(text) < 50:
            return []
        
        # Limit text size to prevent memory issues
        if len(text) > max_text_size:
            print(f"  Warning: Text is very large ({len(text)} chars). Truncating to {max_text_size} chars.")
            text = text[:max_text_size]
        
        chunks = []
        start = 0
        chunk_id = 0
        max_iterations = max_chunks * 2  # Safety limit to prevent infinite loops
        
        iteration = 0
        while start < len(text) and chunk_id < max_chunks and iteration < max_iterations:
            iteration += 1
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct != -1 and last_punct > start + self.chunk_size // 2:
                        end = last_punct + 2
                        break
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) > 50:  # Only add meaningful chunks
                chunks.append({
                    'text': chunk_text,
                    'source_file': source_file,
                    'chunk_id': chunk_id,
                    'start_pos': start,
                    'end_pos': end
                })
                chunk_id += 1
            
            # Move start position with overlap
            new_start = end - self.overlap
            if new_start <= start:  # Prevent infinite loop or going backwards
                new_start = start + self.chunk_size - self.overlap
                if new_start >= len(text):
                    break
            start = new_start
        
        if iteration >= max_iterations:
            print(f"  Warning: Reached maximum iterations. Created {len(chunks)} chunks.")
        if chunk_id >= max_chunks:
            print(f"  Warning: Reached maximum chunks limit ({max_chunks}). Some text may not be chunked.")
        
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

