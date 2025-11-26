# ============================================================================
# INAS POLICY CHATBOT - PRODUCTION VERSION 1.0
# Optimized for project submission - November 2025
# Features: Multi-model RAG, Persistent KB, Numerical safety, Offline support
# ============================================================================

import os
# Note: Offline mode disabled to allow model downloads
# Set these to '1' only after models are downloaded
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import subprocess
import time
import requests
import sys
import json
import pickle
import faiss
import re
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ sentence-transformers not installed.")
    print("Run: pip install sentence-transformers")
    if __name__ == "__main__":
        input("Press Enter to exit...")
        sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Updated paths to work from chatbot folder
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
PERSIST_DIR = os.path.join(BASE_DIR, "persistent_kb")
EMBED_DIR = os.path.join(BASE_DIR, "embedding_models")
os.makedirs(KB_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

KB_MODE = "full"

# Suppress print statements when imported as module
SUPPRESS_PRINTS = False

def log_print(*args, **kwargs):
    if not SUPPRESS_PRINTS:
        print(*args, **kwargs)

# ============================================================================
# MODEL REGISTRY - OPTIMIZED FOR PRODUCTION
# ============================================================================
MODEL_REGISTRY = {
    "llama3.2:1b": {
        "size_gb": 1.3,
        "min_ram_gb": 3.5,
        "speed": "very_fast",
        "quality": "good",
        "temperature": 0.1,
        "num_ctx": 2048,
        "num_predict": 600,
        "top_p": 0.85,
        "top_k": 30,
    },
    "llama3.2:3b": {
        "size_gb": 2.0,
        "min_ram_gb": 5.0,
        "speed": "fast",
        "quality": "very_good",
        "temperature": 0.2,
        "num_ctx": 3072,
        "num_predict": 800,
        "top_p": 0.85,
        "top_k": 30,
    },
    "llama3.1:8b": {
        "size_gb": 4.7,
        "min_ram_gb": 999.0,  # Effectively disabled
        "speed": "slow",
        "quality": "exceptional",
        "temperature": 0.3,
        "num_ctx": 8192,
        "num_predict": 1500,
        "top_p": 0.9,
        "top_k": 40,
    },
}

MODEL_ALIASES = {
    "phi3.5": "phi3.5:3.8b-mini-instruct-q4_K_M",
    "phi": "phi3.5:3.8b-mini-instruct-q4_K_M",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_system_resources():
    if not PSUTIL_AVAILABLE:
        return {
            "total_ram_gb": 8.0,
            "available_ram_gb": 4.0,
            "used_ram_percent": 50.0,
            "cpu_count": 4,
            "cpu_usage_percent": 50.0,
        }
    try:
        mem = psutil.virtual_memory()
        return {
            "total_ram_gb": round(mem.total / (1024 ** 3), 1),
            "available_ram_gb": round(mem.available / (1024 ** 3), 1),
            "used_ram_percent": round(mem.percent, 1),
            "cpu_count": psutil.cpu_count(),
            "cpu_usage_percent": round(psutil.cpu_percent(interval=0.5), 1),
        }
    except:
        return {"total_ram_gb": 8.0, "available_ram_gb": 4.0, "used_ram_percent": 50.0, "cpu_count": 4, "cpu_usage_percent": 50.0}

def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except:
        return False

def start_ollama():
    if check_ollama():
        return True
    log_print("Starting Ollama server...")
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        if check_ollama():
            log_print("✓ Ollama started.")
            return True
        log_print("⚠ Ollama did not respond on port 11434.")
        return False
    except Exception as e:
        log_print(f"⚠ Could not start Ollama: {e}")
        return False

def get_available_models():
    try:
        if not check_ollama():
            return []
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
        return []
    except:
        return []

# ============================================================================
# QUERY CLASSIFICATION
# ============================================================================
def classify_query_complexity(query: str) -> str:
    q = query.lower()
    complex_kw = ["analyze", "analyse", "compare", "evaluate", "summarize", "summarise",
                  "explain in detail", "comprehensive", "step by step", "reasoning", "discuss"]
    numerical_kw = ["how many", "number", "count", "percentage", "percent", "%", "statistics",
                    "total", "amount", "budget", "cost", "figure", "share", "growth", "lakh", "crore"]
    simple_kw = ["what is", "who is", "when", "where", "define", "meaning of", "full form", "stands for"]

    score = 0
    if any(k in q for k in complex_kw): score += 3
    if any(k in q for k in numerical_kw): score += 2
    if any(k in q for k in simple_kw): score -= 1

    words = len(query.split())
    if words > 15: score += 2
    elif words > 8: score += 1

    if score >= 4: return "complex"
    elif score >= 2 or any(k in q for k in numerical_kw): return "moderate"
    else: return "simple"

def resolve_model_name(name: str) -> str:
    name = name.strip()
    if name in MODEL_REGISTRY: return name
    if name in MODEL_ALIASES: return MODEL_ALIASES[name]
    return name

def select_optimal_model(query, detailed=False, force_model=None):
    available = get_available_models()
    resources = get_system_resources()
    avail_ram = resources["available_ram_gb"]

    if force_model:
        fm = resolve_model_name(force_model)
        if fm in MODEL_REGISTRY and fm in available:
            if MODEL_REGISTRY[fm]["min_ram_gb"] > 100:
                log_print(f"⚠ Model '{force_model}' is disabled (known issues). Using auto-selection.")
            else:
                log_print(f"📌 Forced model: {fm}")
                return fm, MODEL_REGISTRY[fm]
        else:
            log_print(f"⚠ Forced model '{force_model}' not available; falling back to auto.")

    installed = {m: cfg for m, cfg in MODEL_REGISTRY.items() if m in available and cfg["min_ram_gb"] < 100}
    if not installed:
        if available:
            return available[0], MODEL_REGISTRY.get(available[0], MODEL_REGISTRY["llama3.2:1b"])
        return "llama3.2:1b", MODEL_REGISTRY["llama3.2:1b"]

    complexity = classify_query_complexity(query)
    candidates = []
    for name, cfg in installed.items():
        if avail_ram < cfg["min_ram_gb"]: continue
        score = 0
        if detailed or complexity == "complex":
            score += 10 if cfg["quality"] in ["exceptional", "excellent"] else 7
        elif complexity == "moderate":
            score += 8 if cfg["quality"] in ["excellent", "very_good"] else 5
        else:
            score += 8 if cfg["speed"] in ["very_fast", "fast"] else 4
        candidates.append((name, cfg, score))

    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[0][0], candidates[0][1]

    smallest = min(installed.items(), key=lambda x: x[1]["size_gb"])
    return smallest[0], smallest[1]

# ============================================================================
# LOAD KB - Initialize on import
# ============================================================================
faiss_index = None
chunks = []
persist_chunks = []
embedding_model = None

def initialize_kb():
    global faiss_index, chunks, persist_chunks, embedding_model
    
    log_print("\nLoading knowledge base...")
    try:
        # Check if KB files exist
        faiss_path = os.path.join(KB_DIR, "faiss_index.bin")
        chunks_path = os.path.join(KB_DIR, "chunks.pkl")
        
        if os.path.exists(faiss_path) and os.path.exists(chunks_path):
            faiss_index = faiss.read_index(faiss_path)
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            for c in chunks:
                c["is_persistent"] = False
                if "data_bank" not in c: c["data_bank"] = "original"
            log_print(f"✓ Loaded {len(chunks)} chunks from knowledge base")
        else:
            # Create empty index if KB doesn't exist
            log_print("⚠ Knowledge base not found. Creating empty index.")
            log_print("  Downloading embedding model (first time setup)...")
            try:
                embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
                faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
                chunks = []
                log_print("✓ Created empty knowledge base")
            except Exception as e:
                log_print(f"❌ Failed to download model: {e}")
                raise

        emb_path = os.path.join(EMBED_DIR, "all-MiniLM-L6-v2")
        if os.path.exists(emb_path):
            try:
                embedding_model = SentenceTransformer(emb_path, device="cpu")
                log_print(f"✓ Loaded embedding model from: {emb_path}")
            except Exception as e:
                log_print(f"⚠ Error loading from {emb_path}: {e}")
                log_print("  Downloading model from HuggingFace...")
                embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
                log_print("✓ Loaded embedding model from HuggingFace")
        else:
            log_print("  Downloading embedding model (first time setup)...")
            try:
                embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
                log_print("✓ Loaded embedding model from HuggingFace")
            except Exception as e:
                log_print(f"❌ Failed to download model: {e}")
                raise

        persist_chunks = []
        persist_chunks_path = os.path.join(PERSIST_DIR, "persistent_chunks.pkl")
        persist_embeddings_path = os.path.join(PERSIST_DIR, "persistent_embeddings.npy")
        if os.path.exists(persist_chunks_path) and os.path.exists(persist_embeddings_path):
            import numpy as np
            with open(persist_chunks_path, "rb") as f:
                persist_chunks = pickle.load(f)
            for c in persist_chunks: c["is_persistent"] = True
            persist_embeddings = np.load(persist_embeddings_path)
            if faiss_index is not None:
                faiss_index.add(persist_embeddings.astype("float32"))
            chunks.extend(persist_chunks)
            log_print(f"✓ Loaded {len(persist_chunks)} chunks from uploaded documents")

        log_print(f"✓ Total chunks: {len(chunks)}")

    except Exception as e:
        log_print(f"❌ Error loading KB: {e}")
        # Create minimal setup
        try:
            log_print("  Attempting to download embedding model...")
            embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
            chunks = []
            persist_chunks = []
            log_print("✓ Created minimal knowledge base")
        except Exception as e2:
            log_print(f"❌ Critical error: {e2}")
            log_print("  Make sure you have internet connection to download the model.")
            log_print("  Or install the model manually: python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')\"")
            if __name__ == "__main__":
                input("Press Enter to exit...")
                sys.exit(1)

# Initialize KB
initialize_kb()

# Start Ollama
start_ollama()

# ============================================================================
# NUMERICAL VALIDATION
# ============================================================================
def extract_numbers(text: str):
    return re.findall(r"\d[\d,\.]*", text)

def validate_numerical_answer(question: str, answer: str, context: str):
    ans_nums = set(extract_numbers(answer))
    ctx_nums = set(extract_numbers(context))
    if not ans_nums: return {"verified": True, "warning": ""}
    unverified = [n for n in ans_nums if n not in ctx_nums]
    if unverified:
        return {"verified": False, "warning": f"\n⚠ Numbers not in source: {', '.join(unverified)}"}
    return {"verified": True, "warning": ""}

# ============================================================================
# PERSISTENT KB MANAGEMENT
# ============================================================================
def save_persistent_kb():
    try:
        if persist_chunks:
            import numpy as np
            log_print("  💾 Saving persistent KB...")
            emb = embedding_model.encode([c["text"] for c in persist_chunks], batch_size=32, convert_to_numpy=True)
            with open(os.path.join(PERSIST_DIR, "persistent_chunks.pkl"), "wb") as f:
                pickle.dump(persist_chunks, f)
            np.save(os.path.join(PERSIST_DIR, "persistent_embeddings.npy"), emb.astype("float32"))
            log_print("  ✓ Persistent KB saved.")
    except Exception as e:
        log_print(f"  ⚠ Error saving: {e}")

def clear_persistent_kb():
    try:
        for f in [os.path.join(PERSIST_DIR, "persistent_chunks.pkl"), 
                  os.path.join(PERSIST_DIR, "persistent_embeddings.npy")]:
            if os.path.exists(f): os.remove(f)
        global chunks, persist_chunks, faiss_index
        chunks = [c for c in chunks if not c.get("is_persistent")]
        persist_chunks.clear()
        import numpy as np
        faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        if chunks:
            texts = [c["text"] for c in chunks]
            emb = embedding_model.encode(texts, batch_size=32, convert_to_numpy=True)
            faiss_index.add(emb.astype("float32"))
        log_print("✅ Cleared persistent KB.")
    except Exception as e:
        log_print(f"❌ Error: {e}")

def clear_file_chunks(filename: str):
    global chunks, persist_chunks, faiss_index
    log_print(f"\n🧹 Removing: {filename}")
    try:
        remaining = [c for c in chunks if c.get("source_file") != filename]
        persist_remaining = [c for c in persist_chunks if c.get("source_file") != filename]
        removed = len(chunks) - len(remaining)
        chunks = remaining
        persist_chunks = persist_remaining
        import numpy as np
        faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        if chunks:
            emb = embedding_model.encode([c["text"] for c in chunks], batch_size=32, convert_to_numpy=True)
            faiss_index.add(emb.astype("float32"))
        save_persistent_kb()
        log_print(f"✅ Removed {removed} chunks")
    except Exception as e:
        log_print(f"❌ Error: {e}")

# ============================================================================
# FILE PROCESSING
# ============================================================================
def simple_text_splitter(text, chunk_size=1500, overlap=500):
    out, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        out.append(text[start:end])
        start += chunk_size - overlap
    return out

def process_file(filepath, bank_name=None, persist=True):
    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)
    text = ""

    log_print(f"\n{'='*70}\n📄 Processing: {filename}\n{'='*70}")

    try:
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(filepath)
                    for page in pdf_reader.pages:
                        pt = page.extract_text()
                        if pt and len(pt.strip()) > 10: text += pt + "\n"
                except:
                    log_print(f"  ❌ Error reading PDF")
                    return None
            else:
                reader = PdfReader(filepath)
                log_print(f"  📖 {len(reader.pages)} pages")
                for i, page in enumerate(reader.pages):
                    if i and i % 50 == 0: log_print(f"    {i}/{len(reader.pages)} pages...")
                    try:
                        pt = page.extract_text()
                        if pt and len(pt.strip()) > 10: text += pt + "\n"
                    except: continue
        elif ext == ".docx":
            try:
                from docx import Document
                for p in Document(filepath).paragraphs:
                    if p.text.strip(): text += p.text + "\n"
            except ImportError:
                log_print("  ⚠ python-docx not installed. Install with: pip install python-docx")
                return None
        elif ext in [".xlsx", ".xls"]:
            try:
                import openpyxl
                for sheet in openpyxl.load_workbook(filepath, read_only=True, data_only=True).worksheets:
                    for row in sheet.iter_rows(values_only=True):
                        line = " ".join(str(c) for c in row if c)
                        if line.strip(): text += line + "\n"
            except ImportError:
                log_print("  ⚠ openpyxl not installed. Install with: pip install openpyxl")
                return None
        else:
            log_print(f"  ⚠ Unsupported: {ext}")
            return None
    except Exception as e:
        log_print(f"  ❌ Error: {e}")
        return None

    if len(text) < 50:
        log_print("  ⚠ File too short")
        return None

    log_print("  ⏳ Chunking...")
    chunk_texts = simple_text_splitter(text)
    log_print(f"  ✓ {len(chunk_texts)} chunks")

    new_chunks = [{"text": ct, "source_file": filename, "source_folder": bank_name or "uploaded",
                   "file_type": ext, "chunk_id": i, "total_chunks": len(chunk_texts),
                   "data_bank": bank_name or "uploaded", "timestamp": datetime.now().isoformat(),
                   "is_persistent": bool(persist)} for i, ct in enumerate(chunk_texts)]

    log_print("  ⏳ Encoding...")
    emb = embedding_model.encode([c["text"] for c in new_chunks], batch_size=32, convert_to_numpy=True)
    faiss_index.add(emb.astype("float32"))
    chunks.extend(new_chunks)
    if persist:
        persist_chunks.extend(new_chunks)
        save_persistent_kb()
    log_print(f"✅ Added {len(new_chunks)} chunks\n{'='*70}\n")
    return new_chunks

def load_databank(folder_path, persist=True):
    folder_path = folder_path.strip().strip('"').strip("'")
    if not os.path.exists(folder_path):
        log_print(f"❌ Not found: {folder_path}")
        return
    log_print(f"\n📂 Loading: {os.path.basename(folder_path)}")
    files = []
    for root, dirs, fs in os.walk(folder_path):
        for f in fs:
            if f.lower().endswith((".pdf", ".docx", ".txt", ".xlsx", ".xls")):
                files.append(os.path.join(root, f))
    if not files:
        log_print("❌ No supported files")
        return
    log_print(f"Found {len(files)} files")
    count = sum(1 for i, p in enumerate(files, 1) if process_file(p, bank_name=os.path.basename(folder_path), persist=persist))
    log_print(f"\n✅ Loaded {count}/{len(files)} files\n")

# ============================================================================
# RAG QUERY
# ============================================================================
def query_rag(question, top_k=25, detailed=False, target_file=None, force_model=None):
    if not check_ollama():
        log_print("⚠ Ollama not responding")
        return None

    global KB_MODE
    complexity = classify_query_complexity(question)
    is_numerical = any(k in question.lower() for k in ["how many", "number", "count", "percentage", "percent", "%", "statistics", "total", "amount", "budget", "cost", "figure", "share", "growth"])

    model_name, model_cfg = select_optimal_model(question, detailed=detailed, force_model=force_model)

    log_print(f"\n🔍 Searching: {question}")
    log_print(f"🤖 Selected Model: {model_name} ({model_cfg['quality']} quality)")
    log_print(f"📊 Query Type: {complexity.upper()}{' + NUMERICAL' if is_numerical else ''}")
    log_print(f"📁 KB Mode: {KB_MODE.upper()}")

    if detailed: top_k = 40
    search_k = min(len(chunks), top_k * (20 if target_file else 1))
    q_emb = embedding_model.encode([question])
    distances, indices = faiss_index.search(q_emb.astype("float32"), search_k)

    context, sources, found = "", [], 0
    for idx, dist in zip(indices[0], distances[0]):
        if idx >= len(chunks): continue
        ch = chunks[idx]
        if target_file:
            if target_file.lower() not in ch.get("source_file", "").lower(): continue
        else:
            if KB_MODE == "base" and ch.get("is_persistent"): continue
        context += f"\n[{ch['source_file']}]\n{ch['text']}\n"
        sources.append(ch["source_file"])
        found += 1
        if found >= top_k: break

    if found == 0:
        log_print("❌ No relevant info found")
        return None

    unique_sources = sorted(set(sources))
    log_print(f"📋 Retrieved {found} chunks from {len(unique_sources)} file(s)")

    if is_numerical:
        prompt = f"""You are a precise data analyst. Answer ONLY using the document excerpts.

RULES:
1. Extract numbers EXACTLY as written (e.g. "3.6%", "1.54 lakh")
2. Include units and context
3. If specific number NOT in excerpts, say: "The specific numerical value requested is not mentioned in the provided excerpts."
4. DO NOT infer, estimate, or guess
5. NO phrases like "we can infer", "likely", "probably"

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

PRECISE ANSWER:"""
    elif detailed or complexity == "complex":
        prompt = f"""Provide detailed answer using ONLY the excerpts.

RULES:
1. Use only explicit information
2. State numbers/dates exactly as written
3. If insufficient info, say so clearly
4. Structure with clear paragraphs/bullets

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

DETAILED ANSWER:"""
    else:
        prompt = f"""Answer concisely using ONLY the excerpts.

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

CONCISE ANSWER:"""

    try:
        use_stream = not is_numerical

        if use_stream:
            log_print("⏳ Generating answer (streaming)...")
            r = requests.post("http://localhost:11434/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": True,
                      "options": {"temperature": model_cfg["temperature"], "top_p": model_cfg["top_p"],
                                "top_k": model_cfg["top_k"], "repeat_penalty": 1.3,
                                "num_ctx": model_cfg["num_ctx"], "num_predict": model_cfg["num_predict"]}},
                stream=True, timeout=300)

            log_print(f"\n{'='*70}\n📝 ANSWER (Model: {model_name}):\n{'='*70}")
            full_response = ""
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            text = data["response"]
                            log_print(text, end="", flush=True)
                            full_response += text
                    except: continue
        else:
            log_print("⏳ Generating answer (numerical, non-streaming)...")
            r = requests.post("http://localhost:11434/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False,
                      "options": {"temperature": model_cfg["temperature"], "top_p": model_cfg["top_p"],
                                "top_k": model_cfg["top_k"], "repeat_penalty": 1.3,
                                "num_ctx": model_cfg["num_ctx"], "num_predict": model_cfg["num_predict"]}},
                timeout=300)
            full_response = r.json().get("response", "")
            validation = validate_numerical_answer(question, full_response, context)

            log_print(f"\n{'='*70}\n📝 ANSWER (Model: {model_name}):\n{'='*70}")
            if validation["verified"]:
                log_print(full_response)
            else:
                log_print("The specific numerical value requested cannot be safely extracted from the provided excerpts without guessing.")
                log_print(validation["warning"])

        log_print(f"\n{'='*70}")
        log_print(f"📚 Sources ({len(unique_sources)}): {', '.join(unique_sources[:5])}")
        if len(unique_sources) > 5: log_print(f"    + {len(unique_sources) - 5} more")
        log_print(f"🤖 Model: {model_name} | 📊 Type: {complexity}{' + numerical' if is_numerical else ''}")
        log_print("=" * 70)
        return full_response

    except requests.exceptions.Timeout:
        log_print("\n❌ Timeout. Try simpler question.")
        return None
    except Exception as e:
        log_print(f"\n❌ Error: {e}")
        return None
