import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from scripts.clean_chunk_data import load_all_transcripts, BASE_DIR

# 📁 Output path for FAISS index
OUTPUT_DIR = BASE_DIR.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = OUTPUT_DIR / "mnc_faiss_index"

# 🧠 Load transcripts
print("📚 Loading and chunking transcripts...")
docs = load_all_transcripts(BASE_DIR)
print(f"✅ Loaded {len(docs)} chunks.")

# 🔎 Embedding model
print("🔧 Creating embeddings...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🗃️ Create FAISS index
print("💾 Creating FAISS index...")
vectorstore = FAISS.from_documents(docs, embedding_model)
vectorstore.save_local(str(INDEX_PATH))
print(f"✅ FAISS index saved to: {INDEX_PATH}")
