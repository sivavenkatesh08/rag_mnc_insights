import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Config paths
INDEX_PATH = r"C:Navigate Labs\rag_mnc_insights\data\Transcripts\outputs\mnc_faiss_index"

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
vectorstore = FAISS.load_local(
    folder_path=INDEX_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

def search_query(query: str, k: int = 3):
    """Perform semantic search on the FAISS index."""
    results = vectorstore.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    print("🔍 FAISS Search Engine Ready")
    while True:
        user_query = input("\nEnter your query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("👋 Exiting.")
            break
        
        docs = search_query(user_query, k=3)

        print(f"\nTop {len(docs)} Results:\n")
        for i, doc in enumerate(docs, 1):
            print(f"--- Result #{i} ---")
            print(f"📁 Company: {doc.metadata.get('company', 'N/A')}")
            print(f"📝 Filename: {doc.metadata.get('filename', 'N/A')}")
            print(f"📄 Content:\n{doc.page_content}\n")
