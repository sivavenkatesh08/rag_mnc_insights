import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv
import re
from langchain.memory import ConversationBufferMemory


# Load environment variables (like GEMINI API key)
load_dotenv()
configure(api_key="Api key")

# Constants
INDEX_PATH = Path(r"C:Navigate Labs\rag_mnc_insights\data\outputs\mnc_faiss_index")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model = GenerativeModel("models/gemini-2.0-flash")

chat_history = []
memory = ConversationBufferMemory(return_messages=True)

from pathlib import Path
import json
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

MEMORY_FILE = Path(r"C:Navigate Labs\rag_mnc_insights\data\outputs\chat_memory.json")

def save_memory_to_file(memory, filepath=MEMORY_FILE):
    data = [
        {"type": msg.type, "content": msg.content}
        for msg in memory.chat_memory.messages
    ]
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("💾 Chat memory saved.")

def load_memory_from_file(memory, filepath=MEMORY_FILE):
    if not filepath.exists():
        return
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    for msg in data:
        if msg["type"] == "human":
            memory.chat_memory.add_user_message(msg["content"])
        elif msg["type"] == "ai":
            memory.chat_memory.add_ai_message(msg["content"])
    print("✅ Chat memory loaded.")



def extract_metadata_from_question(question: str):
    question = question.lower()
    company_aliases = {
        "microsoft": "MSFT", "apple": "AAPL", "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL",
        "intel": "INTC", "amd": "AMD", "nvidia": "NVDA", "asml": "ASML", "micron": "MU", "cisco": "CSCO"
    }
    company = next((ticker for alias, ticker in company_aliases.items() if alias in question), None)
    year_match = re.search(r"(20\d{2})", question)
    year = year_match.group(1) if year_match else None
    quarter_match = re.search(r"(q[1-4])", question)
    quarter = quarter_match.group(1).upper() if quarter_match else None
    return company, year, quarter

def convert_date_to_quarter(filename, fiscal_year=True):
    if fiscal_year:
        months_to_quarters = {
            "Jul": "Q1", "Aug": "Q1", "Sep": "Q1",
            "Oct": "Q2", "Nov": "Q2", "Dec": "Q2",
            "Jan": "Q3", "Feb": "Q3", "Mar": "Q3",
            "Apr": "Q4", "May": "Q4", "Jun": "Q4",
        }
    else:
        months_to_quarters = {
            "Jan": "Q1", "Feb": "Q1", "Mar": "Q1",
            "Apr": "Q2", "May": "Q2", "Jun": "Q2",
            "Jul": "Q3", "Aug": "Q3", "Sep": "Q3",
            "Oct": "Q4", "Nov": "Q4", "Dec": "Q4",
        }
    return next((q for mon, q in months_to_quarters.items() if mon in filename), "")

def load_vector_db(index_path):
    return FAISS.load_local(
        str(index_path),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

def ask_gemini(context: str, user_question: str) -> str:
    # Add latest user message to memory
    memory.chat_memory.add_user_message(user_question)

    # Include memory history
    history = "\n".join([
        f"User: {msg.content}" if msg.type == "human" else f"Assistant: {msg.content}"
        for msg in memory.chat_memory.messages
    ])

    prompt = f"""
You are a financial analyst assistant. Use the following MNC earnings transcript snippets to answer the user's question precisely.

Chat History:
{history}

Transcript Context:
{context}

Answer:"""

    response = model.generate_content(prompt)
    answer = response.text.strip()

    # Add response to memory
    memory.chat_memory.add_ai_message(answer)
    return answer



def format_metadata(metadata):
    filename = metadata.get("filename", "Unknown")
    company = metadata.get("company", "Unknown")
    date_match = re.search(r"(\d{4})-([A-Za-z]+)-\d{2}", filename)
    if date_match:
        year = date_match.group(1)
        month = date_match.group(2)
        quarter = convert_date_to_quarter(month, fiscal_year=True)
        return f"{company}, {quarter} {year} Earnings Call ({filename})"
    return filename

def filter_documents(docs, company, year, quarter, fiscal=True):
    if not all([company, year, quarter]):
        return docs
    filtered = []
    for doc in docs:
        filename = doc.metadata.get("filename", "")
        if company in filename and year in filename and quarter == convert_date_to_quarter(filename, fiscal_year=fiscal):
            filtered.append(doc)
    return filtered

def rag_query(index_path, user_question, streamlit_mode=False, company=None, year=None, quarter=None):
    db = load_vector_db(index_path)

    if not all([company, year, quarter]):
        company, year, quarter = extract_metadata_from_question(user_question)

    if not streamlit_mode:
        print(f"\n🔍 Detected Metadata — Company: {company}, Year: {year}, Quarter: {quarter}")

    retriever = db.as_retriever(search_kwargs={"k": 10})
    docs = retriever.get_relevant_documents(user_question)
    filtered_docs = filter_documents(docs, company, year, quarter)

    if filtered_docs:
        if not streamlit_mode:
            print(f"✅ Filtered {len(filtered_docs)} documents for {company}, {year}, {quarter}")
    else:
        if not streamlit_mode:
            print("⚠️ No exact match found with metadata. Using broader context.")
        filtered_docs = docs

    context = "\n\n".join(doc.page_content for doc in filtered_docs[:5])
    answer = ask_gemini(context, user_question)

    if streamlit_mode:
        return {
            "answer": answer,
            "sources": [doc.metadata.get("filename", "Unknown") for doc in filtered_docs[:5]]
        }

    print("\n📌 Answer:\n", answer)
    print("\n📄 Sources:")
    for doc in filtered_docs[:5]:
        print("→", doc.metadata.get("filename", "Unknown"))


if __name__ == "__main__":
    print("💬 Chat with your RAG system (type 'exit' to quit, 'reset' to clear memory, 'history' to view chat):")

    load_memory_from_file(memory)

    while True:
        question = input(">> ").strip()
        
        if question.lower() == "exit":
            save_memory_to_file(memory)
            break

        elif question.lower() == "reset":
            memory.clear()
            print("🔄 Memory cleared.")
            continue

        elif question.lower() == "history":
            print("\n🕓 Chat History:")
            for msg in memory.chat_memory.messages:
                prefix = "👤 You" if msg.type == "human" else "🤖 RAG"
                print(f"{prefix}: {msg.content}\n")
            continue

        rag_query(INDEX_PATH, question)




