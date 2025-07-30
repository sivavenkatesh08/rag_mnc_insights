import argparse
from rag_pipeline_gemini import rag_query, memory, load_memory_from_file, save_memory_to_file

from pathlib import Path
INDEX_PATH = Path(r"C:\22ad053\Navigate Labs\rag_mnc_insights\data\outputs\mnc_faiss_index")


def print_history():
    print("\n🕓 Chat History:")
    for msg in memory.chat_memory.messages:
        prefix = "👤 You" if msg.type == "human" else "🤖 RAG"
        print(f"{prefix}: {msg.content}\n")


def interactive_chat():
    print("💬 Interactive Mode (type 'exit' to quit, 'reset' to clear memory, 'history' to view log):")
    while True:
        question = input(">> ").strip()

        if question.lower() == "exit":
            save_memory_to_file(memory)
            break
        elif question.lower() == "reset":
            memory.clear()
            print("🔄 Memory cleared.")
        elif question.lower() == "history":
            print_history()
        else:
            rag_query(INDEX_PATH, question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG CLI Assistant")
    parser.add_argument("--question", type=str, help="Ask a one-time question")
    parser.add_argument("--history", action="store_true", help="Show chat history")
    parser.add_argument("--reset", action="store_true", help="Clear chat memory")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat")

    args = parser.parse_args()

    load_memory_from_file(memory)

    if args.reset:
        memory.clear()
        print("🧹 Memory cleared.")

    if args.history:
        print_history()

    if args.question:
        rag_query(INDEX_PATH, args.question)

    if args.interactive or not any([args.question, args.reset, args.history]):
        interactive_chat()

    save_memory_to_file(memory)
