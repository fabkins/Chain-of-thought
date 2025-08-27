import ollama
import chromadb
import uuid, datetime, os
from colorama import Fore, Style, init

# --- Config ---
MODEL = "llama3.2"   # use same model for chat + embeddings

# Initialize colorama
init(autoreset=True)

# --- Setup ChromaDB ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_bot_collection(bot_id):
    """Return (or create) a ChromaDB collection for this bot."""
    return chroma_client.get_or_create_collection(
        name=f"chat_{bot_id}"
    )

# --- Memory Helpers ---
def embed_text(text: str):
    """Get an embedding using Ollama."""
    return ollama.embeddings(model=MODEL, prompt=text)["embedding"]

def store_memory(bot_id, content, memory_type="event", importance=0.5):
    """Store a memory (core fact or event) into Chroma."""
    collection = get_bot_collection(bot_id)
    doc_id = str(uuid.uuid4())
    collection.add(
        ids=[doc_id],
        embeddings=[embed_text(content)],
        documents=[content],
        metadatas=[{
            "type": memory_type,
            "timestamp": str(datetime.datetime.now()),
            "importance": importance
        }]
    )
    color = Fore.CYAN
    print(color + f"[Memory Stored] ({memory_type.upper()}) {content}")

def retrieve_memories(bot_id, query, n_results=5):
    """Retrieve relevant memories for this bot (core + events)."""
    collection = get_bot_collection(bot_id)
    results = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=n_results
    )
    return results.get("documents", [[]])[0]

def classify_memory(user_input, ai_reply):
    """Classify whether interaction should be core memory or event."""
    classification_prompt = f"""
Decide if this interaction is a CORE FACT (long-term important info about identity, relationships, life events, or consistent traits) 
or a TEMPORARY EVENT (situational, casual, or transient detail).

User: {user_input}
AI: {ai_reply}

Answer only 'core' or 'event'.
"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": classification_prompt}])
    result = response["message"]["content"].strip().lower()
    return "core" if "core" in result else "event"

def summarize_interaction(user_input, ai_reply):
    """Summarize the exchange into a factual sentence."""
    prompt = f"""
Summarize the following interaction into ONE concise factual sentence that captures what was learned.

User: {user_input}
AI: {ai_reply}

Summary:"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

# --- Chat Function ---
def chat_with_bot(bot_id, user_input, recent_history):
    memories = retrieve_memories(bot_id, user_input)
    history = "\n".join([f"{role}: {msg}" for role, msg in recent_history])
    context = "\n".join(memories)

    prompt = f"""
You are a unique character with personality and evolving history.

Core personality and relevant memories:
{context}

Recent conversation:
{history}

User: {user_input}
Assistant (in character):"""

    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    reply = response["message"]["content"]

    # Summarize & classify this turn
    summary = summarize_interaction(user_input, reply)
    memory_type = classify_memory(user_input, reply)
    store_memory(bot_id, summary, memory_type)

    return reply

# --- Seed Bots ---
def seed_bot(bot_id, backstory_facts):
    for fact in backstory_facts:
        store_memory(bot_id, fact, memory_type="core", importance=1.0)

# Example backstories (run once)
def seed_example_bots():
    seed_bot("isabella", [
        "My name is Isabella Rossi.",
        "I was born on April 3rd, 1985, in Florence, Italy.",
        "I studied literature at the University of Bologna.",
        "I worked as a librarian before becoming a travel writer.",
        "I am thoughtful, friendly, and love Renaissance art."
    ])

    seed_bot("kai", [
        "My name is Kai Tanaka.",
        "I was born on November 12th, 1992, in Osaka, Japan.",
        "I used to be a robotics engineer before opening a coffee shop.",
        "I am curious, innovative, and enjoy cyberpunk novels."
    ])

def database_is_empty():
    return len(chroma_client.list_collections()) == 0

# --- Interactive Chat Loop ---
if __name__ == "__main__":
    print(Fore.CYAN + "Multi-Bot Chatbot ready! Type 'quit' to exit.\n")

    # Auto-seed if database doesn't exist
    if database_is_empty():
        print(Fore.CYAN + "No database found, seeding example bots...")
        os.makedirs("./chroma_db", exist_ok=True)
        seed_example_bots()

    recent_histories = {}

    bot_id = input(Fore.YELLOW + "\nTalk to which bot? (e.g. isabella/kai): ").strip()
    if bot_id.lower() in ["quit", "exit"]:
        exit(0)

    while True:
        if bot_id not in recent_histories:
            recent_histories[bot_id] = []

        user_input = input(Fore.YELLOW + f"\nYou to {bot_id}: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        print("\n")
        reply = chat_with_bot(bot_id, user_input, recent_histories[bot_id])
        print("\n")

        print(Fore.GREEN + f"{bot_id}: {reply}")

        # Update short-term memory (last 10 turns)
        recent_histories[bot_id].append(("user", user_input))
        recent_histories[bot_id].append(("assistant", reply))
        recent_histories[bot_id] = recent_histories[bot_id][-10:]
