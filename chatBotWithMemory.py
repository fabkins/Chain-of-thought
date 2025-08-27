import ollama
import chromadb
import uuid, datetime

# --- Config ---
MODEL = "llama3.2"   # use same model for chat + embeddings

# --- Setup ChromaDB ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_bot_collection(bot_id):
    """Return (or create) a ChromaDB collection for this bot."""
    # Do NOT use DefaultEmbeddingFunction here, we supply embeddings ourselves
    return chroma_client.get_or_create_collection(
        name=f"chat_{bot_id}"
    )

# --- Memory Helpers ---
def embed_text(text: str):
    """Get an embedding using Ollama."""
    return ollama.embeddings(model=MODEL, prompt=text)["embedding"]

def store_core_fact(bot_id, content):
    collection = get_bot_collection(bot_id)
    doc_id = str(uuid.uuid4())
    collection.add(
        ids=[doc_id],
        embeddings=[embed_text(content)],
        documents=[content],
        metadatas=[{
            "type": "core",
            "timestamp": str(datetime.datetime.now()),
            "importance": 1.0
        }]
    )

def store_event(bot_id, content, importance=0.5):
    collection = get_bot_collection(bot_id)
    doc_id = str(uuid.uuid4())
    collection.add(
        ids=[doc_id],
        embeddings=[embed_text(content)],
        documents=[content],
        metadatas=[{
            "type": "event",
            "timestamp": str(datetime.datetime.now()),
            "importance": importance
        }]
    )

def retrieve_memories(bot_id, query, n_results=5):
    """Retrieve relevant memories for this bot (core + events)."""
    collection = get_bot_collection(bot_id)
    results = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=n_results
    )
    return results.get("documents", [[]])[0]

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

    # Save conversation as an event
    store_event(bot_id, f"Conversation with user: '{user_input}' → '{reply}'", importance=0.5)
    return reply

# --- Seed Bots ---
def seed_bot(bot_id, backstory_facts):
    for fact in backstory_facts:
        store_core_fact(bot_id, fact)

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
import os

if __name__ == "__main__":
    print("Multi-Bot Chatbot ready! Type 'quit' to exit.\n")

    # Auto-seed if database doesn't exist
    if database_is_empty():
        print("No database found, seeding example bots...")
        os.makedirs("./chroma_db", exist_ok=True)
        seed_example_bots()

    recent_histories = {}

    bot_id = input("\nTalk to which bot? (e.g. isabella/kai): ").strip()
    if bot_id.lower() in ["quit", "exit"]:
        # Exit if user wants to quit here
        exit(0)

    while True:

        if bot_id not in recent_histories:
            recent_histories[bot_id] = []

        user_input = input(f"\n\nYou to {bot_id}: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        print(f"\n")

        reply = chat_with_bot(bot_id, user_input, recent_histories[bot_id])
        print(f"{bot_id}: {reply}")

        # Update short-term memory (last 10 turns)
        recent_histories[bot_id].append(("user", user_input))
        recent_histories[bot_id].append(("assistant", reply))
        recent_histories[bot_id] = recent_histories[bot_id][-10:]
