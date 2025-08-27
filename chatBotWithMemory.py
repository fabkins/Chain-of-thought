import ollama
import chromadb
import uuid, datetime
import os
from colorama import Fore, Style, init

# --- Config ---
MODEL = "llama3.2"   # use same model for chat + embeddings
DEBUG = True         # set to False to hide memory/classifier logs

# --- Setup Colorama ---
init(autoreset=True)

# --- Setup ChromaDB ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def debug_log(msg):
    """Print debug messages only if DEBUG is enabled."""
    if DEBUG:
        print(Fore.CYAN + msg + Style.RESET_ALL)

def get_bot_collection(bot_id):
    """Return (or create) a ChromaDB collection for this bot."""
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
    debug_log(f"[MEMORY] Stored CORE fact for {bot_id}: {content}")

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
    debug_log(f"[MEMORY] Stored EVENT for {bot_id}: {content}")

def retrieve_memories(bot_id, query, n_results=5):
    """Retrieve relevant memories for this bot (core + events)."""
    collection = get_bot_collection(bot_id)
    results = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=n_results
    )
    return results.get("documents", [[]])[0]

# --- Classification Helper ---
def classify_fact(user_input):
    """
    Ask the model if this user statement should be stored as a core memory (fact) 
    or just an event. Returns 'core' or 'event'.
    """
    system_prompt = """You are a memory classifier.
Decide if a user statement should be stored as:
- CORE FACT: enduring, important, personal detail (e.g. name, birthday, favorite things, relationships, jobs, places lived).
- EVENT: temporary or casual statements, greetings, chit-chat.

Answer with only 'core' or 'event'."""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
    )
    classification = response["message"]["content"].strip().lower()
    debug_log(f"[CLASSIFIER] Input classified as: {classification.upper()}")
    return classification

# --- Chat Function ---
def chat_with_bot(bot_id, user_input, recent_history):
    # 1. Classify and store user input
    classification = classify_fact(user_input)
    if classification == "core":
        store_core_fact(bot_id, f"User fact: {user_input}")
    else:
        store_event(bot_id, f"Conversation: {user_input}", importance=0.5)

    # 2. Retrieve relevant memories
    memories = retrieve_memories(bot_id, user_input)
    history = "\n".join([f"{role}: {msg}" for role, msg in recent_history])
    context = "\n".join(memories)

    # 3. Build prompt
    prompt = f"""
You are a unique character with personality and evolving history.

Core personality and relevant memories:
{context}

Recent conversation:
{history}

User: {user_input}
Assistant (in character):"""

    # 4. Generate bot reply
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    reply = response["message"]["content"]

    # 5. Store assistant reply as an event
    store_event(bot_id, f"Bot reply: {reply}", importance=0.4)

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
        exit(0)

    while True:
        if bot_id not in recent_histories:
            recent_histories[bot_id] = []

        user_input = input(Fore.YELLOW + f"\n\nYou to {bot_id}: " + Style.RESET_ALL)
        if user_input.lower() in ["quit", "exit"]:
            break

        reply = chat_with_bot(bot_id, user_input, recent_histories[bot_id])
        print(Fore.GREEN + f"\n{bot_id}: {reply}" + Style.RESET_ALL)

        # Update short-term memory (last 10 turns)
        recent_histories[bot_id].append(("user", user_input))
        recent_histories[bot_id].append(("assistant", reply))
        recent_histories[bot_id] = recent_histories[bot_id][-10:]
