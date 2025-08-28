import ollama
import chromadb
import uuid, datetime, os
from colorama import Fore, Style, init

# --- Init colorama ---
init(autoreset=True)

# --- Config ---
MODEL = "llama3.2"   # use same model for chat + embeddings

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

def store_fact(bot_id, content, mem_type="event", importance=0.5):
    collection = get_bot_collection(bot_id)
    doc_id = str(uuid.uuid4())
    collection.add(
        ids=[doc_id],
        embeddings=[embed_text(content)],
        documents=[content],
        metadatas=[{
            "type": mem_type,
            "timestamp": str(datetime.datetime.now()),
            "importance": importance
        }]
    )
    print(Fore.CYAN + f"[Memory Stored:{bot_id.upper()}] ({mem_type.upper()}) {content}" + Style.RESET_ALL)

def retrieve_memories(bot_id, query, n_results=5):
    """Retrieve relevant memories for this bot (core + events)."""
    collection = get_bot_collection(bot_id)
    
    # Always get core memories first
    core_results = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=n_results,
        where={"type": "core"}
    )
    
    # Then get relevant event memories
    event_results = collection.query(
        query_embeddings=[embed_text(query)],
        n_results=n_results,
        where={"type": "event"}
    )
    
    # Combine results, prioritizing core memories
    memories = core_results.get("documents", [[]])[0] + event_results.get("documents", [[]])[0]
    return memories

# --- Summarization + Classification ---
def summarize_interaction(user_input, ai_reply):
    """Extract only explicit factual info from the exchange."""
    prompt = f"""
Extract ALL clear factual statements from this interaction. For each statement:
1. Start with WHO the fact is about ("The user" or "The AI")
2. State the fact clearly and completely
3. Include context if directly stated

CAREFULLY CHECK FOR:

1. NAMES & IDENTITY:
   - Names and titles ("My name is X", "I am X", "I'm X")
   - Roles and professions ("I am a X", "I work as X")
   - Family relationships ("X is my Y")

2. OWNERSHIP & POSSESSION:
   - Current ownership ("I have X", "I own X")
   - Business ownership ("My shop", "My store", "My company")
   - Property ("My house", "My car")

3. HISTORY & PAST:
   - Past experiences ("I used to X")
   - Previous ownership ("I had X")
   - Past events ("When I was X")
   - Previous jobs ("I worked as X")
   - Memories ("reminds me of when")

4. CURRENT STATUS:
   - Present occupation
   - Current location
   - Current activities
   - Business details
   - Living situation

5. PERSONAL TRAITS:
   - Preferences ("I like X", "I love X")
   - Skills ("I can X", "I know how to X")
   - Habits ("I always X", "I never X")
   - Beliefs ("I believe X", "I think X")
   - Feelings ("I feel X", "I am X")
   - Hobbies ("I enjoy X")

6. RELATIONSHIPS & CONNECTIONS:
   - Social connections ("I know X")
   - Professional relationships ("I work with X")
   - Personal relationships ("I am friends with X")

Format each fact as a complete sentence (e.g., "The user used to own a tea house in Japan").
If no clear facts are present, reply exactly: NO FACT.

Format each fact as a complete sentence starting with "The user" or "The AI".
If no clear facts are present, reply exactly: NO FACT.

User: {user_input}
AI: {ai_reply}

Facts:"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

def classify_memory(summary):
    """Decide if the summary is core/event, and whether it's about bot or user."""
    if summary == "NO FACT":
        return None, None

    prompt = f"""
Classify this fact STRICTLY:
"{summary}"

Rules:
1. CORE facts (always mark as core):
   - Names ("My name is X", "I am X")
   - Past experiences ("I used to have/be X")
   - Personal history ("I worked as X")
   - Permanent traits ("I love X", "I enjoy X")
   
2. EVENT facts:
   - Current actions ("I am drinking")
   - Temporary states ("I feel tired")
   - One-time occurrences

3. Target:
   - If fact starts with "The user" or contains user's name -> USER
   - If fact starts with "The AI" or contains bot's name -> BOT

Answer ONLY as: "<core/event>,<bot/user>".
"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    result = response["message"]["content"].strip().lower()

    mem_type = "core" if "core" in result else "event"
    target = "user" if "user" in result else "bot"
    return mem_type, target

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

    # --- Summarize + classify ---
    summary = summarize_interaction(user_input, reply)
    mem_type, target = classify_memory(summary)

    if summary != "NO FACT" and mem_type:
        store_fact(bot_id, summary, mem_type=mem_type, importance=1.0 if mem_type == "core" else 0.5)

    return reply

# --- Seed Bots ---
def seed_bot(bot_id, backstory_facts):
    for fact in backstory_facts:
        store_fact(bot_id, fact, mem_type="core", importance=1.0)

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
        "I own a coffee shop called 'Cyber Brew'.",
        "My coffee shop is located in Osaka, Japan.",
        "I am curious, innovative, and enjoy cyberpunk novels."
    ])

def database_is_empty():
    return len(chroma_client.list_collections()) == 0

# --- Interactive Chat Loop ---
if __name__ == "__main__":
    print("Multi-Bot Chatbot ready! Type 'quit' to exit.\n")

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

        user_input = input(Fore.YELLOW + f"\nYou to {bot_id}: " + Style.RESET_ALL)
        if user_input.lower() in ["quit", "exit"]:
            break

        reply = chat_with_bot(bot_id, user_input, recent_histories[bot_id])

        # separation + AI reply in green
        print(Fore.GREEN + f"\n{bot_id}: {reply}" + Style.RESET_ALL)

        # Update short-term memory (last 10 turns)
        recent_histories[bot_id].append(("user", user_input))
        recent_histories[bot_id].append(("assistant", reply))
        recent_histories[bot_id] = recent_histories[bot_id][-10:]
