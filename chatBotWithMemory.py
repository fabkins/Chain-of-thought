import chromadb
import uuid
import json
import datetime
import re
from colorama import Fore, Style, init
import ollama

# --- Initialize Colorama ---
init(autoreset=True)

# --- Configuration ---
MODEL = "llama3.2"
ALLOW_EMOTES = False  # set True to keep *emotes* in bot replies
MEMORY_RESULTS = 5    # number of memories to retrieve for context

# --- Setup ChromaDB ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")

core_collection = chroma_client.get_or_create_collection("core_memory")
event_collection = chroma_client.get_or_create_collection("event_memory")


# --- Embeddings Helper ---
def embed_text(text: str):
    """Return embedding using Ollama."""
    try:
        return ollama.embeddings(model=MODEL, prompt=text)["embedding"]
    except Exception as e:
        print(Fore.CYAN + f"[Embedding Error] {e}" + Style.RESET_ALL)
        return None


# --- Fact Extraction ---
def extract_facts(bot_id, user_id, user_input, bot_reply):
    """Ask Ollama to extract structured facts from the last exchange."""
    prompt = f"""
You are a fact extractor.
Summarize the conversation below into structured JSON facts.
Categorize facts as:
- user_core: enduring facts about the user
- user_event: ephemeral/situational events about the user
- bot_core: enduring facts about the bot ({bot_id})
- bot_event: ephemeral/situational events about the bot

Conversation:
User: {user_input}
Bot ({bot_id}): {bot_reply}

Respond ONLY with valid JSON using this schema:
{{
  "user_core": [],
  "user_event": [],
  "bot_core": [],
  "bot_event": []
}}
"""
    try:
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        raw_output = response["message"]["content"]
        print(Fore.CYAN + "[Fact Extractor Output (raw)]:\n" + raw_output + Style.RESET_ALL)
        facts = json.loads(raw_output)
    except Exception as e:
        print(Fore.CYAN + f"[Fact Extraction Error] {e}" + Style.RESET_ALL)
        facts = {"user_core": [], "user_event": [], "bot_core": [], "bot_event": []}

    # Guarantee schema
    for key in ["user_core", "user_event", "bot_core", "bot_event"]:
        if key not in facts or not isinstance(facts[key], list):
            facts[key] = []
    print(Fore.CYAN + "[FACT-EXTRACTOR OUTPUT]\n" + json.dumps(facts, indent=2) + Style.RESET_ALL)
    return facts


# --- Memory Storage ---
def store_fact(bot_id, user_id, text, who="user", fact_type="core"):
    """Store a single fact in the appropriate collection."""
    emb = embed_text(text)
    if emb is None:
        print(Fore.CYAN + f"[Skipping fact storage: embedding failed] {text}" + Style.RESET_ALL)
        return
    collection = core_collection if fact_type == "core" else event_collection
    collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[emb],
        documents=[text],
        metadatas=[{"bot_id": bot_id, "user_id": user_id, "type": fact_type, "who": who, "timestamp": str(datetime.datetime.now())}]
    )
    print(Fore.CYAN + f"[Memory Stored:{who.upper()}] ({fact_type.upper()}) {text}" + Style.RESET_ALL)


def parse_and_store_facts(bot_id, user_id, user_input, bot_reply):
    facts = extract_facts(bot_id, user_id, user_input, bot_reply)

    # Store core facts
    for fact in facts["user_core"]:
        store_fact(bot_id, user_id, fact, who="user", fact_type="core")
    for fact in facts["bot_core"]:
        store_fact(bot_id, user_id, fact, who="bot", fact_type="core")

    # Store events
    for fact in facts["user_event"]:
        store_fact(bot_id, user_id, fact, who="user", fact_type="event")
    for fact in facts["bot_event"]:
        store_fact(bot_id, user_id, fact, who="bot", fact_type="event")


# --- Memory Recall ---
def recall_memories(bot_id, user_id, query, n_results=MEMORY_RESULTS):
    """Retrieve top relevant memories for a given query."""
    emb = embed_text(query)
    if emb is None:
        return []

    core_results = core_collection.query(
        query_embeddings=[emb],
        n_results=n_results,
        where={"bot_id": bot_id, "user_id": user_id}
    )
    event_results = event_collection.query(
        query_embeddings=[emb],
        n_results=n_results,
        where={"bot_id": bot_id, "user_id": user_id}
    )

    memories = []
    for r in core_results.get("documents", [[]])[0]:
        memories.append(r)
    for r in event_results.get("documents", [[]])[0]:
        memories.append(r)
    return memories


# --- Chat Function ---
def chat_with_bot(bot_id, user_id, user_input, history, allow_emotes=ALLOW_EMOTES):
    # Recall memories
    recalled = recall_memories(bot_id, user_id, user_input)
    context = "\n".join(recalled[-MEMORY_RESULTS:])

    # Include short-term history
    history_text = "\n".join([f"{role}: {msg}" for role, msg in history[-10:]])

    system_prompt = f"""
You are {bot_id}, a character with personality and evolving memories.

Core and relevant memories:
{context}

Recent conversation:
{history_text}

User: {user_input}
Assistant (in character):"""

    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": system_prompt}])
    bot_reply = response["message"]["content"]

    # Optionally strip emotes like *smiles*
    if not allow_emotes:
        bot_reply = re.sub(r"\*.*?\*", "", bot_reply).strip()

    # Extract and store new facts
    parse_and_store_facts(bot_id, user_id, user_input, bot_reply)

    return bot_reply


# --- Main Interactive Loop ---
if __name__ == "__main__":
    bot_id = input(Fore.YELLOW + "Talk to which bot? (e.g. isabella/kai): " + Style.RESET_ALL).strip()
    user_id = input(Fore.YELLOW + "Your user ID/name: " + Style.RESET_ALL).strip()
    history = []

    print(Fore.CYAN + f"Starting chat with bot '{bot_id}' as user '{user_id}'..." + Style.RESET_ALL)

    while True:
        user_input = input(Fore.BLUE + f"You to {bot_id}: " + Style.RESET_ALL)
        if user_input.lower() in ["quit", "exit"]:
            break

        reply = chat_with_bot(bot_id, user_id, user_input, history, allow_emotes=ALLOW_EMOTES)

        print("\n" + Fore.GREEN + f"{bot_id}: {reply}" + Style.RESET_ALL)

        # Update short-term memory
        history.append(("user", user_input))
        history.append(("assistant", reply))
        history = history[-10:]
