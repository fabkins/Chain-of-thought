# chatBotWithMemory.py
import os
import uuid
import datetime
import json
import re
import chromadb
from colorama import Fore, Style, init
import ollama

# ---------- CONFIG ----------
# Set your chat and embed model names here (default to the same model to avoid "model not found")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.2")
EMBED_MODEL = os.environ.get("EMBED_MODEL", CHAT_MODEL)

DB_DIR = "./chroma_db"            # persistent Chroma directory
DEBUG = True                      # set False to reduce logs
SUPPRESS_EMOTES = True            # strip *emotes* like *smiles* from AI output
RETRIEVE_K = 4                    # how many similar memories to retrieve per side

# colors
init(autoreset=True)

# ---------- ChromaDB client ----------
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR, exist_ok=True)
client = chromadb.PersistentClient(path=DB_DIR)

# ---------- Helpers: collections ----------
def get_collection(bot_id: str, target: str):
    """
    target: 'bot' or 'user'
    collection name pattern: chat_<bot>_bot or chat_<bot>_user
    """
    name = f"chat_{bot_id}_{target}"
    # Explicitly do NOT supply an embedding_function here — we'll provide embeddings ourselves
    return client.get_or_create_collection(name=name)

def list_collections():
    try:
        return client.list_collections()
    except Exception:
        # fallback: list dirs (best-effort)
        try:
            return os.listdir(DB_DIR)
        except Exception:
            return []

def database_is_empty():
    cols = list_collections()
    return len(cols) == 0

# ---------- Embeddings ----------
def embed_text(text: str):
    """Return embedding vector or None on failure."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        emb = resp.get("embedding")
        if not emb:
            if DEBUG:
                print(Fore.RED + "[Embedding Error] model returned no embedding" + Style.RESET_ALL)
            return None
        return emb
    except Exception as e:
        if DEBUG:
            print(Fore.RED + f"[Embedding Error] {e}" + Style.RESET_ALL)
        return None

# ---------- Memory storage ----------
def store_fact(bot_id: str, fact: str, who: str = "user", fact_type: str = "core"):
    """
    who: 'user' or 'bot'
    fact_type: 'core' or 'event'
    """
    fact = (fact or "").strip()
    if not fact:
        return

    emb = embed_text(fact)
    if emb is None:
        # embedding failed; skip storing to avoid collection.add errors
        if DEBUG:
            print(Fore.CYAN + f"[Memory SKIP] Could not embed fact, skipping storage: {fact}" + Style.RESET_ALL)
        return

    coll = get_collection(bot_id, who)
    doc_id = str(uuid.uuid4())
    coll.add(
        ids=[doc_id],
        embeddings=[emb],
        documents=[fact],
        metadatas=[{
            "who": who,
            "type": fact_type,
            "timestamp": str(datetime.datetime.now())
        }]
    )
    if DEBUG:
        print(Fore.CYAN + f"[Memory Stored:{bot_id.upper()}] ({fact_type.upper()}) [{who.upper()}] {fact}" + Style.RESET_ALL)

# ---------- Retrieval ----------
def retrieve_relevant(bot_id: str, query: str, target: str = "bot", n_results: int = RETRIEVE_K):
    """Return a list of retrieved documents (may be empty)."""
    coll = get_collection(bot_id, target)
    emb = embed_text(query)
    if emb is None:
        # can't query without an embedding
        if DEBUG:
            print(Fore.CYAN + f"[Retrieve SKIP] embedding failed; returning no memories for {target}" + Style.RESET_ALL)
        return []
    try:
        results = coll.query(query_embeddings=[emb], n_results=n_results)
        docs = results.get("documents", [[]])[0] if results.get("documents") else []
        return docs or []
    except Exception as e:
        if DEBUG:
            print(Fore.RED + f"[Retrieve Error] {e}" + Style.RESET_ALL)
        return []

def retrieve_core_docs(bot_id: str, target: str = "bot"):
    """Always include core facts (non-retrieval-based) to keep identity stable."""
    coll = get_collection(bot_id, target)
    try:
        res = coll.get(where={"type": "core"})
        return res.get("documents", []) if res else []
    except Exception:
        return []

# ---------- Fact extraction (JSON) ----------
def extract_facts_json(user_input: str, bot_reply: str):
    """
    Ask the model to return strict JSON with arrays:
    {"user_core": [], "user_event": [], "bot_core": [], "bot_event": []}
    """
    prompt = f"""
You will extract explicit factual statements from the short dialogue below.
Return ONLY valid JSON with exactly these four keys:
{{"user_core": [], "user_event": [], "bot_core": [], "bot_event": []}}

Rules:
- Include only facts that were explicitly said in the dialogue. Do NOT invent, infer, or expand.
- Put stable identity/history facts (e.g., jobs, places lived, family, previous ownership) in *_core.
- Put situational/temporary/one-off details in *_event.
- Each fact should be a concise neutral sentence (no opinions, no emotive asides).
- If no facts for a category, return an empty array for that key.

Dialogue:
User: {user_input}
Bot: {bot_reply}

Return only the JSON object.
"""
    try:
        resp = ollama.chat(model=CHAT_MODEL, messages=[{"role":"user","content":prompt}])
        txt = resp["message"]["content"].strip()

        # Isolate first {...} block in the response
        start = txt.find("{")
        end = txt.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON found in extractor output")
        json_text = txt[start:end+1]

        data = json.loads(json_text)
        # Ensure keys exist
        for k in ("user_core", "user_event", "bot_core", "bot_event"):
            if k not in data or not isinstance(data[k], list):
                data[k] = []
        return data
    except Exception as e:
        if DEBUG:
            print(Fore.RED + f"[Fact Extraction Error] {e}" + Style.RESET_ALL)
            print(Fore.CYAN + "[Fact Extractor Output (raw)]:", flush=True)
            try:
                # show the raw model output for debugging
                resp = ollama.chat(model=CHAT_MODEL, messages=[{"role":"user","content":prompt}])
                print(resp["message"]["content"])
            except Exception as e2:
                print(Fore.RED + f"[Fact Extractor Retry Error] {e2}" + Style.RESET_ALL)
        # Fallback: return empty lists
        return {"user_core": [], "user_event": [], "bot_core": [], "bot_event": []}

# ---------- Seeding example bots ----------
def seed_bot(bot_id: str, backstory_facts: list[str]):
    for fact in backstory_facts:
        store_fact(bot_id, fact, who="bot", fact_type="core")

def seed_example_bots():
    # Example backstories (run once)
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

# ---------- Chat flow ----------
def build_context(bot_id: str, user_input: str, recent_history: list[tuple]):
    # Always include core facts (bot and user)
    bot_core = retrieve_core_docs(bot_id, target="bot")
    user_core = retrieve_core_docs(bot_id, target="user")

    # Retrieve relevant events (semantic)
    bot_events = retrieve_relevant(bot_id, user_input, target="bot", n_results=RETRIEVE_K)
    user_events = retrieve_relevant(bot_id, user_input, target="user", n_results=RETRIEVE_K)

    pieces = []
    if bot_core:
        pieces.append("Bot core facts:\n" + "\n".join(bot_core))
    if user_core:
        pieces.append("User core facts:\n" + "\n".join(user_core))
    if bot_events:
        pieces.append("Bot relevant events:\n" + "\n".join(bot_events))
    if user_events:
        pieces.append("User relevant events:\n" + "\n".join(user_events))

    # Recent short-term dialogue
    history = "\n".join([f"{role}: {msg}" for role, msg in recent_history])
    if history:
        pieces.append("Recent conversation:\n" + history)

    return "\n\n".join(pieces)

def chat_with_bot(bot_id: str, user_input: str, recent_history: list[tuple], allow_emotes: bool = True):
    # Build context + prompt
    context = build_context(bot_id, user_input, recent_history)
    system_msg = f"You are roleplaying as {bot_id}. Stay in character and be consistent with your core facts."

    # If we want to avoid emotes, tell the model
    user_instructions = ("Avoid emotive stage directions like *smiles*, *laughs*, or actions unless the user asks for roleplay."
                         if not allow_emotes else "You may use light emotive stage directions when expressive cues help.")

    prompt = f"""{user_instructions}

Context (from memory and recent chat):
{context}

User: {user_input}
Bot:"""

    # Send system + user messages
    try:
        resp = ollama.chat(model=CHAT_MODEL, messages=[{"role":"system","content":system_msg}, {"role":"user","content":prompt}])
        reply = resp["message"]["content"].strip()
    except Exception as e:
        print(Fore.RED + f"[Chat Error] {e}" + Style.RESET_ALL)
        return "Sorry — model error."

    # Optionally suppress emotes for display
    display_reply = reply
    if SUPPRESS_EMOTES and not allow_emotes:
        display_reply = re.sub(r"\*[^*]+\*", "", display_reply).strip()

    # Extract facts (JSON) and debug-print them
    fact_data = extract_facts_json(user_input, reply)
    if DEBUG:
        print(Fore.CYAN + "[FACT-EXTRACTOR OUTPUT]" + Style.RESET_ALL)
        print(json.dumps(fact_data, indent=2))
    # Store facts
    for f in fact_data.get("user_core", []):
        store_fact(bot_id, f, who="user", fact_type="core")
    for f in fact_data.get("user_event", []):
        store_fact(bot_id, f, who="user", fact_type="event")
    for f in fact_data.get("bot_core", []):
        store_fact(bot_id, f, who="bot", fact_type="core")
    for f in fact_data.get("bot_event", []):
        store_fact(bot_id, f, who="bot", fact_type="event")

    # Print conversation with colors and separation
    print(Fore.YELLOW + f"\nYou to {bot_id}: {user_input}" + Style.RESET_ALL)
    print(Fore.GREEN + f"\n{bot_id}: {display_reply}\n" + Style.RESET_ALL)

    return reply

# ---------- Main ----------
if __name__ == "__main__":
    # Auto-seed if DB appears empty
    if database_is_empty():
        print(Fore.CYAN + "No database collections found — seeding example bots..." + Style.RESET_ALL)
        seed_example_bots()

    print(Fore.CYAN + "Multi-Bot Chatbot ready! Type 'quit' to exit." + Style.RESET_ALL)

    recent_histories = {}

    bot_id = input(Fore.YELLOW + "\nTalk to which bot? (e.g. isabella/kai): " + Style.RESET_ALL).strip()
    if not bot_id:
        print("No bot selected — exiting.")
        raise SystemExit(0)

    while True:
        if bot_id not in recent_histories:
            recent_histories[bot_id] = []

        user_input = input(Fore.YELLOW + f"\nYou to {bot_id}: " + Style.RESET_ALL)
        if user_input.lower() in ["quit", "exit"]:
            break

        # Chat & memory
        reply = chat_with_bot(bot_id, user_input, recent_histories[bot_id], allow_emotes=not SUPPRESS_EMOTES)

        # update short term memory buffer (last 10 items as (role,msg))
        recent_histories[bot_id].append(("user", user_input))
        recent_histories[bot_id].append(("assistant", reply))
        recent_histories[bot_id] = recent_histories[bot_id][-10:]

