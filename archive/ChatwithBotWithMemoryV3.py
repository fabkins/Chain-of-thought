import ollama
import chromadb
import uuid, datetime, os, json
from colorama import Fore, Style, init
import re
from functools import lru_cache

"""
Multi-user memory chatbot with single-pass LLM flow and development-grade debugging.
- Single LLM call per turn returns BOTH the assistant reply and structured memory updates.
- Multi-user support via user_id metadata and retrieval filters.
- ChromaDB persistence per-bot collection. 
- Duplicate detection with tunable thresholds and verbose diagnostics.
- Optional emote stripping.

NOTE: This iteration intentionally omits contradiction handling and memory aging/pruning.
"""

# --- Init colorama ---
init(autoreset=True)

# --- Config ---
MODEL = "llama3.2"   # use same model for chat + embeddings
ALLOW_EMOTES = False  # set True to keep *emotes* in bot replies
CHROMA_PATH = "./chroma_db2"

# Duplicate detection tuning (distances come from Chroma; lower = closer for most metrics)
# You will see observed distances printed when storing to help calibrate these thresholds.
DUP_THRESHOLD_CORE = 0.12   # treat as duplicate if distance < threshold
DUP_THRESHOLD_EVENT = 0.10
TOP_K_RETRIEVE = 12

# --- Setup ChromaDB ---
os.makedirs(CHROMA_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_bot_collection(bot_id: str):
    """Return (or create) a ChromaDB collection for this bot."""
    # You can pass metadata, embedding function, and distance function here if desired.
    return chroma_client.get_or_create_collection(name=f"chat_{bot_id}")


# --- Embeddings (with simple in-memory cache) ---
@lru_cache(maxsize=8192)
def _embed_cached(model: str, text: str):
    return tuple(ollama.embeddings(model=model, prompt=text)["embedding"])  # tuples are cacheable


def embed_text(text: str):
    """Get an embedding using Ollama with local cache."""
    vec = _embed_cached(MODEL, text)
    return list(vec)  # Chroma expects list


# --- Memory Helpers ---

def store_fact(*, bot_id: str, user_id: str | None, content: str, target: str, mem_type: str, importance: float = 0.5):
    """Store a fact with duplicate checking and verbose diagnostics.

    Args:
        bot_id: Which bot's collection to use.
        user_id: The human user's id (None for bot memories).
        content: Full sentence fact.
        target: "user" or "bot".
        mem_type: "core" or "event".
        importance: 0..1 importance score.
    """
    collection = get_bot_collection(bot_id)

    print("\n" + Fore.YELLOW + "=== DUPLICATE CHECK ===" + Style.RESET_ALL)
    print(Fore.YELLOW + f"Target={target} Type={mem_type} UserID={user_id} Importance={importance}" + Style.RESET_ALL)
    print(Fore.YELLOW + f"Candidate: {content}" + Style.RESET_ALL)

    q_emb = embed_text(content)
    existing = collection.query(query_embeddings=[q_emb], where={"target": target} if target else {}, n_results=5)

    # Show distances for calibration
    docs = existing.get("documents", [[]])[0]
    dists = existing.get("distances", [[]])[0]
    metas = existing.get("metadatas", [[]])[0]

    for i, (doc, dist) in enumerate(zip(docs, dists)):
        m = metas[i] if i < len(metas) else {}
        print(Fore.YELLOW + f"vs[{i}] dist={dist:.4f} type={m.get('type')} target={m.get('target')} user={m.get('user_id')}\n    {doc}" + Style.RESET_ALL)

    # Threshold by memory type
    threshold = DUP_THRESHOLD_CORE if mem_type == "core" else DUP_THRESHOLD_EVENT
    is_dup = any((d is not None and d < threshold) for d in dists)

    if is_dup:
        print(Fore.RED + "SKIPPED STORE: considered duplicate by distance threshold" + Style.RESET_ALL)
        return None

    # Store
    doc_id = str(uuid.uuid4())
    metadata = {
        "type": mem_type,
        "timestamp": datetime.datetime.now().isoformat(),
        "importance": float(importance),
        "target": target,           # "user" or "bot"
        "user_id": user_id or "",  # empty for bot facts
    }

    collection.add(ids=[doc_id], embeddings=[q_emb], documents=[content], metadatas=[metadata])

    print("\n" + Fore.GREEN + "=== STORED NEW FACT ===" + Style.RESET_ALL)
    print(Fore.GREEN + f"ID: {doc_id}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Meta: {metadata}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Content: {content}" + Style.RESET_ALL)
    return doc_id


def retrieve_memories(*, bot_id: str, user_id: str | None, query: str, top_k: int = TOP_K_RETRIEVE):
    """Retrieve relevant memories in four buckets with filters and minimal calls."""
    collection = get_bot_collection(bot_id)

    print("\n" + Fore.BLUE + "=== MEMORY RETRIEVAL ===" + Style.RESET_ALL)
    print(Fore.BLUE + f"Query: {query}" + Style.RESET_ALL)

    # Build a single combined retrieval embedding for the user's input
    q_emb = embed_text(query)

    # Pull candidates broadly, then bucket client-side
    results = collection.query(query_embeddings=[q_emb], n_results=64)

    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    items = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else 999.0
        items.append((doc, dist, meta))

    # Bucket by (target, type) and apply filters
    def pick(bucket_filter):
        bucket = [
            (doc, dist) for (doc, dist, m) in items
            if bucket_filter(m, doc)
        ]
        bucket.sort(key=lambda x: x[1])  # lower distance first
        return [doc for doc, _ in bucket[:top_k]]

    user_core = pick(lambda m, doc: m.get("target") == "user" and m.get("type") == "core" and m.get("user_id") == (user_id or ""))
    user_events = pick(lambda m, doc: m.get("target") == "user" and m.get("type") == "event" and m.get("user_id") == (user_id or ""))
    bot_core = pick(lambda m, doc: m.get("target") == "bot" and m.get("type") == "core")
    bot_events = pick(lambda m, doc: m.get("target") == "bot" and m.get("type") == "event")

    print(Fore.CYAN + "User Core:" + Style.RESET_ALL, *[f"- {d}" for d in user_core[:6]], sep="\n")
    print(Fore.CYAN + "User Events:" + Style.RESET_ALL, *[f"- {d}" for d in user_events[:6]], sep="\n")
    print(Fore.CYAN + "Bot Core:" + Style.RESET_ALL, *[f"- {d}" for d in bot_core[:6]], sep="\n")
    print(Fore.CYAN + "Bot Events:" + Style.RESET_ALL, *[f"- {d}" for d in bot_events[:6]], sep="\n")

    return {
        "user_core": user_core,
        "user_events": user_events,
        "bot_core": bot_core,
        "bot_events": bot_events,
    }


# --- Single-pass LLM Orchestrator ---
SINGLE_PASS_SYSTEM = (
    "You are a helpful assistant that also outputs structured memory updates. "
    "Stay consistent with provided bot identity and user facts."
)

SINGLE_PASS_INSTRUCTIONS = r'''
You will produce TWO sections:

ASSISTANT_REPLY:
<free-form assistant reply in character>

MEMORY_JSON:
A JSON array where each item has fields:
- text: full-sentence fact
- target: "user" or "bot"
- type: "core" or "event"
- importance: number from 0 to 1

Rules:
- Extract only definite facts explicitly stated by the user or the bot in THIS turn, or evergreen bot traits explicitly stated.
- Use "target=user" ONLY for facts about the current user.
- Use "target=bot" for facts about the assistant.
- Keep facts short, atomic, and useful for future turns.
- Prefer marking identity/background as type=core; activities/offers/commitments as type=event.
- If no facts, output an empty array [].

Output format MUST follow exactly:
ASSISTANT_REPLY:
...your reply...

MEMORY_JSON:
[ ...valid JSON array... ]
'''


def build_single_pass_prompt(*, bot_core: list[str], bot_events: list[str], user_core: list[str], user_events: list[str], recent_history: list[tuple[str, str]], user_input: str):
    history_text = "\n".join([f"{role}: {msg}" for role, msg in recent_history[-10:]])
    prompt = f"""
{SINGLE_PASS_INSTRUCTIONS}

BOT_CORE_IDENTITY:\n{os.linesep.join(bot_core) if bot_core else 'None'}
BOT_RECENT_EVENTS:\n{os.linesep.join(bot_events) if bot_events else 'None'}
USER_CORE_FACTS:\n{os.linesep.join(user_core) if user_core else 'None'}
USER_RECENT_EVENTS:\n{os.linesep.join(user_events) if user_events else 'None'}

RECENT_CONVERSATION:\n{history_text if history_text else 'None'}

USER_MESSAGE:\n{user_input}
"""
    return prompt


def run_single_pass(*, bot_id: str, user_id: str, memories: dict, recent_history: list[tuple[str, str]], user_input: str, allow_emotes: bool):
    # 1) Build prompt
    prompt = build_single_pass_prompt(
        bot_core=memories["bot_core"],
        bot_events=memories["bot_events"],
        user_core=memories["user_core"],
        user_events=memories["user_events"],
        recent_history=recent_history,
        user_input=user_input,
    )

    # 2) Dev-level logging: show system + prompt
    print(Fore.MAGENTA + "\n=== SINGLE-PASS PROMPT (SYSTEM) ===" + Style.RESET_ALL)
    print(Fore.MAGENTA + SINGLE_PASS_SYSTEM + Style.RESET_ALL)
    print(Fore.MAGENTA + "\n=== SINGLE-PASS PROMPT (USER) ===" + Style.RESET_ALL)
    print(prompt)

    # 3) One chat call
    response = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": SINGLE_PASS_SYSTEM},
        {"role": "user", "content": prompt},
    ])

    raw = response["message"]["content"]
    print(Fore.MAGENTA + "\n=== SINGLE-PASS RAW RESPONSE ===" + Style.RESET_ALL)
    print(raw)

    # 4) Parse sections
    reply = ""
    memory_json = []

    # Extract ASSISTANT_REPLY section
    m = re.search(r"ASSISTANT_REPLY:\n(.*?)\n\s*MEMORY_JSON:\n", raw, flags=re.DOTALL)
    if m:
        reply = m.group(1).strip()
    else:
        # Fallback: if format slightly off, take up to JSON marker or all
        parts = raw.split("MEMORY_JSON:")
        reply = parts[0].replace("ASSISTANT_REPLY:", "").strip()

    # Extract MEMORY_JSON array
    j = re.search(r"MEMORY_JSON:\n(.*)$", raw, flags=re.DOTALL)
    if j:
        json_blob = j.group(1).strip()
        # Try to isolate JSON array
        jm = re.search(r"\[.*\]", json_blob, flags=re.DOTALL)
        if jm:
            json_text = jm.group(0)
            try:
                memory_json = json.loads(json_text)
            except Exception as e:
                print(Fore.RED + f"JSON parse error: {e}" + Style.RESET_ALL)
                print(Fore.YELLOW + "Attempted JSON:" + Style.RESET_ALL)
                print(json_text)
                memory_json = []
        else:
            print(Fore.RED + "No JSON array found in MEMORY_JSON section" + Style.RESET_ALL)
    else:
        print(Fore.RED + "MEMORY_JSON section missing" + Style.RESET_ALL)

    # 5) Post-process reply (strip emotes if configured)
    if not allow_emotes:
        reply = re.sub(r"\*.*?\*", "", reply).strip()

    # 6) Store memories
    print(Fore.MAGENTA + "\n=== MEMORY ITEMS TO STORE ===" + Style.RESET_ALL)
    for i, item in enumerate(memory_json):
        print(Fore.MAGENTA + f"Item[{i}]: {item}" + Style.RESET_ALL)
        try:
            text = str(item.get("text", "")).strip()
            target = item.get("target", "").lower()
            mtype = item.get("type", "").lower()
            importance = float(item.get("importance", 0.6))

            if not text or target not in {"user", "bot"} or mtype not in {"core", "event"}:
                print(Fore.RED + f"SKIP invalid memory item: {item}" + Style.RESET_ALL)
                continue

            # Enforce user ownership on user facts
            uid = user_id if target == "user" else None

            store_fact(bot_id=bot_id, user_id=uid, content=text, target=target, mem_type=mtype, importance=importance)
        except Exception as e:
            print(Fore.RED + f"ERROR storing item[{i}]: {e}" + Style.RESET_ALL)

    return reply


# --- Public Chat API ---
def chat_with_bot(*, bot_id: str, user_id: str, user_input: str, recent_history: list[tuple[str, str]], allow_emotes: bool = ALLOW_EMOTES):
    memories = retrieve_memories(bot_id=bot_id, user_id=user_id, query=user_input)

    reply = run_single_pass(
        bot_id=bot_id,
        user_id=user_id,
        memories=memories,
        recent_history=recent_history,
        user_input=user_input,
        allow_emotes=allow_emotes,
    )

    return reply


# --- Seeding ---
def seed_bot(bot_id: str, backstory_facts: list[str]):
    for fact in backstory_facts:
        store_fact(bot_id=bot_id, user_id=None, content=fact, target="bot", mem_type="core", importance=1.0)


def seed_example_bots():
    seed_bot("isabella", [
        "My name is Isabella Rossi.",
        "I was born on April 3rd, 1985, in Florence, Italy.",
        "I studied literature at the University of Bologna.",
        "I worked as a librarian before becoming a travel writer.",
        "I am thoughtful, friendly, and love Renaissance art.",
    ])

    seed_bot("kai", [
        "My name is Kai Tanaka.",
        "I was born on November 12th, 1992, in Osaka, Japan.",
        "I used to be a robotics engineer before opening a coffee shop.",
        "I own a coffee shop called 'Cyber Brew'.",
        "My coffee shop is located in Osaka, Japan.",
        "I am curious, innovative, and enjoy cyberpunk novels.",
    ])


def database_is_empty():
    return len(chroma_client.list_collections()) == 0


# --- CLI for demo ---
if __name__ == "__main__":
    print("Multi-user Memory Chatbot ready! Type 'quit' to exit.\n")

    if database_is_empty():
        print("No database found, seeding example bots...")
        seed_example_bots()

    recent_histories: dict[tuple[str, str], list[tuple[str, str]]] = {}

    # Choose bot
    bot_id = input("Talk to which bot? (e.g. isabella/kai): ").strip() or "kai"
    if bot_id.lower() in ["quit", "exit"]:
        raise SystemExit(0)

    # Provide a user id (multi-user support)
    user_id = input("Enter your user_id (any string): ").strip() or "default-user"

    while True:
        key = (bot_id, user_id)
        if key not in recent_histories:
            recent_histories[key] = []

        user_input = input(Fore.YELLOW + f"\nYou[{user_id}] -> {bot_id}: " + Style.RESET_ALL)
        if user_input.lower() in ["quit", "exit"]:
            break

        reply = chat_with_bot(
            bot_id=bot_id,
            user_id=user_id,
            user_input=user_input,
            recent_history=recent_histories[key],
            allow_emotes=ALLOW_EMOTES,
        )

        print(Fore.GREEN + f"\n{bot_id}: {reply}" + Style.RESET_ALL)

        # Update short-term memory (last 10 turns per (bot,user))
        recent_histories[key].append(("user", user_input))
        recent_histories[key].append(("assistant", reply))
        recent_histories[key] = recent_histories[key][-10:]
