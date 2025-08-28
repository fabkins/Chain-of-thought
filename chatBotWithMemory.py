import chromadb
from chromadb.utils import embedding_functions
import json
import uuid
from colorama import Fore, Style, init
import ollama

# Initialize colorama
init(autoreset=True)

# ---- Setup ----
chroma_client = chromadb.Client()

embedding_func = embedding_functions.DefaultEmbeddingFunction()

core_collection = chroma_client.get_or_create_collection(
    name="core_memory", embedding_function=embedding_func
)
event_collection = chroma_client.get_or_create_collection(
    name="event_memory", embedding_function=embedding_func
)


# ---- Fact Extraction ----
def extract_facts(bot_id, user_input, bot_reply):
    """
    Ask Ollama to extract structured facts from the last user+bot exchange.
    Returns a guaranteed schema JSON.
    """
    extractor_prompt = f"""
You are a fact extractor. 
Summarize the following conversation into structured facts. 
Facts must be categorized into:
- user_core: enduring facts about the user
- user_event: ephemeral or situational events about the user
- bot_core: enduring facts about the bot ({bot_id})
- bot_event: ephemeral or situational events about the bot

Conversation:
User: {user_input}
Bot ({bot_id}): {bot_reply}

Respond ONLY with valid JSON in this schema:
{{
  "user_core": [],
  "user_event": [],
  "bot_core": [],
  "bot_event": []
}}
"""

    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": extractor_prompt}])
    raw_output = response["message"]["content"]

    print("[Fact Extractor Output (raw)]:")
    print(raw_output)

    try:
        facts = json.loads(raw_output)
    except json.JSONDecodeError as e:
        print("[Fact Extraction Error] JSON parse failed:", e)
        facts = {}

    # Guarantee schema
    for key in ["user_core", "user_event", "bot_core", "bot_event"]:
        if key not in facts or not isinstance(facts[key], list):
            facts[key] = []

    print("\n[FACT-EXTRACTOR OUTPUT]")
    print(json.dumps(facts, indent=2))
    return facts


# ---- Memory Storage ----
def store_fact(bot_id, text, who="user", fact_type="core"):
    collection = core_collection if fact_type == "core" else event_collection
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[f"{who}:{text}"],
        metadatas={"bot_id": bot_id, "type": fact_type},
    )


def parse_and_store_facts(bot_id, user_input, bot_reply):
    facts = extract_facts(bot_id, user_input, bot_reply)

    # Store core memories
    for fact in facts["user_core"]:
        store_fact(bot_id, fact, who="user", fact_type="core")
    for fact in facts["bot_core"]:
        store_fact(bot_id, fact, who="bot", fact_type="core")

    # Store event memories
    for fact in facts["user_event"]:
        store_fact(bot_id, fact, who="user", fact_type="event")
    for fact in facts["bot_event"]:
        store_fact(bot_id, fact, who="bot", fact_type="event")


# ---- Bot Chat ----
def chat_with_bot(bot_id, user_input, history, allow_emotes=False):
    # Recall memories
    core_results = core_collection.query(
        query_texts=[user_input], n_results=3, where={"bot_id": bot_id}
    )
    event_results = event_collection.query(
        query_texts=[user_input], n_results=3, where={"bot_id": bot_id}
    )

    recalled_memories = []
    for r in core_results.get("documents", []):
        recalled_memories.extend(r)
    for r in event_results.get("documents", []):
        recalled_memories.extend(r)

    context = "\n".join(recalled_memories[-5:])
    system_prompt = f"""
You are {bot_id}. Stay in character.
Here are some of your memories:
{context}
User: {user_input}
"""

    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": system_prompt}])
    bot_reply = response["message"]["content"]

    # Optionally strip emotes like *smiles*
    if not allow_emotes:
        import re
        bot_reply = re.sub(r"\*.*?\*", "", bot_reply).strip()

    # Extract and store facts
    parse_and_store_facts(bot_id, user_input, bot_reply)

    return bot_reply


# ---- Main Loop ----
if __name__ == "__main__":
    bot_id = input("Talk to which bot? (e.g. isabella/kai): ").strip()
    history = []

    while True:
        user_input = input(Fore.BLUE + f"You to {bot_id}: " + Style.RESET_ALL)
        if user_input.lower() in ["quit", "exit"]:
            break

        reply = chat_with_bot(bot_id, user_input, history, allow_emotes=False)
        print(Fore.GREEN + f"{bot_id}: {reply}" + Style.RESET_ALL)
        history.append((user_input, reply))
