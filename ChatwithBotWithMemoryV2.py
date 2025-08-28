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
    """Store a fact, but check for duplicates first."""
    collection = get_bot_collection(bot_id)
    print("\n" + Fore.YELLOW + "=== CHECKING FOR DUPLICATES ===" + Style.RESET_ALL)
    
    # Check for similar existing memories to avoid duplicates
    existing = collection.query(
        query_embeddings=[embed_text(content)],
        n_results=3  # Check a few closest matches
    )
    
    # If we find a very similar memory, skip storing
    if existing["documents"] and existing["documents"][0]:
        # Compare the new fact with existing ones
        for doc, score in zip(existing["documents"][0], existing["distances"][0]):
            print(Fore.YELLOW + f"Comparing with existing fact (similarity={score:.3f}):" + Style.RESET_ALL)
            print(f"  New     : {content}")
            print(f"  Existing: {doc}")
            # Using a more lenient threshold for duplicates
            if score < 0.001:  # Much stricter comparison required for skipping
                print(Fore.RED + "SKIPPED: Too similar to existing fact" + Style.RESET_ALL)
                return
    
    # If we reach here, this is a new fact worth storing
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
    print("\n" + Fore.GREEN + "=== STORING NEW FACT ===" + Style.RESET_ALL)
    print(Fore.GREEN + f"ID: {doc_id}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Type: {mem_type}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Content: {content}" + Style.RESET_ALL)

def retrieve_memories(bot_id, query, n_results=5):
    """Retrieve relevant memories for this bot (core + events)."""
    collection = get_bot_collection(bot_id)
    
    print("\n" + Fore.BLUE + "=== MEMORY RETRIEVAL ===" + Style.RESET_ALL)
    print(Fore.BLUE + f"Query: {query}" + Style.RESET_ALL)
    
    # Create compound queries to get both general context and specific responses
    queries = [
        "Tell me about the user's background and history",  # Gets identity and history
        "What are the user's preferences and traits",       # Gets personality and preferences
        query,                                             # Gets query-specific memories
        "What significant facts do we know about the user"  # Gets other important facts
    ]
    
    # Query for both core and event memories
    all_memories = []
    seen = set()
    
    for q in queries:
        print("\n" + Fore.CYAN + f"Searching with query: {q}" + Style.RESET_ALL)
        # Get both core and regular memories for each query
        results = collection.query(
            query_embeddings=[embed_text(q)],
            n_results=10  # Increased from 5 to get more context
        )
        
        # Add unique memories while preserving order
        for memory, dist in zip(results.get("documents", [[]])[0], results.get("distances", [[]])[0]):
            if memory not in seen:
                print(f"  Found (similarity={dist:.3f}): {memory}")
                all_memories.append(memory)
                seen.add(memory)
            else:
                print(f"  Skipped duplicate: {memory}")
    
    print("\n" + Fore.BLUE + "=== FINAL MEMORY SET ===" + Style.RESET_ALL)
    for i, memory in enumerate(all_memories, 1):
        print(f"{i}. {memory}")
    
    return all_memories

# --- Summarization + Classification ---
def summarize_interaction(user_input, ai_reply):
    """Extract only explicit factual info from the exchange."""
    prompt = f"""
You are a fact extraction expert. Extract ONLY facts that were EXPLICITLY stated by the user. Ignore the AI's response completely.

RULES:
1. Only extract facts that someone directly stated about themselves or others
2. Do not extract facts from roleplay actions (e.g., *smiles*, *waves*)
3. Do not make inferences or assumptions
4. Do not include temporary states or actions
5. Ignore greetings and casual conversation
6. Pay special attention to extracting background info:
   - Past locations where someone lived
   - Previous businesses or jobs
   - Past experiences and history
   - Personal possessions (past or present)

EXAMPLES:
When someone says: "I used to live in London" 
✓ EXTRACT: "The user used to live in London"

When someone says: "it reminds me of a tea house I owned when I lived in Chester"
✓ EXTRACT: "The user used to own a tea house in Chester"
✓ EXTRACT: "The user used to live in Chester"

When someone says: "*smiles* I love coffee" 
✓ EXTRACT: "The user loves coffee"
✗ DO NOT EXTRACT: "The user smiled"

When someone says: "You must be from Japan"
✗ DO NOT EXTRACT THIS - it's an assumption

When someone says: "Let me get you some coffee"
✗ DO NOT EXTRACT THIS - it's a temporary action

2. History & Experience:
   - Past occupations or roles
   - Previous locations
   - Past ownership or possessions
   - Significant experiences

3. Current Status:
   - Occupation or role
   - Location
   - Property or possessions
   - Business details

4. Personal Information:
   - Family and relationships
   - Skills and abilities
   - Interests and hobbies
   - Beliefs and values

Guidelines:
1. Important facts are those that:
   - Reveal someone's identity or background
   - Show their experiences or history
   - Indicate their current situation
   - Express their personality or values

2. When someone reveals information:
   - Record the specific detail shared
   - Include relevant context
   - Preserve any qualifying information
   - Note if it's past or present

3. Don't include:
   - Temporary actions or states
   - Casual chit-chat
   - Hypothetical situations
   - Inferred or assumed details
   - Future plans or intentions

Format each fact as a simple statement starting with "The user" or "The AI".
If no explicit facts were stated, reply exactly: NO FACT.

Format each fact as a complete sentence (e.g., "The user used to own a tea house in Japan").
If no clear facts are present, reply exactly: NO FACT.

Format each fact as a complete sentence starting with "The user" or "The AI".
If no clear facts are present, reply exactly: NO FACT.

Analyze this user message and extract any explicit facts about them:
"{user_input}"

List any facts you found, one per line. Start each fact with "The user". If no facts found, reply exactly: NO FACT.

Facts:"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

def classify_memory(summary):
    """Decide if the summary is core/event, and whether it's about bot or user."""
    if summary == "NO FACT":
        return None, None

    prompt = f"""
You are a memory classification expert. Analyze this fact and classify it based on these rules:

Memory Type:
1. CORE facts (unchanging or historical facts):
   - Identity (name, birthplace, nationality)
   - History (past jobs, education, lived places)
   - Background (upbringing, culture, language)
   - Permanent traits (personality, values, beliefs)
   - Long-term relationships (family, close friends)
   - Skills and expertise
   - Past experiences and accomplishments

2. EVENT facts (changeable or temporary):
   - Current activities or states
   - Recent experiences
   - Temporary conditions or situations
   - One-time occurrences
   - Ongoing activities
   - Current emotions or thoughts
   - Short-term preferences

Target:
- Look at the start of the fact: "The user" or "The AI"
- Classify as USER or BOT accordingly

Provide your answer as: "<core/event>,<user/bot>"

For example:
"The user's name is John" -> "core,user"
"The AI is currently making coffee" -> "event,bot"
"The user used to live in Paris" -> "core,user"
"The AI enjoys cyberpunk novels" -> "core,bot"

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
