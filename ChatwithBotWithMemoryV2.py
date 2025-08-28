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
            
            # Special handling for identity information
            is_identity_info = any(x in content.lower() for x in ["name is", "my name", "i am", "i'm"])
            threshold = 2000 if is_identity_info else 3000
            
            if score < threshold:  
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
    
    # Create compound queries with different weights for different types of information
    queries = [
        ("What is known about the user's identity, background, and history?", 10),  # Highest priority - user core info
        ("What recent interactions have occurred with this user?", 8),              # High priority - recent interactions
        (query, 7),                                                                # Query-specific info
        ("What are the user's preferences and traits?", 6),                        # User preferences
        ("What is my personality and background?", 5),                             # Bot's core personality
        ("What are my recent actions and current state?", 4)                       # Bot's current state
    ]
    
    # Query for both core and event memories
    all_memories = []
    seen = set()
    
    for q, weight in queries:
        print("\n" + Fore.CYAN + f"Searching with query: {q}" + Style.RESET_ALL)
        results = collection.query(
            query_embeddings=[embed_text(q)],
            n_results=10
        )
        
        # Add unique memories while preserving order and applying weights
        for memory, dist in zip(results.get("documents", [[]])[0], results.get("distances", [[]])[0]):
            if memory not in seen and memory != "NO FACT":
                # Base weight adjustment
                weighted_dist = dist / weight
                
                # Further adjust based on memory type
                if memory.startswith("The user"):
                    if "name is" in memory.lower():
                        weighted_dist *= 0.2  # Highest boost for user's name
                    elif "used to" in memory.lower():
                        weighted_dist *= 0.4  # High boost for user's background
                    else:
                        weighted_dist *= 0.6  # Boost for other user facts
                elif memory.startswith("I "):
                    if "name is" in memory.lower():
                        weighted_dist *= 0.3  # High boost for bot's name
                    elif any(core in memory.lower() for core in ["was born", "am "]):
                        weighted_dist *= 0.5  # Boost for bot's core identity
                
                print(f"  Found (similarity={dist:.3f}, weighted={weighted_dist:.3f}): {memory}")
                all_memories.append((memory, weighted_dist))
                seen.add(memory)
            else:
                print(f"  Skipped duplicate: {memory}")
    
    # Sort memories by weighted distance and take top N
    all_memories.sort(key=lambda x: x[1])  # Sort by weighted distance
    final_memories = [m[0] for m in all_memories]  # Extract just the memories
    
    print("\n" + Fore.BLUE + "=== FINAL MEMORY SET ===" + Style.RESET_ALL)
    for i, memory in enumerate(final_memories, 1):
        print(f"{i}. {memory}")
    
    return final_memories

# --- Summarization + Classification ---
def summarize_interaction(user_input, ai_reply):
    """Extract only explicit factual info from the exchange."""
    prompt = f"""
You are a fact extraction expert. Extract ONLY facts that were EXPLICITLY stated by the user in this interaction.

RULES for Extracting Facts:
1. Identity Facts (Highest Priority):
   - Names MUST be stored exactly as stated (e.g., "I'm Fred" -> "The user's name is Fred")
   - All identity information must be explicit, never inferred
   - Store with proper context of who the fact belongs to (user vs bot)
   
2. Core Background Facts:
   - Locations (current or past)
   - Occupations (current or past)
   - Relationships
   - Long-term experiences ("I used to..." or "I have...")

2. History & Experience:
   - Past locations and residences 
   - Past ownership of places/things
   - Previous jobs and roles
   - Past experiences
   - Places visited

3. Current Information:
   - Current location
   - Current occupation
   - Current preferences/interests
   - Current relationships

4. DO NOT Include:
   - Temporary actions (smiling, waving)
   - Greetings or casual chat
   - Future plans
   - The AI's responses or actions
   - Hypothetical statements
   - Questions asked by the user

Examples:
Input: "Hi, I'm Sarah and I used to live in Paris"
✓ Output:
The user's name is Sarah
The user used to live in Paris

Input: "This reminds me of my old cafe in Paris"
✓ Output:
The user used to own a cafe in Paris

Input: "I used to live in London before moving here"
✓ Output:
The user used to live in London

Input: "*smiles* How are you?"
✗ Output: NO FACT

Analyze this user message and extract any explicit facts stated by the user:
"{user_input}"

Format each fact as a complete sentence on its own line. If no clear facts are present, reply exactly: NO FACT.
Each fact must be a complete, standalone sentence starting with "The user".
"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    facts = response["message"]["content"].strip()
    
    if facts == "NO FACT":
        return facts
        
    # Split multiple facts into individual facts
    fact_lines = [f.strip() for f in facts.split('\n') if f.strip() and f.strip().startswith("The user")]
    
    # If no valid facts were found, return NO FACT
    if not fact_lines:
        return "NO FACT"
        
    # Store each fact separately
    for fact in fact_lines:
        if fact != "NO FACT":
            store_fact(bot_id, fact, mem_type="core" if "name is" in fact.lower() or "used to" in fact.lower() else "event")
            
    return fact_lines[0]  # Return first fact for classification

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
    
    # Separate bot memories, user memories, and recent bot statements
    bot_memories = []
    user_memories = []
    bot_recent_statements = []
    
    for memory in memories:
        if memory.startswith("The user"):
            user_memories.append(memory)
        elif memory.startswith("The bot stated") or memory.startswith("The bot mentioned"):
            bot_recent_statements.append(memory)
        elif not memory.startswith("NO FACT"):
            bot_memories.append(memory)
    
    history = "\n".join([f"{role}: {msg}" for role, msg in recent_history])
    
    prompt = f"""
You are a unique character with personality and evolving history.

YOUR CORE PERSONALITY AND MEMORIES (these are facts about you, use "I" statements):
{chr(10).join(bot_memories)}

YOUR RECENT STATEMENTS (things you said or offered):
{chr(10).join(bot_recent_statements)}

WHAT YOU KNOW ABOUT THE USER (these are facts about them):
{chr(10).join(user_memories)}

Recent conversation:
{history}

User: {user_input}

Respond in character. Remember these rules:
1. You are the character defined in YOUR CORE PERSONALITY section
2. Stay consistent with YOUR RECENT STATEMENTS about menu items, offers, etc.
3. Use the facts from WHAT YOU KNOW ABOUT THE USER to inform your interactions
4. Never present yourself as having the user's traits or experiences
5. Stay consistent with your own background and personality

Assistant (in character):"""

    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    reply = response["message"]["content"]

    # Extract both user facts and bot statements from the interaction
    summary = summarize_interaction(user_input, reply)
    mem_type, target = classify_memory(summary)

    if summary != "NO FACT" and mem_type:
        store_fact(bot_id, summary, mem_type=mem_type, importance=1.0 if mem_type == "core" else 0.5)

    # Extract and store significant statements about the bot's activities and knowledge
    extract_bot_statement_prompt = f"""
Analyze the bot's reply for significant information that should be remembered for future conversations:

1. Any specific activities the bot performs
2. Any knowledge or expertise the bot demonstrates
3. Any specific offerings, capabilities, or services mentioned
4. Any distinctive features or characteristics described
5. Any commitments or promises made

Reply:
{reply}

Format output as "The bot [verb] [specific detail]", using verbs like:
- stated (for facts)
- mentioned (for references)
- offered (for services/capabilities)
- described (for characteristics)
- promised (for commitments)

Examples:
Bot: "I can analyze your data using quantum algorithms, and I specialize in cryptography."
✓ Output:
The bot stated they can analyze data using quantum algorithms
The bot mentioned they specialize in cryptography

Bot: "Let me help you with that task tomorrow, I'll make sure it's done by noon."
✓ Output:
The bot promised to complete the task by noon tomorrow

Bot: "How are you today? The weather is nice."
✓ Output: NO FACT

Each statement should be:
1. Specific and concrete
2. Relevant for future conversations
3. Based on explicit statements, not implications
4. Written in third person (The bot...)

Format as one statement per line, or NO FACT if no significant statements found.
"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": extract_bot_statement_prompt}])
    bot_statements = response["message"]["content"].strip()
    
    if bot_statements != "NO FACT":
        for statement in bot_statements.split('\n'):
            if statement.strip():
                # Determine importance based on statement type
                if "promised" in statement.lower():
                    importance = 0.9  # Highest importance for commitments
                elif any(word in statement.lower() for word in ["stated", "described"]):
                    importance = 0.8  # High importance for facts and descriptions
                else:
                    importance = 0.6  # Normal importance for general statements
                
                store_fact(bot_id, statement.strip(), mem_type="event", importance=importance)

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

    # bot_id = input("\nTalk to which bot? (e.g. isabella/kai): ").strip()
    bot_id = "kai"
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
