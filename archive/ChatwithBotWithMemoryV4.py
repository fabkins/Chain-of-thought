import ollama
import chromadb
import uuid, datetime, os, json
from colorama import Fore, Style, init
import re
from functools import lru_cache

"""
Multi-user memory chatbot with separated collections for user memories and shared bot memories.
Key features:
- Separate ChromaDB collections for each user's memories
- Shared collection for bot's core memories
- Single LLM call per turn returns BOTH the assistant reply and structured memory updates
- Duplicate detection with tunable thresholds and verbose diagnostics
- Optional emote stripping
"""

# --- Init colorama ---
init(autoreset=True)

# --- Config ---
MODEL = "llama3.2"   # use same model for chat + embeddings
ALLOW_EMOTES = False  # set True to keep *emotes* in bot replies
CHROMA_PATH = "./chroma_db2"

# Duplicate detection tuning (distances come from Chroma; lower = closer for most metrics)
DUP_THRESHOLD_CORE = 0.12   # treat as duplicate if distance < threshold
DUP_THRESHOLD_EVENT = 0.10
TOP_K_RETRIEVE = 12

# --- Setup ChromaDB ---
os.makedirs(CHROMA_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_user_collection(bot_id: str, user_id: str):
    """Return (or create) a ChromaDB collection for user-specific memories."""
    return chroma_client.get_or_create_collection(name=f"chat_{bot_id}_user_{user_id}")

def get_shared_bot_collection(bot_id: str):
    """Return (or create) a ChromaDB collection for shared bot memories."""
    return chroma_client.get_or_create_collection(name=f"chat_{bot_id}_shared")

# --- Embeddings (with simple in-memory cache) ---
@lru_cache(maxsize=8192)
def _embed_cached(model: str, text: str):
    return tuple(ollama.embeddings(model=model, prompt=text)["embedding"])

def embed_text(text: str):
    """Get an embedding using Ollama with caching."""
    return list(_embed_cached(MODEL, text))

def store_memory(collection, content: str, mem_type: str = "event", metadata: dict = None):
    """Store a memory in the specified collection with duplicate checking."""
    if metadata is None:
        metadata = {}
    
    print(f"\n{Fore.YELLOW}=== CHECKING FOR DUPLICATES ==={Style.RESET_ALL}")
    print(f"Collection: {collection.name}")
    print(f"Content: {content}")
    
    # Check for similar existing memories
    existing = collection.query(
        query_embeddings=[embed_text(content)],
        n_results=3
    )
    
    if existing["documents"] and existing["documents"][0]:
        for doc, dist in zip(existing["documents"][0], existing["distances"][0]):
            print(f"{Fore.YELLOW}Comparing with existing (dist={dist:.4f}):{Style.RESET_ALL}")
            print(f"  New: {content}")
            print(f"  Old: {doc}")
            
            # Use appropriate threshold based on memory type
            threshold = DUP_THRESHOLD_CORE if mem_type == "core" else DUP_THRESHOLD_EVENT
            
            if dist < threshold:
                print(f"{Fore.RED}SKIPPED: Too similar (dist={dist:.4f} < threshold={threshold}){Style.RESET_ALL}")
                return None
    
    # If we get here, store the new memory
    doc_id = str(uuid.uuid4())
    
    # Add standard metadata
    full_metadata = {
        "type": mem_type,
        "timestamp": str(datetime.datetime.now()),
        **metadata
    }
    
    collection.add(
        ids=[doc_id],
        embeddings=[embed_text(content)],
        documents=[content],
        metadatas=[full_metadata]
    )
    
    print(f"\n{Fore.GREEN}=== STORED NEW MEMORY ==={Style.RESET_ALL}")
    print(f"ID: {doc_id}")
    print(f"Type: {mem_type}")
    print(f"Content: {content}")
    return doc_id

def store_fact(bot_id: str, user_id: str, content: str, mem_type: str = "event", metadata: dict = None):
    """Store a fact in the appropriate collection based on content."""
    if metadata is None:
        metadata = {}
    
    # Determine if this is a bot memory or user memory
    is_bot_memory = content.lower().startswith("i ") or \
                    any(x in content.lower() for x in ["my personality", "my core", "my background"])
    
    # Add user_id to metadata
    metadata["user_id"] = "shared" if is_bot_memory else user_id
    
    # Store in appropriate collection
    if is_bot_memory:
        collection = get_shared_bot_collection(bot_id)
    else:
        collection = get_user_collection(bot_id, user_id)
    
    return store_memory(collection, content, mem_type, metadata)

def query_memories(collection, query: str, metadata_filter: dict = None, top_k: int = TOP_K_RETRIEVE):
    """Query a single collection for memories."""
    query_params = {
        "query_embeddings": [embed_text(query)],
        "n_results": top_k
    }
    
    if metadata_filter:
        query_params["where"] = metadata_filter
    
    results = collection.query(**query_params)
    
    return {
        "documents": results.get("documents", [[]])[0],
        "distances": results.get("distances", [[]])[0],
        "metadatas": results.get("metadatas", [[]])[0]
    }

def retrieve_memories(bot_id: str, user_id: str, query: str):
    """Retrieve relevant memories from both user-specific and shared collections."""
    user_collection = get_user_collection(bot_id, user_id)
    shared_collection = get_shared_bot_collection(bot_id)
    
    print(f"\n{Fore.BLUE}=== MEMORY RETRIEVAL ==={Style.RESET_ALL}")
    print(f"Query: {query}")
    
    # Define memory retrieval queries with weights
    retrieval_queries = [
        ("What is known about the user's identity and background?", 10),
        ("What recent interactions have occurred with this user?", 8),
        (query, 7),
        ("What are the user's preferences and traits?", 6),
        ("What is my personality and background?", 5),
        ("What are my recent actions and current state?", 4)
    ]
    
    all_memories = []
    seen_memories = set()
    
    for q, weight in retrieval_queries:
        print(f"\n{Fore.CYAN}Searching: {q}{Style.RESET_ALL}")
        
        # Query user memories
        user_results = query_memories(user_collection, q)
        # Query shared bot memories
        shared_results = query_memories(shared_collection, q)
        
        for results in [user_results, shared_results]:
            for doc, dist, meta in zip(results["documents"], results["distances"], results["metadatas"]):
                if doc not in seen_memories:
                    # Adjust weight based on memory type and content
                    adj_dist = dist / weight
                    if doc.startswith("The user"):
                        adj_dist *= 0.4 if meta["type"] == "core" else 0.7
                    elif doc.startswith("I "):
                        adj_dist *= 0.5 if meta["type"] == "core" else 0.8
                    
                    all_memories.append((doc, adj_dist, meta))
                    seen_memories.add(doc)
                    print(f"Found (dist={dist:.4f}, adj={adj_dist:.4f}): {doc}")
    
    # Sort by adjusted distance
    all_memories.sort(key=lambda x: x[1])
    
    # Organize into categories
    memories = {
        "user_core": [],
        "user_events": [],
        "bot_core": [],
        "bot_events": []
    }
    
    for doc, _, meta in all_memories:
        if doc.startswith("The user"):
            if meta["type"] == "core":
                memories["user_core"].append(doc)
            else:
                memories["user_events"].append(doc)
        elif doc.startswith("I "):
            if meta["type"] == "core":
                memories["bot_core"].append(doc)
            else:
                memories["bot_events"].append(doc)
    
    print_memory_summary(memories)
    return memories

def print_memory_summary(memories: dict):
    """Print an organized summary of memories."""
    print(f"\n{Fore.BLUE}=== MEMORY SUMMARY ==={Style.RESET_ALL}")
    
    categories = [
        ("USER CORE:", "user_core"),
        ("USER EVENTS:", "user_events"),
        ("BOT CORE:", "bot_core"),
        ("BOT EVENTS:", "bot_events")
    ]
    
    for title, key in categories:
        print(f"\n{Fore.CYAN}{title}{Style.RESET_ALL}")
        for i, memory in enumerate(memories[key], 1):
            print(f"{i}. {memory}")

def format_memories_for_prompt(memories: dict) -> str:
    """Format memories for inclusion in the chat prompt."""
    sections = []
    
    if memories["user_core"]:
        sections.append("Core Facts About User:\n" + "\n".join(memories["user_core"]))
    
    if memories["user_events"]:
        sections.append("Recent User Context:\n" + "\n".join(memories["user_events"]))
    
    if memories["bot_core"]:
        sections.append("Your Core Identity:\n" + "\n".join(memories["bot_core"]))
    
    if memories["bot_events"]:
        sections.append("Your Recent Actions:\n" + "\n".join(memories["bot_events"]))
    
    return "\n\n".join(sections)

def extract_memories(text: str) -> list:
    """Extract memory-worthy statements from text."""
    if not text or text.strip().lower() == "no fact":
        return []
    return [line.strip() for line in text.split('\n') if line.strip()]

def process_chat_message(bot_id: str, user_id: str, user_input: str, prev_messages: list = None):
    """Process a chat message and manage memory updates."""
    if prev_messages is None:
        prev_messages = []
    
    # Get relevant memories
    memories = retrieve_memories(bot_id, user_id, user_input)
    memory_context = format_memories_for_prompt(memories)
    
    # Format conversation history
    history_context = "Recent Messages:\n" + "\n".join(prev_messages) if prev_messages else "No previous messages."
    
    # Generate response and extract memories in a single LLM call
    prompt = f"""You are a chat companion with access to memory. Process this interaction and provide:
1. Your response to the user
2. Any new factual memories to store about the user
3. Any new memories about your own actions/state

Current Knowledge:
{memory_context}

{history_context}

User: {user_input}

Respond in this format:
RESPONSE: Your natural conversational response
USER MEMORIES: New facts about the user (one per line, start with "The user")
BOT MEMORIES: New facts about yourself (one per line, start with "I")
"""
    
    result = ollama.generate(model=MODEL, prompt=prompt)
    sections = result['response'].split('\n\n')
    
    # Parse response sections
    response = ""
    user_memories = []
    bot_memories = []
    
    current_section = None
    for line in sections:
        line = line.strip()
        if line.startswith("RESPONSE:"):
            current_section = "response"
            response = line.replace("RESPONSE:", "").strip()
        elif line.startswith("USER MEMORIES:"):
            current_section = "user"
            continue
        elif line.startswith("BOT MEMORIES:"):
            current_section = "bot"
            continue
        elif line:
            if current_section == "user":
                if line.lower() != "none" and line.lower() != "no fact":
                    user_memories.append(line)
            elif current_section == "bot":
                if line.lower() != "none" and line.lower() != "no fact":
                    bot_memories.append(line)
    
    # Clean response if needed
    if not ALLOW_EMOTES:
        response = re.sub(r'\*.*?\*', '', response).strip()
    
    # Store new memories
    for memory in user_memories:
        store_fact(bot_id, user_id, memory)
    for memory in bot_memories:
        store_fact(bot_id, user_id, memory)
    
    return response

if __name__ == "__main__":
    bot_id = "test_bot"
    print(f"{Fore.GREEN}Multi-user Chat System Initialized{Style.RESET_ALL}")
    print("Type 'exit' to end chat")
    
    user_id = input("Enter your user ID: ").strip()
    
    # Initialize bot's personality if this is first time
    shared_collection = get_shared_bot_collection(bot_id)
    if not shared_collection.count():
        print(f"{Fore.YELLOW}Initializing bot personality...{Style.RESET_ALL}")
        core_personality = [
            "I am a friendly and knowledgeable AI assistant",
            "I aim to be helpful while maintaining appropriate boundaries",
            "I have a gentle sense of humor but prioritize being informative",
            "I remember conversations with different users separately",
            "I maintain consistent personality traits across all conversations"
        ]
        for trait in core_personality:
            store_fact(bot_id, "shared", trait, "core")
    
    prev_messages = []
    while True:
        user_input = input(f"\n{Fore.CYAN}You:{Style.RESET_ALL} ").strip()
        if user_input.lower() == 'exit':
            break
        
        response = process_chat_message(bot_id, user_id, user_input, prev_messages)
        print(f"\n{Fore.GREEN}Bot:{Style.RESET_ALL} {response}")
        
        # Update conversation history
        prev_messages.append(f"User: {user_input}")
        prev_messages.append(f"Bot: {response}")
        if len(prev_messages) > 6:  # Keep last 3 exchanges
            prev_messages = prev_messages[-6:]
