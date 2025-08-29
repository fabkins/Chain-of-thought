import os
import uuid
import chromadb
from chromadb.utils import embedding_functions

# =========================
# Global Configuration
# =========================
DB_DIR = "./chroma_db_2_1"  # <--- change location here
CHROMA_CLIENT = chromadb.PersistentClient(path=DB_DIR)
EMBEDDINGS = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


# =========================
# Collection Utilities
# =========================
def get_bot_collection(bot_id, user_id):
    """
    Returns (or creates) the ChromaDB collection for a given bot and user.
    """
    collection_name = f"chat_{bot_id}_{user_id}"
    return CHROMA_CLIENT.get_or_create_collection(
        name=collection_name, embedding_function=EMBEDDINGS
    )


def store_memory(bot_id, user_id, memory_type, content):
    """
    Store a memory in the bot’s database.
    Each memory is uniquely identified with a UUID.
    """
    collection = get_bot_collection(bot_id, user_id)
    memory_id = str(uuid.uuid4())

    collection.add(
        ids=[memory_id], documents=[content], metadatas=[{"type": memory_type}]
    )

    print(
        f"\n=== STORING NEW FACT ===\nID: {memory_id}\nType: {memory_type}\nContent: {content}\n"
    )
    return memory_id


def retrieve_memories(bot_id, user_id, query, n_results=5):
    """
    Retrieve both global (bot identity) and user-specific memories.
    Combines results into a single list.
    """
    user_collection = get_bot_collection(bot_id, user_id)
    global_collection = get_bot_collection(bot_id, "global")

    results = []

    for collection in [global_collection, user_collection]:
        res = collection.query(query_texts=[query], n_results=n_results)
        if res and res.get("documents"):
            results.extend(res["documents"][0])

    return results


# =========================
# Bot Setup
# =========================
def seed_bots():
    """
    Seeds example bots with core memories into the GLOBAL collection only.
    """
    bots = {
        "isabella": [
            "My name is Isabella Rossi.",
            "I was born on April 3rd, 1985, in Florence, Italy.",
            "I studied literature at the University of Bologna.",
            "I worked as a librarian before becoming a travel writer.",
            "I am thoughtful, friendly, and love Renaissance art.",
        ],
        "kai": [
            "My name is Kai Tanaka.",
            "I was born on November 12th, 1992, in Osaka, Japan.",
            "I used to be a robotics engineer before opening a coffee shop.",
            "I own a coffee shop called 'Cyber Brew'.",
            "My coffee shop is located in Osaka, Japan.",
            "I am curious, innovative, and enjoy cyberpunk novels.",
        ],
    }

    for bot_id, facts in bots.items():
        collection = get_bot_collection(bot_id, "global")

        # Avoid reseeding duplicates
        existing = collection.count()
        if existing > 0:
            print(f"Bot '{bot_id}' already has {existing} memories, skipping seeding.")
            continue

        for fact in facts:
            store_memory(bot_id, "global", "core", fact)


# =========================
# Example Chat Loop
# =========================
def chat_loop():
    print("Multi-User Multi-Bot Chatbot ready! Type 'quit' to exit.\n")

    # Seed bots if database empty
    if not os.path.exists(DB_DIR) or len(os.listdir(DB_DIR)) == 0:
        print("No database found, seeding example bots...")
        seed_bots()

    while True:
        bot_id = input("Talk to which bot? (e.g. isabella/kai): ").strip().lower()
        if bot_id == "quit":
            break

        user_id = input("Enter your username: ").strip().lower()
        if user_id == "quit":
            break

        while True:
            user_input = input(f"\nYou ({user_id}) to {bot_id}: ")
            if user_input.strip().lower() == "quit":
                break

            # Retrieve memories
            print("\n=== MEMORY RETRIEVAL ===")
            memories = retrieve_memories(bot_id, user_id, user_input)
            print("Retrieved memories:", memories)

            # Store the new user message as "recent"
            store_memory(bot_id, user_id, "recent", user_input)

            # ---- Replace this with actual LLM call ----
            print(
                f"\n{bot_id}: I remember some things about myself and our chats... "
                f"Based on '{user_input}', I recall: {memories[:2]} ...\n"
            )


if __name__ == "__main__":
    chat_loop()
