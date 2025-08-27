import ollama
import chromadb
from chromadb.utils import embedding_functions
import uuid, datetime

# --- Setup ChromaDB client ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or load a collection for a bot
def get_bot_collection(bot_id):
    return chroma_client.get_or_create_collection(
        name=f"chat_{bot_id}",
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )

# Store a core fact for a specific bot
def store_core_fact(bot_id, content):
    collection = get_bot_collection(bot_id)
    doc_id = str(uuid.uuid4())
    embedding = ollama.embeddings(model="mxbai-embed-large", prompt=content)["embedding"]
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{
            "type": "core",
            "timestamp": str(datetime.datetime.now()),
            "importance": 1.0
        }]
    )

# Seed a bot with backstory
def seed_bot(bot_id, backstory_facts):
    for fact in backstory_facts:
        store_core_fact(bot_id, fact)

# Example usage
if __name__ == "__main__":
    seed_bot("isabella", [
        "My name is Isabella Rossi.",
        "I was born on April 3rd, 1985, in Florence, Italy.",
        "My father, Marco Rossi, was a carpenter.",
        "I studied literature at the University of Bologna.",
        "I worked as a librarian before becoming a travel writer.",
        "I love Renaissance art and Italian cooking."
    ])

    seed_bot("kai", [
        "My name is Kai Tanaka.",
        "I was born on November 12th, 1992, in Osaka, Japan.",
        "I used to be a robotics engineer before opening a coffee shop.",
        "I am curious, innovative, and enjoy cyberpunk novels."
    ])

    print("Seeded Isabella and Kai with backstories!")
