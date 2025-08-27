import ollama

# Explicitly set the host
client = ollama.Client(host="http://192.168.1.32:11434")  # replace with your Windows IP

response = client.chat(
    model="gpt-oss:20b",
    messages=[{"role": "user", "content": "Hello from WSL!"}]
)

print(response.message["content"])
