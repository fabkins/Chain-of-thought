import ollama
import math

def ask_ollama(messages, model="gpt-oss:20b"):
    client = ollama.Client(host="http://192.168.1.32:11434")
    response = client.chat(model=model, messages=messages)


    return response["message"]["content"]

class HybridToolAgent:
    def __init__(self, model="gpt-oss:20b", max_steps=6):
        self.model = model
        self.max_steps = max_steps
        self.history = []
        self.tools = {}

        # Fixed system message sets the agent’s behavior
        self.system_message = {
            "role": "system",
            "content": (
                "You are a reasoning agent.\n"
                "Always respond in this format:\n"
                "- ACTION: <reason | tool:TOOLNAME(arguments)>\n"
                "- REASON: <why this step is needed>\n"
                "- DONE: <yes/no>"
            )
        }

    def register_tool(self, name, func, description=""):
        """Register a tool dynamically at runtime."""
        self.tools[name] = {"func": func, "description": description}

    def run(self, question):
        print(f"🤔 Question: {question}\n")
        results = []

        # Conversation log starts with system + user
        messages = [
            self.system_message,
            {"role": "user", "content": f"Solve this problem: {question}"}
        ]

        for step in range(1, self.max_steps + 1):
            # Show tools + results so far
            tool_list = "\n".join([f"- {name}: {info['description']}" for name, info in self.tools.items()])
            messages.append({
                "role": "user",
                "content": (
                    f"QUESTION: {question}\n"
                    f"RESULTS SO FAR: {results}\n\n"
                    f"TOOLS AVAILABLE:\n{tool_list}\n"
                )
            })

            # Model decides what to do
            decision = ask_ollama(messages, self.model)
            print(f"📋 Step {step} Decision:\n{decision}\n")
            self.history.append((f"decision_{step}", decision))

            # Add to assistant history
            messages.append({"role": "assistant", "content": decision})

            # Parse action
            action_line = [line for line in decision.splitlines() if line.startswith("ACTION:")]
            done_line = [line for line in decision.splitlines() if line.startswith("DONE:")]

            action = action_line[0].replace("ACTION:", "").strip() if action_line else None
            done = "yes" in done_line[0].lower() if done_line else False

            if not action:
                print("⚠️ No valid action, stopping.")
                break

            # Execute action
            if action.startswith("tool:"):
                try:
                    tool_name, arg_str = action.replace("tool:", "").split("(", 1)
                    tool_name = tool_name.strip()
                    arg_str = arg_str.rstrip(")").strip()
                except:
                    print("⚠️ Bad tool syntax. Skipping.")
                    continue

                if tool_name in self.tools:
                    result = self.tools[tool_name]["func"](arg_str)
                else:
                    result = f"Unknown tool: {tool_name}"
            else:
                reasoning_message = {
                    "role": "user",
                    "content": f"Perform this reasoning step: {action}\nPrevious results: {results}"
                }
                result = ask_ollama([self.system_message] + messages + [reasoning_message], self.model)

            results.append(result)
            print(f"✅ Step {step} Result:\n{result}\n")
            self.history.append((f"result_{step}", result))

            # Add result as assistant message (so Ollama “remembers” what it output)
            messages.append({"role": "assistant", "content": f"Result: {result}"})

            if done:
                break

        # Final Answer
        messages.append({
            "role": "user",
            "content": (
                f"QUESTION: {question}\n"
                f"RESULTS: {results}\n\n"
                f"Provide the FINAL CONCISE ANSWER."
            )
        })
        final_answer = ask_ollama(messages, self.model)
        self.history.append(("final_answer", final_answer))

        print("🎯 Final Answer:", final_answer)
        return final_answer


# --- Example usage ---
if __name__ == "__main__":
    agent = HybridToolAgent(model="gpt-oss:20b")

    # Register tools dynamically
    agent.register_tool("calculator", lambda expr: str(eval(expr, {"__builtins__": None, "math": math}, {})),
                        description="Evaluate math expressions, supports math.sqrt, math.log, etc.")
    
    agent.register_tool("word_length", lambda word: str(len(word)),
                        description="Count the number of letters in a word.")

    question = "What is the square root of 225 plus the number of letters in the word 'aardwolf'?"
    answer = agent.run(question)
