from deepagents import create_deep_agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

def my_tool(x: int) -> int:
    """Doubles the input."""
    return x * 2

agent = create_deep_agent(model=model, tools=[my_tool])

print("Agent created.")
# Inspect the graph to see tools if possible, or just run it
# The tools are likely bound to the model in the graph.
# We can try to invoke it and see if it knows about 'ls'

print("Invoking agent...")
try:
    # We mock the state
    result = agent.invoke({"messages": [("user", "List the files in the current directory.")]})
    print(result["messages"][-1].content)
except Exception as e:
    print(f"Error: {e}")
