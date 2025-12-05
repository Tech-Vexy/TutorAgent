from deepagents import create_deep_agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

agent = create_deep_agent(model=model)

print("State Schema:")
print(agent.get_graph().schema)
