import asyncio
from agent_graph import graph
from langchain_core.messages import HumanMessage

async def verify_deep_agent():
    print("Verifying Deep Agent Integration...")
    
    # A query that might trigger the deep agent's planning or tool use
    query = "Create a python script to calculate the first 10 Fibonacci numbers and explain how it works."
    
    print(f"\nQuery: {query}")
    
    inputs = {
        "messages": [HumanMessage(content=query)],
        "user_profile": {"student_id": "test_user"},
        "pedagogy_strategy": "DIRECT_ANSWER",
        "intent": "COMPLEX_REASONING", # Force deep thinker
        "model_preference": "smart"
    }
    
    try:
        async for event in graph.astream(inputs):
            for key, value in event.items():
                if key == "deep_thinker":
                    print("\n--- Deep Thinker Response ---")
                    messages = value.get("messages", [])
                    if messages:
                        print(messages[-1].content)
                    else:
                        print("No messages returned from deep thinker.")
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    asyncio.run(verify_deep_agent())
