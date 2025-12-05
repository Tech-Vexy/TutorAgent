import os
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("No API key found")
    exit()

client = Groq(api_key=api_key)
try:
    models = client.models.list()
    for m in models.data:
        print(m.id)
except Exception as e:
    print(e)
