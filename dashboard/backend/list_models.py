from google import genai
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("No API Key found")
    exit(1)

client = genai.Client(api_key=api_key)

print("Listing (some) available models:")
try:
    # Attempting to list models using the new client
    # If this method varies, we might need to adjust.
    for m in client.models.list():
        print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
