from dotenv import load_dotenv
import os

# Load .env explicitly
loaded = load_dotenv(verbose=True)
print(f"Dotenv loaded: {loaded}")

print(f"Current Working Directory: {os.getcwd()}")
print(f"File exists: {os.path.exists('.env')}")

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"GEMINI_API_KEY found: {api_key[:5]}...{api_key[-5:]}")
else:
    print("GEMINI_API_KEY is None or Empty")

# Print all keys in .env
from dotenv import dotenv_values
config = dotenv_values(".env")
print("Keys in .env:", list(config.keys()))
