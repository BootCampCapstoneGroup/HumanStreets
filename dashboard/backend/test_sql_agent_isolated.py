
import asyncio
import os
import sys

# Ensure backend path is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app.agents.sql_agent import SQLAgent
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


from app.services.llm import llm_service

async def test_sql_generation():
    # Set encoding for Windows console (best effort)
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    print("Initializing Models...")
    llm_service.initialize_models()

    print("Initializing SQLAgent...")
    agent = SQLAgent()
    
    test_queries = [
        "Show Al-Narjis and Al-Malqa",
        "show top 5 walkable neighborhoods",
        "show me حي الملقا و حي النسيم", # Arabic multi-select
        "Show me the database tables"
    ]
    
    for query in test_queries:
        try:
            print(f"\n--- Testing Query: {query} ---")
        except UnicodeEncodeError:
            print(f"\n--- Testing Query: (Arabic/Special Chars) ---")
            
        try:
            async for chunk in agent.handle_request(query, provider="gemini"):
                if isinstance(chunk, str):
                   try:
                       print(chunk, end="")
                   except UnicodeEncodeError:
                       print(chunk.encode('ascii', 'replace').decode(), end="")
        except Exception as e:
            print(f"\nError: {e}")
        print("\n--------------------------------")

if __name__ == "__main__":
    asyncio.run(test_sql_generation())
