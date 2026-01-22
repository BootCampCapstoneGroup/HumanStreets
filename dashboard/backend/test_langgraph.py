import asyncio
from app.agents.graph.workflow import build_graph
from langchain_core.messages import HumanMessage
from app.core.config import settings

from app.core.config import settings
from app.services.llm import llm_service

async def main():
    print("Initializing Models (this may take a while)...")
    llm_service.initialize_models()
    
    print("Testing LangGraph Agent...")
    
    graph = build_graph()
    
    # Test 1: General Query
    print("\n--- Test 1: General Query ---")
    inputs = {
        "messages": [HumanMessage(content="Hello, what can you do?")],
        "sql_query": "", 
        "query_result": "",
        "error": "",
        "next": ""
    }
    result = await graph.ainvoke(inputs)
    print("Response:", result['messages'][-1].content)
    
    # Test 2: SQL Query
    print("\n--- Test 2: SQL Query ---")
    inputs = {
        "messages": [HumanMessage(content="Count how many neighborhoods are in the database.")],
        "sql_query": "", 
        "query_result": "",
        "error": "",
        "next": ""
    }
    result = await graph.ainvoke(inputs)
    print("SQL:", result.get('sql_query'))
    print("Result:", result.get('query_result'))
    print("Response:", result['messages'][-1].content)
    
    # Test 3: Schema Query (previously failing)
    print("\n--- Test 3: Schema Query ---")
    inputs = {
        "messages": [HumanMessage(content="show me the database tables")],
        "sql_query": "", 
        "query_result": "",
        "error": "",
        "next": ""
    }
    result = await graph.ainvoke(inputs)
    print("SQL:", result.get('sql_query'))
    print("Result:", result.get('query_result'))
    print("Response:", result['messages'][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
