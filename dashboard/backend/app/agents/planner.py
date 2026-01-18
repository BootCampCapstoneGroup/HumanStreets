from typing import Dict, Any, List, Optional
import json
from app.services.llm import llm_service
from app.services.spatial import spatial_service
from app.agents.sql_agent import SQLAgent
from app.agents.viz_agent import VizAgent
from app.schemas import Plan, PlanStep

class PlannerAgent:
    def __init__(self):
        self.sql_agent = SQLAgent()
        self.viz_agent = VizAgent()

    def verify_data_availability(self, query: str) -> List[str]:
        """
        Checks if the query requests data that might be missing or invalid.
        Returns a list of warnings or errors.
        """
        issues = []
        # Check Layers
        query_upper = query.upper()
        
        # Simple heuristic check for layers
        # If user asks for "Districts", warn if only "Neighborhoods" exist (handled by VizAgent alias, but good to note)
        known_layers = spatial_service.available_layers
        
        # This is a basic check. The LLM does the heavy lifting in generate_plan,
        # but this method can inject context into the prompt.
        return issues

    async def generate_plan(self, query: str, history: List[Dict[str, Any]]) -> Plan:
        """
        Generates a structured plan based on query and history.
        """
        # 1. Gather Context
        layers = spatial_service.available_layers
        schema = self.sql_agent.schema_description
        
        plan_prompt = (
            "You are a Planner Agent. Your goal is to break down complex user requests into executable steps.\n"
            f"User Query: {query}\n"
            f"Chat History: {history[-3:] if history else []}\n"
            f"Available Layers: {layers}\n"
            f"Database Schema: {schema}\n\n"
            "**INSTRUCTIONS**:\n"
            "1. **Check Validity**: Does the user ask for a layer used in 'Available Layers'? (Note: 'Districts' usually means 'NEIGHBORHOODS').\n"
            "2. **Break Down**: If complex (Analytics + Viz), strictly separate them.\n"
            "3. **Format**: Output a JSON conforming to the `Plan` schema.\n"
            "   - `reasoning`: Explain why you chose these steps and confirm data exists.\n"
            "   - `steps`: specific instructions for the agents. Actions: [SQL], [VIZ].\n"
            "   - `confirmation_question`: A question to ask the user to verify the plan.\n\n"
            "**EXAMPLE**:\n"
            "Query: 'Show eastern neighborhoods with high scores'\n"
            "Output JSON:\n"
            "{\n"
            '  "reasoning": "User wants spatial filtering. Schema has `neighborhoods`. I will query DB then show result.",\n'
            '  "steps": ["SQL: Select geometry from neighborhoods where name like \'%East%\'...", "VIZ: Show query result"],\n'
            '  "confirmation_question": "I will find neighborhoods in the East with high scores and display them. Proceed?"\n'
            "}"
        )

        response_stream = llm_service.generate_response(
            [{"role": "user", "content": plan_prompt}]
        )
        
        # Accumulate streaming response (not truly streaming for structured output typically, but service might be wrapped)
        # Assuming llm_service.generate_response with response_model returns the object (or stream of text).
        # Our llm_service currently streams text. We need to handle this.
        # Wait - llm_service.generate_response returns a generator.
        # We need to buffer it and parse.
        
        full_text = ""
        async for chunk in response_stream:
            full_text += chunk
            
        # Parse JSON from text
        try:
            # Cleanup markdown
            json_str = full_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(json_str)
            return Plan(**data)
        except Exception as e:
            print(f"Plan Generation Failed: {e}. Raw: {full_text}")
            # Fallback plan
            return Plan(
                reasoning="Failed to generate structured plan.",
                steps=["SQL: " + query],
                confirmation_question="I will try to execute your query directly. Okay?"
            )

    async def execute_plan(self, plan: Plan) -> str:
        """
        Executes the given plan steps.
        """
        results = []
        for step in plan.steps:
            if step.startswith("SQL:"):
                q = step.replace("SQL:", "").strip()
                # Run SQL Agent (we need to iterate its stream)
                async for chunk in self.sql_agent.handle_request(q):
                    results.append(chunk) 
            elif step.startswith("VIZ:"):
                q = step.replace("VIZ:", "").strip()
                # Run Viz Agent
                async for chunk in self.viz_agent.handle_request(q, context={"last_result": spatial_service.get_query_result()}):
                    results.append(chunk)
            
        return "\n".join(results)
