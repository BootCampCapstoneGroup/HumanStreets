from typing import Dict, Any, AsyncGenerator
import json
from app.services.llm import llm_service
from app.agents.sql_agent import SQLAgent
from app.agents.viz_agent import VizAgent
from app.agents.chart_agent import ChartAgent

class RouterAgent:
    def __init__(self):
        self.sql_agent = SQLAgent()
        self.viz_agent = VizAgent()
        self.chart_agent = ChartAgent()

    async def route_and_execute(self, query: str, context: Dict[str, Any] = None, provider: str = None) -> AsyncGenerator[str, None]:
        """
        Analyzes the user query and delegates to the appropriate agent.
        """
        # 1. Intent Classification
        classification_prompt = (
            "You are a router. Classify the following user query into one of these categories:\n"
            "- CHART: Requests for graphs, bar charts, scatter plots, distributions.\n"
            "- MAP: Requests to show/hide layers, highlight areas, or update the map view.\n"
            "- ANALYTICS: Questions about data, statistics, counts, or specific values (that don't explicitly ask for a chart).\n"
            "- GENERAL: General conversation, greetings, or questions not covered above.\n\n"
            f"Query: {query}\n\n"
            "Return ONLY the category name."
        )
        
        # We use a non-streaming call for classification (conceptually)
        # But our LLM service is streaming. We'll just aggregate.
        category = "GENERAL" 
        try:
            stream = llm_service.generate_response([{"role": "user", "content": classification_prompt}], provider=provider)
            full_resp = ""
            async for chunk in stream:
                full_resp += chunk
            category = full_resp.strip().upper()
            # Cleanup potential extra text
            for valid in ["CHART", "MAP", "ANALYTICS", "GENERAL"]:
                if valid in category:
                    category = valid
                    break
        except Exception as e:
            print(f"Routing failed: {e}. Defaulting to GENERAL.")
            category = "GENERAL"

        print(f"Routing to: {category}")

        # 2. Delegation
        if category == "CHART":
            async for chunk in self.chart_agent.generate_chart(query, context, provider=provider):
                yield chunk
        elif category == "MAP":
            async for chunk in self.viz_agent.handle_request(query, context, provider=provider):
                yield chunk
        elif category == "ANALYTICS":
            async for chunk in self.sql_agent.handle_request(query, context, provider=provider):
                yield chunk
        else:
            # GENERAL
            system_prompt = (
                "You are a helpful assistant for the HumanStreets Walkability Dashboard.\n"
                "**FORMATTING RULES:**\n"
                "- use **Markdown** for all responses.\n"
                "- Use **bold** for key terms and layer names.\n"
                "- Use `code blocks` for technical terms or tags.\n"
                "- Use bullet points or numbered lists for multiple items.\n"
                "- Ensure there is a blank line between list items.\n\n"
                "**Capabilities:**\n"
                "- **Map Layers**: Neighborhoods, H3 Walkability Heatmap.\n"
                "- **Drawing Tools**: The user can draw Points, Lines, and Polygons. You have access to these in `context['drawn_features']` if available.\n"
                "- **Analytics**: Street scores, statistics.\n"
                "- **Charts**: Distributions of scores.\n\n"
                "If the user asks to see layers, imply you are showing them (and output the tag).\n"
                "Available tags: `[[SHOW_LAYER: NEIGHBORHOODS]]`, `[[SHOW_LAYER: WALKABILITY]]`"
            )
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
            async for chunk in llm_service.generate_response(messages, provider=provider):
                yield chunk
