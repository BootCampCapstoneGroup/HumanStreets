from typing import Dict, Any, AsyncGenerator
import json
from app.services.llm import llm_service
from app.agents.sql_agent import SQLAgent
from app.agents.viz_agent import VizAgent
from app.agents.chart_agent import ChartAgent
from app.agents.planner import PlannerAgent
import json

class RouterAgent:
    def __init__(self):
        self.sql_agent = SQLAgent()
        self.viz_agent = VizAgent()
        self.chart_agent = ChartAgent()
        self.planner_agent = PlannerAgent()
        self.active_plans = {} # Map 'user_id' or simple context to plan. Since no user_id, valid for single session.

    async def route_and_execute(self, query: str, history: list[dict] = [], context: Dict[str, Any] = None, provider: str = None) -> AsyncGenerator[str, None]:
        """
        Analyzes the user query and delegates to the appropriate agent.
        """
        # 1. Intent Classification
        classification_prompt = (
            "You are a router. Classify the following user query into one of these categories:\n"
            "- LOOKUP: Simple requests to find or show a specific neighborhood by name (Arabic or English), e.g. 'show walkability for حي النرجس', 'find النزهة neighborhood'.\n"
            "- CHART: Requests for graphs, bar charts, scatter plots, distributions.\n"
            "- MAP: Requests to show/hide layers, highlight areas, or update the map view.\n"
            "- ANALYTICS: Questions about data, statistics, counts, or specific values.\n"
            "- PLANNER: Complex requests requiring multiple steps (e.g., 'Find X then Show Y where Z'), or verification of a plan.\n"
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
            # Cleanup
            for valid in ["LOOKUP", "CHART", "MAP", "ANALYTICS", "PLANNER", "GENERAL"]:
                if valid in category:
                    category = valid
                    break

            # AGGRESSIVE FILTER: Force GENERAL if query lacks spatial/data intent keywords
            # This prevents "how r u" from triggering SQL lookups.
            spatial_keywords = [
                'map', 'show', 'display', 'view', 'find', 'search', 'where', 'which', 'what',
                'walkability', 'score', 'data', 'chart', 'graph', 'plot', 'list',
                'neighborhood', 'district', 'riyadh', 'saudi',
                'حي', 'الرياض', 'منطقة', 'خريطة', 'مخطط', 'بيانات', 'أين', 'كم', 'عرض', 'أظهر'
            ]
            
            has_spatial_keyword = any(k in query.lower() for k in spatial_keywords)
            
            if not has_spatial_keyword and category != "GENERAL":
                print(f"[Router] Query '{query}' lacks spatial keywords. Fallback to GENERAL.")
                category = "GENERAL"

            # Heuristic: If user says "Yes" or "Approved", check if we have a pending plan
            # Relaxed check: look for keywords in the string
            confirm_keywords = ["yes", "approve", "confirmed", "go", "okay", "ok", "proceed"]
            if any(keyword in query.lower() for keyword in confirm_keywords) and self.active_plans.get("last_plan"):
                category = "EXECUTE_PLAN"
            
            import re
            
            # Heuristic: LOOKUP - Simple neighborhood queries (check BEFORE PLANNER)
            # ONLY matches when a SPECIFIC Arabic neighborhood name is mentioned:
            # - "حي النرجس" (neighborhood Al-Narjis)
            # - "الروضة" (Al-Rawda - with definite article)
            # Does NOT match generic "neighborhoods" (plural) queries
            lookup_pattern = r"(walkability|score|show|find|أظهر|عرض|اعرض).*(حي\s+[\u0600-\u06FF]+|for\s+ال[\u0600-\u06FF]+)"
            
            # Additional check: exclude if query contains complex filters or is about multiple neighborhoods
            complex_query_pattern = r"(where|above|below|>|<|best|worst|top|bottom|أفضل|أسوأ|eastern|western|north|south|neighborhoods\b|all\s+the)"
            
            if re.search(lookup_pattern, query, re.IGNORECASE | re.UNICODE):
                if not re.search(complex_query_pattern, query, re.IGNORECASE):
                    category = "LOOKUP"
            
            # Heuristic: Regex for PLANNER Trigger
            # If query implies complex multi-step logic (e.g. "Show... where...", "Find... then...")
            if category != "LOOKUP":  # Don't override LOOKUP
                if re.search(r"(show|find|list)\s+.*(where|with|which)\s+", query, re.IGNORECASE):
                    category = "PLANNER"
                if "plan" in query.lower() or "verify" in query.lower():
                    category = "PLANNER"
                
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
        elif category == "PLANNER":
            # Generate Plan
            plan = await self.planner_agent.generate_plan(query, history)
            self.active_plans["last_plan"] = plan
            yield f"**Plan Proposed**:\n{plan.reasoning}\n\n**Steps**:\n"
            for step in plan.steps:
                yield f"- {step}\n"
            yield f"\n**{plan.confirmation_question}**"
            
        elif category == "EXECUTE_PLAN":
            plan = self.active_plans.pop("last_plan", None)
            if plan:
                yield "**Executing Plan...**\n\n"
                result = await self.planner_agent.execute_plan(plan)
                yield result
            else:
                yield "No active plan found to execute."
        
        elif category == "LOOKUP":
            # Direct neighborhood lookup - route to SQL agent with spatial query hint
            lookup_query = f"Find the neighborhood matching the name in this request and return its geometry and walkability score: {query}"
            async for chunk in self.sql_agent.handle_request(lookup_query, context, provider=provider):
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
