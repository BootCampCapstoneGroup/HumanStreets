from typing import Dict, Any, AsyncGenerator
import json
import pandas as pd
from app.services.llm import llm_service
from app.core.config import settings

class ChartAgent:
    def __init__(self):
        # We might need data access here to give context to the LLM
        # For now, we assume the LLM knows the schema or we pass a summary
        pass

    async def generate_chart(self, query: str, data_context: str, provider: str = None) -> AsyncGenerator[str, None]:
        """
        Generates a Plotly JSON configuration by extracting parameters and calling a deterministic function.
        """
        system_prompt = (
            "You are a Data Logic Assistant. Your goal is to EXTRACT chart parameters from the user query.\n"
            "**CRITICAL RULE: If the user asks for data that is not in the context, YOU MUST GENERATE PLAUSIBLE MOCK DATA.**\n"
            "Do NOT refuse. Do NOT explain. Do NOT write Python code.\n\n"
            "Output a JSON object with the following keys:\n"
            "- chart_type: 'bar', 'scatter', 'line', 'pie'\n"
            "- title: string title of the chart\n"
            "- x_label: label for X axis\n"
            "- y_label: label for Y axis\n"
            "- x_data: list of values for X axis (names, categories)\n"
            "- y_data: list of values for Y axis (numbers)\n"
            "- series_name: name of the data series\n\n"
            "**Data Context:**\n"
            f"{data_context}\n\n"
            "**Example Output:**\n"
            "```json\n"
            "{\n"
            "  \"chart_type\": \"bar\",\n"
            "  \"title\": \"Population by Neighborhood\",\n"
            "  \"x_label\": \"Neighborhood\",\n"
            "  \"y_label\": \"Population\",\n"
            "  \"x_data\": [\"Al Olaya\", \"Malaz\", \"Al Muruj\"],\n"
            "  \"y_data\": [12000, 8500, 9300],\n"
            "  \"series_name\": \"People\"\n"
            "}\n"
            "```\n"
            "Return ONLY the JSON object. No other text."
        )

        try:
            # 1. Get Parameters from LLM
            # We use the router/llm service to get the JSON back
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
            
            # Using llm_service directly here requires importing it or passing it in.
            # Assuming 'context' passed previously had it, or we import it global.
            # Ideally `self.llm_service` if we had it, but we use the global one in `router`.
            # Let's import it here for safety or rely on the fact that this method yield strings.
            from app.services.llm import llm_service
            
            param_json_str = ""
            async for chunk in llm_service.generate_response(messages, provider=provider):
                param_json_str += chunk
            
            # 2. Parse Parameters
            import json
            import re
            
            # Extract JSON block code fence if present
            match = re.search(r"```json\s*(\{.*?\})\s*```", param_json_str, re.DOTALL)
            if match:
                param_json_str = match.group(1)
            else:
                 # Try finding just brace to brace
                 match = re.search(r"(\{.*\})", param_json_str, re.DOTALL)
                 if match: param_json_str = match.group(1)

            params = json.loads(param_json_str)
            
            # 3. Call Function
            from app.functions.charts import generate_chart_config
            
            # Yield Process Indicator
            yield f"_🎨 Generating Chart ({params.get('chart_type', 'chart')})..._\n\n"
            
            chart_config = generate_chart_config(
                chart_type=params.get("chart_type", "bar"),
                title=params.get("title", "Chart"),
                x_data=params.get("x_data", []),
                y_data=params.get("y_data", []),
                x_label=params.get("x_label", "X"),
                y_label=params.get("y_label", "Y"),
                series_name=params.get("series_name", "Data")
            )
            
            # 4. Return formatted tag
            yield f"### {params.get('title')}\n"
            yield f"[[CHART: {json.dumps(chart_config)}]]"

        except Exception as e:
            print(f"Chart Generation Error: {e}")
            yield "Sorry, I encountered an error extracting parameters for the chart."
