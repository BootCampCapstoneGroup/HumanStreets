import pandas as pd
import json
from typing import Dict, Any, AsyncGenerator
from app.core.config import settings
from app.services.llm import llm_service

class SQLAgent:
    def __init__(self):
        self.h3_df = None
        self.neighborhoods_df = None
        self._load_data()

    def _load_data(self):
        try:
            # Load H3 Data
            if settings.H3_DATA_PATH and pd.io.common.file_exists(settings.H3_DATA_PATH):
                self.h3_df = pd.read_parquet(settings.H3_DATA_PATH)
            
            # Load Neighborhoods
            if settings.NEIGHBORHOODS_PATH and pd.io.common.file_exists(settings.NEIGHBORHOODS_PATH):
                with open(settings.NEIGHBORHOODS_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert GeoJSON features to simple DataFrame for querying
                    rows = []
                    for feature in data['features']:
                        props = feature['properties']
                        rows.append(props)
                    self.neighborhoods_df = pd.DataFrame(rows)
        except Exception as e:
            print(f"Error loading data in SQLAgent: {e}")

    async def handle_request(self, query: str, context: Dict[str, Any] = None, provider: str = None) -> AsyncGenerator[str, None]:
        """
        Handles analytics queries by extracting intent and calling deterministic functions.
        """
        if self.neighborhoods_df is None:
            yield "Sorry, neighborhood data is not loaded."
            return

        # 1. Extract Parameters
        system_prompt = (
            "You are a Data Logic Assistant. Your goal is to EXTRACT analysis parameters from the user query.\n"
            "Output JSON with keys:\n"
            "- operation: 'stats', 'top_k', 'count_filter', 'lookup'\n"
            "- column: column name to analyze (e.g., 'walkability_score', 'population', 'avg_score')\n"
            "- k: integer for top_k (default 5)\n"
            "- threshold: number for filter\n"
            "- operator: '>', '<', '>=', '<=', '=='\n"
            "- neighborhood_name: if looking up specific neighborhood\n\n"
            f"Available Columns: {list(self.neighborhoods_df.columns)}\n"
            "Map 'population' to available columns if needed (or assume 'population' if not present, but function will handle error).\n"
            "Example: 'How many > 80?' -> {'operation': 'count_filter', 'column': 'walkability_score', 'threshold': 80, 'operator': '>'}\n"
            "Return ONLY JSON."
        )

        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
            from app.services.llm import llm_service
            
            param_json_str = ""
            async for chunk in llm_service.generate_response(messages, provider=provider):
                param_json_str += chunk
                
            import json
            import re
            
            # Clean JSON
            match = re.search(r"(\{.*\})", param_json_str, re.DOTALL)
            if match: param_json_str = match.group(1)
            
            params = json.loads(param_json_str)
            
            # 2. Call Function
            from app.functions.analytics import get_neighborhood_stats, get_top_k_neighborhoods, count_filtered_neighborhoods
            
            # Yield Process Indicator
            yield f"_🔍 Analyzing Request..._\n\n"
            
            op = params.get("operation")
            col = params.get("column", "walkability_score")
            
            # Yield Operation details
            yield f"Executing: `{op}` on column `{col}`\n\n"
            
            result = "No operation performed."
            
            if op == "stats":
                data = get_neighborhood_stats(self.neighborhoods_df, col)
                result = (
                    f"### Statistics for `{col}`\n"
                    "| Metric | Value |\n"
                    "| :--- | :--- |\n"
                    f"| **Mean** | {data.get('mean'):.2f} |\n"
                    f"| **Median** | {data.get('median'):.2f} |\n"
                    f"| **Max** | {data.get('max')} |\n"
                    f"| **Min** | {data.get('min')} |\n"
                )
            
            elif op == "top_k":
                k = params.get("k", 5)
                data = get_top_k_neighborhoods(self.neighborhoods_df, col, k=k)
                result = f"### Top {k} Neighborhoods by `{col}`\n\n"
                for i, item in enumerate(data, 1):
                    # Try to find a name column
                    name = item.get("name_en") or item.get("name_ar") or item.get("name") or "Unknown"
                    val = item.get(col)
                    result += f"{i}. **{name}**: {val}\n"
            
            elif op == "count_filter":
                threshold = params.get("threshold", 0)
                operator = params.get("operator", ">")
                count = count_filtered_neighborhoods(self.neighborhoods_df, col, threshold, operator)
                result = f"### Count Result\nThere are **{count}** neighborhoods with `{col} {operator} {threshold}`."
                
            elif op == "lookup":
                target = params.get("neighborhood_name")
                result = f"Lookup for **{target}** not fully implemented in demo functions yet."
                
            else:
                result = "I understood the query but couldn't map it to a specific analytic function."
                
            yield result

        except Exception as e:
            print(f"Analytics Error: {e}")
            yield f"Error processing analytics: {str(e)}"
