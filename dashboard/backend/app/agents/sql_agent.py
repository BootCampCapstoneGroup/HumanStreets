import json
import re
from typing import Dict, Any, AsyncGenerator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import geopandas as gpd
from app.core.config import settings
from app.services.llm import llm_service
from app.services.spatial import spatial_service

class SQLAgent:
    def __init__(self):
        self.engine = create_engine(settings.DATABASE_URL)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._neighborhood_names = []
        
    @property
    def schema_description(self):
        return self._get_schema_info()

    def _get_neighborhood_names(self):
        """
        Fetches all neighborhood names to provide context for fuzzy matching.
        Caches the result after the first fetch.
        """
        if self._neighborhood_names:
            return self._neighborhood_names
        
        print("[DEBUG] Fetching neighborhood names for Context Injection...")
        db = self.SessionLocal()
        try:
            res = db.execute(text("SELECT name FROM neighborhoods")).fetchall()
            self._neighborhood_names = [r[0] for r in res if r[0]]
            print(f"[DEBUG] Loaded {len(self._neighborhood_names)} neighborhood names.")
            return self._neighborhood_names
        except Exception as e:
            print(f"[ERROR] Failed to fetch neighborhoods: {e}")
            return []
        finally:
            db.close()

    def _get_schema_info(self):
        schema = (
            "Table: neighborhoods\n"
            "Columns: name (text), geometry (geometry), avg_walkability (float 0-100)\n"
            "⚠️ **IMPORTANT**: This table ALREADY contains avg_walkability scores - NO JOINS NEEDED for neighborhood lookups!\n\n"
            "Table: h3_grid (for detailed H3-level analysis ONLY)\n"
            "Columns: h3_index (text), avg_street_score (float), neighborhood_name_ar (text), geometry (geometry)\n"
            "Note: Use h3_grid ONLY if user asks for H3-level hexagon data. Do NOT join with neighborhoods for simple lookups.\n"
        )
        return schema

    async def handle_request(self, query: str, context: Dict[str, Any] = None, provider: str = None) -> AsyncGenerator[str, None]:
        schema = self._get_schema_info()
        valid_names = self._get_neighborhood_names()
        valid_names_str = ", ".join(valid_names)
        
        system_prompt = (
            "You are a PostGIS SQL Expert. Convert user questions into SQL.\n"
            "**Database Schema:**\n"
            f"{schema}\n"
            "**Rules:**\n"
            "1. Output ONLY the SQL query inside ```sql ... ```.\n"
            "2. Table: `neighborhoods` has `geometry` and `avg_walkability`.\n"
            "3. Select `geometry AS geom` for maps.\n"
            "4. Search names with ILIKE and wildcards (e.g. '%Name%').\n"
            "5. Use these Exact Arabic Names:\n"
            f"   [{valid_names_str}]\n"
        )
        
        # 1. Debug: Thinking
        yield "[[DEBUG: Thinking (SQL Generation)...\n]]"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
        
        generated_text = ""
        async for chunk in llm_service.generate_response(messages, provider=provider):
            generated_text += chunk
        
        sql_match = re.search(r"```sql\n(.*?)\n```", generated_text, re.DOTALL)
        if not sql_match:
             if "SELECT" in generated_text.upper():
                 sql_query = generated_text
             else:
                 yield f"[[DEBUG: I couldn't generate a valid SQL query. Response: {generated_text}]]"
                 return
        else:
            sql_query = sql_match.group(1)
            
        yield f"[[DEBUG: Generated SQL:\n```sql\n{sql_query}\n```\n\n]]"
        
        # 2. Debug: Executing
        yield "[[DEBUG: Executing Query...\n]]"
        
        try:
            # 1b. Clean 'حي'
            if 'حي' in sql_query or 'حى' in sql_query:
                original_sql = sql_query
                sql_query = re.sub(r"%\s*ح[يى]\s+", "%", sql_query)
                sql_query = re.sub(r"'\s*ح[يى]\s+", "'", sql_query)
                if original_sql != sql_query:
                    yield f"[[DEBUG: (Auto-corrected: stripped 'حي' prefix)\n]]"
            
            # 2. Fix Geometry Column Logic
            is_spatial = 'geom' in sql_query.lower() or 'geometry' in sql_query.lower()
            
            if is_spatial:
                sql_lower = sql_query.lower()
                sql_query_fixed = sql_query
                
                # Rule: Force `geom` (with any alias) to be `geometry AS geom`
                # e.g. `geom AS my_geo` -> `geometry AS geom`
                # e.g. `geom` -> `geometry AS geom`
                # e.g. `output.geom` -> `output.geometry AS geom`
                if 'geom' in sql_lower:
                     # Replace `geom` optionally followed by alias (AS \w+) with `geometry AS geom`
                     # Capture word boundary before geom to support table aliases (e.g. t.geom) - wait, \b matches after .
                     sql_query_fixed = re.sub(r'\bgeom\b(?:\s+AS\s+\w+)?', 'geometry AS geom', sql_query, flags=re.IGNORECASE)
                
                # If LLM used `geometry` without alias, rename to `geom` for consistency
                # e.g. `geometry` -> `geometry AS geom`
                # e.g. `geometry AS foo` -> `geometry AS geom` (Optional?)
                # Standardizing is safer.
                if 'geometry' in sql_lower and 'as geom' not in sql_query_fixed.lower():
                     sql_query_fixed = re.sub(r'\bgeometry\b(?:\s+AS\s+\w+)?', 'geometry AS geom', sql_query_fixed, flags=re.IGNORECASE)

                if sql_query != sql_query_fixed:
                     sql_query = sql_query_fixed
                     yield f"[[DEBUG: (Auto-corrected to use 'geometry AS geom')\n]]"
                
                # Always use 'geom' now because we forced it
                geom_col = 'geom'
                
                with self.engine.connect() as conn:
                    gdf = gpd.read_postgis(text(sql_query), conn, geom_col=geom_col)
                    
                    if gdf.empty:
                        yield "Query executed but returned no results. Check if the neighborhood name is correct."
                    else:
                        geojson = json.loads(gdf.to_json())
                        spatial_service.set_query_result(geojson)
                        count = len(gdf)
                        yield f"✅ Found {count} spatial features.\n\n"
                        yield "[[SHOW_LAYER: QUERY_RESULT]]"
            else:
                with self.engine.connect() as conn:
                    result = conn.execute(text(sql_query))
                    rows = result.fetchall()
                    keys = result.keys()
                    
                    if not rows:
                        yield "Query returned no results."
                    else:
                        # Convert results to a simple string format for the LLM
                        data_str = str([dict(zip(keys, row)) for row in rows])
                        
                        prompt = (
                            f"User Request: {query}\n"
                            f"Data: {data_str}\n\n"
                            "Answer the request using the Data. Be concise."
                        )
                        
                        yield "[[DEBUG: Generating Natural Language Response...]]\n"
                        messages = [{"role": "user", "content": prompt}]
                        async for chunk in llm_service.generate_response(messages, provider=provider):
                            yield chunk

        except Exception as e:
            yield f"[[DEBUG: ❌ SQL Execution Error: {e}]]"
            yield f"❌ Error executing query. ({str(e)})"
