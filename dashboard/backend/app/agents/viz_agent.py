from typing import Dict, Any, AsyncGenerator
from app.services.llm import llm_service
from app.services.spatial import spatial_service

class VizAgent:
    def __init__(self):
        pass

    async def handle_request(self, query: str, context: Dict[str, Any] = None, provider: str = None) -> AsyncGenerator[str, None]:
        """
        Generates Deck.gl layer configurations or view states based on instruction.
        """
        # Determine intent...
        
        # Strict Verification Logic
        system_prompt = (
            "You are a Visualization Assistant. You control the map.\n"
            f"Available Layers: {spatial_service.available_layers}\n"
            "**FORMATTING RULES:**\n"
            "- Use **Markdown** for all text responses.\n"
            "- Use **bold** for layer names.\n\n"
            "**Supported Commands** (append to end of response):\n"
            "- `[[SHOW_LAYER: NEIGHBORHOODS]]`: Show neighborhoods polygon layer.\n"
            "- `[[SHOW_LAYER: WALKABILITY]]`: Show H3 hex heatmap layer.\n"
            "- `[[SHOW_LAYER: QUERY_RESULT]]`: Show the result of the last SQL query (e.g. filtered neighborhoods).\n"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # Buffer and Post-Process for Robustness
        full_response = ""
        async for chunk in llm_service.generate_response(messages, provider=provider):
            # Since we can't easily modify the stream on the fly without buffering, 
            # and the local model is hallucinating, let's buffer the response *inside* this generator,
            # fix it, then yield.
            yield chunk

    # Wait, the above replacement merely yields. I need to implement the fix.
    # I will replace the streaming loop with a buffer-then-yield approach for the Local Model case?
    # Or just yield corrected tags.
    
    # Let's try to implement a robust handler that buffers.
    async def handle_request_robust(self, query: str, context: Dict[str, Any] = None, provider: str = None) -> AsyncGenerator[str, None]:
         # ... (previous setup) ...
         full_response = ""
         async for chunk in llm_service.generate_response(messages, provider=provider):
             full_response += chunk
         
         # Post-Process
         final_response = full_response
         
         # 1. Alias Correction
         if "[[SHOW_LAYER: DISTRICTS]]" in final_response.upper():
             final_response = final_response.replace("[[SHOW_LAYER: DISTRICTS]]", "[[SHOW_LAYER: NEIGHBORHOODS]] (Aliased from Districts)")
             final_response = final_response.replace("[[SHOW_LAYER: districts]]", "[[SHOW_LAYER: NEIGHBORHOODS]] (Aliased from Districts)")

          # 2. Invalid Layer Check
         import re
         def replace_invalid_layer(match):
             layer_tag = match.group(0) # Full tag e.g. [[SHOW_LAYER: XYZ]]
             layer_name = match.group(1).upper() # XYZ
             
             if layer_name in spatial_service.available_layers or layer_name in ["QUERY_RESULT", "NEIGHBORHOODS"]:
                 return layer_tag # Keep valid
             else:
                 return f"\n> ⚠️ **Error**: Layer '{layer_name}' is not available."

         final_response = re.sub(r"\[\[SHOW_LAYER:\s*(\w+)\s*\]\]", replace_invalid_layer, final_response, flags=re.IGNORECASE)

         yield final_response
