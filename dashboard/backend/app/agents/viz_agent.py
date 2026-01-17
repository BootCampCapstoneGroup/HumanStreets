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
        # Determine intent
        # 1. Toggle Layers
        # 2. Highlight Areas (Future)
        # 3. Fly To (Future)
        
        system_prompt = (
            "You are a Visualization Assistant. You control the map.\n"
            "**FORMATTING RULES:**\n"
            "- Use **Markdown** for all text responses.\n"
            "- Use **bold** for layer names.\n\n"
            "**Supported Commands** (append to end of response):\n"
            "- `[[SHOW_LAYER: NEIGHBORHOODS]]`: Show neighborhoods polygon layer.\n"
            "- `[[SHOW_LAYER: DISTRICTS]]`: Alias for neighborhoods.\n"
            "- `[[SHOW_LAYER: WALKABILITY]]`: Show H3 hex heatmap layer.\n\n"
            "If the user asks to see a specific layer, confirm it in text (using Markdown) and append the tag."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        async for chunk in llm_service.generate_response(messages, provider=provider):
            yield chunk
