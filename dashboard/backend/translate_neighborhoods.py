import asyncio
import json
import os
from app.services.llm import llm_service
from app.core.config import settings

# Initialize LLM
llm_service.initialize_models()

async def translate_batch():
    # Load GeoJSON
    with open(settings.NEIGHBORHOODS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    neighborhoods = data['features']
    print(f"Loaded {len(neighborhoods)} neighborhoods.")

    # Prepare list of names
    arabic_names = []
    for f in neighborhoods:
        props = f['properties']
        if 'name' in props and props['name']:
            arabic_names.append(props['name'])

    # Chunking to avoid context limits and rate limits
    chunk_size = 10
    chunks = [arabic_names[i:i + chunk_size] for i in range(0, len(arabic_names), chunk_size)]

    translations = {}
    import time

    for i, chunk in enumerate(chunks):
        print(f"Translating chunk {i+1}/{len(chunks)}...")
        prompt = (
            "Translate the following list of Riyadh neighborhood names from Arabic to English. "
            "Return ONLY a JSON object where keys are the Arabic names and values are the English translations. "
            "Do not add any markdown formatting or extra text.\n\n"
            f"{json.dumps(chunk, ensure_ascii=False)}"
        )

        # Try Gemini, then fallback to OpenRouter
        providers = ["gemini", "openrouter"]
        if not settings.GEMINI_API_KEY:
            providers.remove("gemini")
        
        success = False
        for provider in providers:
            if success: break
            try:
                print(f"Trying provider: {provider}")
                # For non-streaming, we just iterate the stream
                response_stream = await llm_service.generate_response([{"role": "user", "content": prompt}], provider=provider)
                response_text = ""
                async for chunk_text in response_stream:
                    response_text += chunk_text
                
                # Clean response
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                chunk_trans = json.loads(response_text)
                translations.update(chunk_trans)
                print(f"Translated {len(chunk_trans)} items.")
                success = True
            except Exception as e:
                print(f"Error with provider {provider} on chunk {i}: {e}")
                time.sleep(5) # Wait before retry

        if not success:
             print(f"Failed to translate chunk {i} after trying all providers.")
        
        # Rate limit safety
        time.sleep(5)
 

    # Update GeoJSON
    count = 0
    for f in neighborhoods:
        name_ar = f['properties'].get('name')
        if name_ar and name_ar in translations:
            f['properties']['name_en'] = translations[name_ar]
            count += 1
    
    print(f"Updated {count} neighborhoods with English names.")

    # Save
    with open(settings.NEIGHBORHOODS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Saved updated GeoJSON.")

if __name__ == "__main__":
    asyncio.run(translate_batch())
