import os
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# 1. Setup Keys
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
# Use the OpenRouter key we saw earlier if not in env, or rely on env
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY") or "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"

print(f"Gemini Key Present: {bool(GEMINI_KEY)}")
print(f"OpenRouter Key Present: {bool(OPENROUTER_KEY)}")
print("-" * 50)

# 2. Test Gemini
print("\nTesting Google Gemini (gemini-2.0-flash)...")
if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content("Hi")
        print(f"✅ Gemini Success: {resp.text.strip()[:50]}...")
    except Exception as e:
        print(f"❌ Gemini Failed: {e}")
else:
    print("⏭️ Skipping Gemini (No Key)")

# 3. Test OpenRouter Models
print("\nTesting OpenRouter Free Models...")
if OPENROUTER_KEY:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
    )
    
    # List of free models to try - verified from gemy.py output
    candidates = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "deepseek/deepseek-r1-0528:free",
        "meta-llama/llama-3.2-3b-instruct:free",
    ]

    for model_id in candidates:
        print(f"\nProbing {model_id}...")
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "Hi"}],
                stream=False
            )
            content = resp.choices[0].message.content
            print(f"✅ SUCCESS: {model_id}")
            print(f"   Response: {content.strip()[:50]}...")
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                print(f"❌ Rate Limited (429): {err_str[:100]}...")
            elif "402" in err_str or "403" in err_str:
                print(f"❌ Payment/Quota (402/403): {err_str[:100]}...")
            else:
                print(f"❌ Failed: {err_str[:100]}...")
else:
    print("⏭️ Skipping OpenRouter (No Key)")
