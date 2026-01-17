import asyncio
import os
import time
from threading import Thread
import google.generativeai as genai
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from app.core.config import settings

class LLMService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def initialize_models(self):
        """Initializes Gemini and Local Model."""
        
        # Check GPU status
        try:
            import torch
            if torch.cuda.is_available():
                print(f"🔥 CUDA Available! Device Count: {torch.cuda.device_count()}")
                print(f"   Current Device: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️ CUDA NOT Available. Running on CPU.")
        except Exception as e:
            print(f"Error checking CUDA: {e}")

        # Setup Gemini
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            print("Gemini Configured.")
        else:
            print("Warning: GEMINI_API_KEY not found")

        # Load Local Model
        try:
            print(f"Loading Base Model: {settings.MODEL_ID}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_ID,
                device_map="auto",
                dtype="bfloat16",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID)
            
            print(f"Loading LoRA Adapter from: {settings.ADAPTER_PATH}")
            self.model = PeftModel.from_pretrained(base_model, settings.ADAPTER_PATH)
            print("Model & Adapter loaded successfully.")
        except Exception as e:
            print(f"Error loading local model: {e}")

    async def generate_response(self, messages: list[dict[str, str]], provider: str = None):
        """Generates a streaming response based on the provider."""
        # Default provider logic
        if not provider:
            # Prefer Local if loaded
            if self.model is not None:
                provider = "local"
            # Else Gemini if key
            elif settings.GEMINI_API_KEY:
                provider = "gemini"
            # Else OpenRouter
            else:
                provider = "openrouter_free"

        stream = None
        # --- FREE LLM (OPENROUTER) ---
        if provider == "openrouter_free":
            stream = self._stream_openrouter(messages, model="meta-llama/llama-3.2-3b-instruct:free")

        # --- DEEPSEEK FREE (OPENROUTER) ---
        elif provider == "deepseek_free":
            stream = self._stream_openrouter(messages, model="deepseek/deepseek-r1-0528:free")
        
        # --- OPENROUTER (Generic) ---
        elif provider == "openrouter":
             # Use a default decent model or configurable
             stream = self._stream_openrouter(messages, model="google/gemini-2.0-flash-lite-preview-02-05:free")

        # --- GEMINI ---
        elif provider == "gemini":
            stream = self._stream_gemini(messages)

        # --- LOCAL ---
        elif provider == "local":
            stream = self._stream_local(messages)

        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        if stream:
            async for chunk in stream:
                yield chunk

    async def _stream_openrouter(self, messages: list[dict[str, str]], model: str):
        api_key = settings.OPENROUTER_API_KEY or "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"
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False 
            )
            content = completion.choices[0].message.content
            if content:
                yield content
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg:
                yield "Rate limit exceeded (Free Tier). Please wait 10-20 seconds and try again."
            elif "403" in err_msg or "402" in err_msg:
                yield "Free quota exceeded or Model unavailable. Try switching to Local model."
            else:
                yield f"Provider Error: {err_msg}"

    async def _stream_gemini(self, messages: list[dict[str, str]]):
        if not settings.GEMINI_API_KEY:
            yield "Stack Error: Gemini API Key not configured."
            return

        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Convert messages to Gemini format (simple text for now)
        full_prompt = ""
        for msg in messages:
            role = msg['role'].upper()
            content = msg['content']
            full_prompt += f"{role}: {content}\n"

        try:
            response = gemini_model.generate_content(full_prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    await asyncio.sleep(0.01)
        except Exception as e:
             yield f"Gemini Error: {e}"

    async def _stream_local(self, messages: list[dict[str, str]]):
        if self.model is None or self.tokenizer is None:
            yield "Stack Error: Local model not loaded."
            return

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            do_sample=True,
            temperature=0.3,
            min_p=0.15,
            repetition_penalty=1.05,
            max_new_tokens=512,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            yield text
            await asyncio.sleep(0.01)

llm_service = LLMService()
