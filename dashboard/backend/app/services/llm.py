import asyncio
import os
import time
from threading import Thread
from google import genai
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from app.core.config import settings

class LLMService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.gemini_client = None
        
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
            try:
                self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
                print("Gemini Client Configured.")
            except Exception as e:
                print(f"Error configuring Gemini Client: {e}")
        else:
            print("Warning: GEMINI_API_KEY not found")

        # Load Local Model
        try:
            print(f"Loading Base Model: {settings.MODEL_ID}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_ID,
                device_map="auto",
                dtype="bfloat16",
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID, trust_remote_code=True)
            
            try:
                print(f"Loading LoRA Adapter from: {settings.ADAPTER_PATH}")
                self.model = PeftModel.from_pretrained(base_model, settings.ADAPTER_PATH)
                print("Model & Adapter loaded successfully.")
            except Exception as e:
                print(f"⚠️ Warning: Failed to load LoRA adapter: {e}")
                print("Falling back to Base Model only.")
                self.model = base_model
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
        if not self.gemini_client:
             yield "Stack Error: Gemini Client not initialized."
             return

        # Convert messages to Gemini format (simple text for now)
        full_prompt = ""
        for msg in messages:
            role = msg['role'].upper()
            content = msg['content']
            full_prompt += f"{role}: {content}\n"

        try:
            # streaming with google-genai SDK
            response = self.gemini_client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=full_prompt
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    await asyncio.sleep(0.01)
        except Exception as e:
             yield f"Gemini Error: {e}"

    async def _stream_local(self, messages: list[dict[str, str]]):
        if self.model is None or self.tokenizer is None:
            # Fallback to Gemini or OpenRouter
            print("⚠️ Local model not loaded. Falling back to Gemini/OpenRouter.")
            if self.gemini_client:
                async for chunk in self._stream_gemini(messages):
                     yield chunk
            else:
                async for chunk in self._stream_openrouter(messages, model="google/gemini-2.0-flash-lite-preview-02-05:free"):
                     yield chunk
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
