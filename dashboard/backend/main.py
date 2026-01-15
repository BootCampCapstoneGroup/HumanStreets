from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
import torch
import time
from threading import Thread
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager, contextmanager
import asyncio
import pandas as pd
import h3
import os
import json

# Global variables
model = None
tokenizer = None
h3_data = None # DataFrame for RAG
model_id = "LiquidAI/LFM2-1.2B"
H3_DATA_PATH = r"V:\MICS\Projects___IN_PROGRESS\DevPorj\BootCamp_Capstone_Project\idea1_walkabilityScoring\cloned\HumanStreets\dashboard\backend\riyadh_h3_r9.parquet"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print(f"Loading Base Model: {model_id}...")
    try:
        # 1. Load Base Model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype="bfloat16",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 2. Load Fine-Tuned LoRA Adapter
        # Path where the fine-tuned checkpoint is saved
        adapter_path = r"V:\MICS\Projects___IN_PROGRESS\DevPorj\BootCamp_Capstone_Project\idea1_walkabilityScoring\cloned\HumanStreets\dashboard\backend\checkpoint-226"
        
        print(f"Loading LoRA Adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        print("Model & Adapter loaded successfully.")
        
        # 3. Load H3 RAG Data
        global h3_data
        if os.path.exists(H3_DATA_PATH):
            print(f"Loading H3 Data from: {H3_DATA_PATH}")
            h3_data = pd.read_parquet(H3_DATA_PATH)
            # Ensure index if needed, but parquet preserves it if saved that way. 
            # If we didn't save index, h3_index is a column.
            # Let's ensure fast lookup by setting index if not set.
            if 'h3_index' in h3_data.columns:
                h3_data.set_index('h3_index', inplace=True)
            print(f"H3 Data loaded: {len(h3_data)} locations.")
        else:
            print(f"Warning: H3 Data file not found at {H3_DATA_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    latitude: float = Field(None, description="User latitude")
    longitude: float = Field(None, description="User longitude")

def get_location_context(lat: float, lon: float) -> str:
    global h3_data
    if h3_data is None or lat is None or lon is None:
        return ""
    
    try:
        # Resolution 9 is what we prepared
        h_idx = h3.latlng_to_cell(lat, lon, 9)
        
        if h_idx in h3_data.index:
            row = h3_data.loc[h_idx]
            # row might be a Series (one match) or DataFrame (multiple matches if duplicate index - shouldn't happen)
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            
            return row.get('text_description', "")
        else:
            return "" # "Location not covered in database."
    except Exception as e:
        print(f"Error looking up location context: {e}")
        return ""

@app.post("/chat")
async def chat(request: ChatRequest):
    global model, tokenizer
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        user_msg = request.message
        loc_context = get_location_context(request.latitude, request.longitude)
        
        if loc_context:
            # Inject context
            prompt = f"Context: {loc_context}\n\nUser: {user_msg}"
        else:
            prompt = user_msg

        # Using chat template but simpler handling for demo
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            do_sample=True,
            temperature=0.3,
            min_p=0.15,
            repetition_penalty=1.05,
            max_new_tokens=512,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        async def response_generator():
            for text in streamer:
                yield text
                await asyncio.sleep(0.01) # Small sleep to yield control

        return StreamingResponse(response_generator(), media_type="text/plain")

    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
