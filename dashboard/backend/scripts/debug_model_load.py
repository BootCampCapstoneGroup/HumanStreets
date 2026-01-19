
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from app.core.config import settings

def test_load():
    print(f"Transformers Version: {import_transformers()}")
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    model_id = settings.MODEL_ID
    adapter_path = settings.ADAPTER_PATH
    
    print(f"\nAttempting to load base model: {model_id}")
    try:
        # Mimic exact call from llm.py
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype="bfloat16",
            trust_remote_code=True,
        )
        print("✅ Base Model Loaded")
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        return

    print("\nAttempting to load tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("✅ Tokenizer Loaded")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        
    print(f"\nAttempting to load adapter from: {adapter_path}")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✅ Adapter Loaded")
    except Exception as e:
        print(f"❌ Failed to load adapter: {e}")

def import_transformers():
    import transformers
    return transformers.__version__

if __name__ == "__main__":
    test_load()
