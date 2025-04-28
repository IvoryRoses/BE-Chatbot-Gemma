from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from huggingface_hub import login
import os
from functools import lru_cache

load_dotenv()

login(os.getenv('HUGGINGFACE_TOKEN'))

# Initialize app
app = FastAPI()

# Allow your React app to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

@lru_cache()
def get_model():
    """Load model with caching - will only load once when first needed"""
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='cpu'
    )

# Define the request body
class PromptRequest(BaseModel):
    prompt: str
    
@app.post("/chat/")
async def chat(request: PromptRequest):
    try:
        # Get or load model
        model = get_model()
        
        # Generate response
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200,
            temperature=0.7,  # Added for better response variety
            do_sample=True    # Enable sampling
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/chat/")
# async def chat(request: PromptRequest):
#     model = get_model()
#     inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(**inputs, max_new_tokens=200)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return {"response": response}