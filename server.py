from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from huggingface_hub import login
import os

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

# Load model
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
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
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}