Installation step by step

cd BEChatbotGemma
source ./Scripts/activate
pip install fastapi uvicorn transformers torch accelerate
pip install pydantic
pip install huggingface_hub
huggingface-cli login

for running

uvicorn server:app --reload --host 0.0.0.0 --port 8000
