from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

def initialize_model():
    model_name = "google/gemma-2b-it"
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='cpu'
    )
    return model, tokenizer

def generate_response(prompt, model, tokenizer):
    start_time = time.time() # Start time for generation
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    end_time = time.time() # End time
    generation_time = end_time - start_time
    return response, generation_time

def main():
    model, tokenizer = initialize_model()
    print("\nGemma Chat CLI - Type 'quit' to exit")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            break
        
        print("\nGenerating response...")
        response, generation_time = generate_response(user_input, model, tokenizer)
        print(f"\nGemma: {response}")
        print(f"\nGeneration time: {generation_time:.2f} seconds")

if __name__ == "__main__":
    main()