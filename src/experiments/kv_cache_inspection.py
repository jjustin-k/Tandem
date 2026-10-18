from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. Define the model repository ID from Hugging Face
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# 2. Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)
model.eval()

inputs = tokenizer("Explain KV caching.", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, 
                    use_cache=True)

cache = outputs.past_key_values

print("Raw cache:", cache)
print("Cached tokens:", cache.get_seq_length())
print("Layers:", len(cache.layers))