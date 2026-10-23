from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def retrieve_kv_cache_hugging_face_pytorch(model_name : str, prompt : str):

    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, 
                        use_cache=True)

    cache = outputs.past_key_values

    return cache