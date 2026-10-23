from util.common import retrieve_kv_cache_hugging_face_pytorch

model_family_one = "Qwen/Qwen2.5-0.5B-Instruct"
model_family_two = "HuggingFaceTB/SmolLM2-360M-Instruct"

prompt = "Explain what a kv cache is."

qwen_cache = retrieve_kv_cache_hugging_face_pytorch(model_family_one, prompt)
smol_cache = retrieve_kv_cache_hugging_face_pytorch(model_family_two, prompt)

print("Qwen Cached tokens:", qwen_cache.get_seq_length())
print("Qwen Layers:", len(qwen_cache.layers))

print("SmolLM2 Cached tokens:", smol_cache.get_seq_length())
print("SmolLM2 Layers:", len(smol_cache.layers))