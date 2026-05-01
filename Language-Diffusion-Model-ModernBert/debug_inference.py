from transformers import AutoModelForMaskedLM
from safetensors.torch import load_file
from tokenizer import get_tokenizer
import torch

tokenizer = get_tokenizer("answerdotai/ModernBERT-base")
model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base", device_map="cuda")
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(load_file("/workspace/experiments/LDM_sft_openorca/final_model/model.safetensors"), strict=False)
model.eval()

chat_str = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is AI?"}],
    tokenize=False,
    add_generation_prompt=True
)
prompt_ids = tokenizer.encode(chat_str)
print("Prompt length:", len(prompt_ids))
print("Prompt decoded:", tokenizer.decode(prompt_ids))
print()

# Run one forward pass on a masked sequence
seq_len = 64
input_tokens = torch.full((1, seq_len), tokenizer.mask_token_id, dtype=torch.long, device="cuda")
pt = torch.tensor(prompt_ids, device="cuda")
input_tokens[0, :len(pt)] = pt
attention_mask = torch.ones((1, seq_len), dtype=torch.long, device="cuda")

with torch.no_grad():
    logits = model(input_tokens, attention_mask=attention_mask).logits

# Check what the model predicts for the first masked position after the prompt
first_mask_pos = len(prompt_ids)
probs = torch.softmax(logits[0, first_mask_pos], dim=-1)
top5 = torch.topk(probs, 5)
print("Top 5 predicted tokens at first masked position:")
for token_id, prob in zip(top5.indices.tolist(), top5.values.tolist()):
    print(f"  '{tokenizer.decode([token_id])}' (id={token_id}) prob={prob:.4f}")
