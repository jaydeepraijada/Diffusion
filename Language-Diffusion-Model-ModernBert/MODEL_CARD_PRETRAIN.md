---
language: en
tags:
  - diffusion
  - language-model
  - masked-language-model
  - modernbert
  - text-generation
license: apache-2.0
---

# LDM-ModernBERT — Pretrained Language Diffusion Model

A language diffusion model built on [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base), pretrained on Project Gutenberg using a masked diffusion objective.

This is the **base pretrained checkpoint** before SFT instruction tuning. For instruction following, see [JaydeepR/ldm-modernbert-base-sft](https://huggingface.co/JaydeepR/ldm-modernbert-base-sft).

![Inference GIF](inference.gif)

---

## Model Details

| Property | Value |
|---|---|
| Base model | ModernBERT-base |
| Parameters | ~150M |
| Architecture | Masked Language Model (diffusion objective) |
| Pretrain data | Project Gutenberg (6,400,553 train chunks, seq_len=1024) |
| Pretrain steps | 30,000 |
| Effective batch size | 128 |
| Learning rate | 5e-5 (cosine, 1500 warmup steps) |
| Hardware | RTX 4090 24GB |
| Training time | ~20 hours |
| Initial train loss | 3.887 |
| Initial val loss | 3.922 |
| Final train loss | 2.917 |
| Final val loss | 2.962 |

---

## Training

The model is pretrained using a **flow-matching diffusion objective**: at each step, a random fraction `t` of tokens is masked, and the model learns to predict the original tokens. The loss is scaled by `1/t` to account for the difficulty of predicting heavily masked sequences.

---

## Inference

```python
from transformers import AutoModelForMaskedLM
from safetensors.torch import load_file
import torch

model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict, strict=False)
model.eval()

# Unconditional generation — start from all masked tokens
seq_len = 128
input_tokens = torch.full((1, seq_len), tokenizer.mask_token_id, dtype=torch.long)
```

Or use the provided scripts from the [GitHub repo](https://github.com/jaydeepraijada/Diffusion):

```bash
# Generate GIF (unconditional)
bash create_gif.sh
```

---

## Limitations

- Trained on a relatively small dataset (Project Gutenberg) with limited steps
- No instruction tuning — use the SFT checkpoint for Q&A tasks
- Output has a literary/formal style reflecting Gutenberg training data

---

## Citation

Built following the approach from:
- [Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- [PyTorch-Adventures — Language Diffusion Model](https://github.com/priyammaz/PyTorch-Adventures/tree/main/PyTorch%20for%20NLP/Language%20Diffusion%20Model) by [@priyammaz](https://github.com/priyammaz)
