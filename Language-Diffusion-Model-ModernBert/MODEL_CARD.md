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

# LDM-ModernBERT — Language Diffusion Model

A language diffusion model built on [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base), pretrained on Project Gutenberg and fine-tuned on Open-Orca for instruction following.

Unlike autoregressive models that generate text left-to-right, this model generates text through iterative **denoising** — starting from a fully masked sequence and progressively unmasking tokens until a coherent output emerges.

![Inference GIF](inference.gif)

---

## Model Details

| Property | Value |
|---|---|
| Base model | ModernBERT-base |
| Parameters | ~150M |
| Architecture | Masked Language Model (diffusion objective) |
| Pretrain data | Project Gutenberg (6,400,553 train chunks, seq_len=1024) |
| SFT data | Open-Orca (~4.2M Q&A pairs) |
| Pretrain steps | 30,000 |
| SFT steps | 10,000 |
| Effective batch size | 128 |
| Pretrain LR | 5e-5 (cosine, 1500 warmup steps) |
| SFT LR | 1e-5 (cosine, 300 warmup steps) |
| Hardware | RTX 4090 24GB |
| Pretrain time | ~20 hours |
| SFT time | ~4.3 hours |

---

## Training

### Pretraining
The model is pretrained using a **flow-matching diffusion objective**: at each step, a random fraction `t` of tokens is masked, and the model learns to predict the original tokens. The loss is scaled by `1/t` to account for the difficulty of predicting heavily masked sequences.

- Dataset: Project Gutenberg (6,400,553 train chunks, 34,287 test chunks)
- Initial train loss: 3.887 | Initial val loss: 3.922
- Final train loss: 2.917 | Final val loss: 2.962

### SFT (Supervised Fine-Tuning)
Fine-tuned on Open-Orca instruction-response pairs. Loss is computed only on the response tokens (not the instruction), using a query mask to identify answer boundaries.

- Dataset: Open-Orca (~4.2M Q&A pairs)
- Initial train loss: 1.559 | Initial val loss: 1.333
- Final train loss: 0.837 | Final val loss: 0.967

---

## Inference

The model supports two generation strategies:

- **`random`** — masked tokens are randomly re-masked at each step
- **`low_confidence`** — the lowest confidence tokens are re-masked, leading to more coherent outputs

### Quickstart

```python
from transformers import AutoModelForMaskedLM
from safetensors.torch import load_file
import torch

# Load model
model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict, strict=False)
model.eval()
```

Or use the provided inference scripts:

```bash
# Interactive inference
bash inference.sh

# Generate GIF
bash create_gif.sh
```

---

## Limitations

- Trained on a relatively small dataset (Project Gutenberg) with limited steps — quality is lower than production-scale models
- SFT data was truncated to 1024 tokens; very long responses may be cut off
- No RLHF or safety fine-tuning applied

---

## Citation

Built following the approach from:
- [Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- [PyTorch-Adventures — Language Diffusion Model](https://github.com/priyammaz/PyTorch-Adventures/tree/main/PyTorch%20for%20NLP/Language%20Diffusion%20Model) by [@priyammaz](https://github.com/priyammaz)
