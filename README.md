# Diffusion LM — TinyStories

A masked-diffusion language model trained from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

Instead of generating tokens left-to-right like a standard LM, this model starts with a fully masked sequence and progressively unmasks tokens over T diffusion steps — refining the output iteratively until the sequence is complete.

**Model on Hugging Face:** [JaydeepR/diffusion-lm-tinystories](https://huggingface.co/JaydeepR/diffusion-lm-tinystories)

![Diffusion inference](https://huggingface.co/JaydeepR/diffusion-lm-tinystories/resolve/main/inference.gif)

---

## Architecture

| Param | Value |
|---|---|
| Parameters | ~45M |
| Hidden dim | 512 |
| Layers | 10 |
| Attention heads | 8 |
| FFN dim | 2048 |
| Diffusion steps T | 128 |
| Sequence length | 256 |
| Vocab size | 26,000 |

The model is a Transformer encoder with:
- Byte-level BPE tokenizer trained from scratch
- Token + positional + timestep embeddings
- Linear mask-ratio schedule during training
- Weight-tied input/output embeddings

---

## How it works

**Training:** At each step, a random fraction of tokens is masked based on a linear schedule. The model learns to predict the original tokens from the corrupted sequence.

**Inference:** Starts with a fully masked sequence (except the prompt). At each diffusion step:
1. Model predicts all masked tokens simultaneously
2. Tokens are filled in from most → least confident
3. Least confident tokens are re-masked for the next step
4. Repeats until fully unmasked

---

## Training

| Setting | Value |
|---|---|
| Dataset | 1M TinyStories examples |
| Train steps | 60,000 |
| Batch size | 32 |
| Grad accumulation | 2 (effective batch 64) |
| Optimizer | AdamW |
| Learning rate | 2e-4 (cosine decay) |
| Warmup steps | 1,000 |
| Weight decay | 0.1 |
| Mixed precision | bf16 |
| Hardware | NVIDIA RTX 3090 24GB |

### Validation Loss

| Step | Val Loss |
|------|----------|
| 5,000 | 6.0313 |
| 10,000 | 5.9045 |
| 15,000 | 5.6092 |
| 20,000 | 4.4481 |
| 25,000 | 3.8447 |
| 30,000 | 3.6634 |
| 35,000 | 3.5419 |
| 40,000 | 3.3554 |
| 45,000 | 3.2779 |
| 50,000 | 3.1767 |
| 55,000 | 3.1012 |
| 60,000 | 3.1067 |

---

## Running the Notebook

### Requirements
- NVIDIA GPU with 16GB+ VRAM (tested on RTX 3090)
- CUDA 12+

### Setup
Open `Diffusion_LLM_from_Scratch_TinyStories.ipynb` in JupyterLab or VS Code.

Set the run mode in the first cell:

```python
RUN_MODE = 'quick'       # 2k steps, ~15 min, good for testing
RUN_MODE = 'budget_100'  # 60k steps, ~3 hrs, full training run
```

Then run all cells — the notebook handles everything:
- Installs dependencies
- Downloads TinyStories
- Trains a tokenizer from scratch
- Builds and trains the model
- Runs inference and saves `inference.gif`
- Saves checkpoint to `checkpoints/final/`

### Recommended: Run on RunPod
Use an RTX 3090 (24GB) pod with the PyTorch template. Upload the notebook via JupyterLab and run all cells.

---

## Repository Structure

```
Diffusion_LLM_from_Scratch_TinyStories.ipynb  # full pipeline in one notebook
```

Trained weights, tokenizer, and config are on [Hugging Face](https://huggingface.co/JaydeepR/diffusion-lm-tinystories).
