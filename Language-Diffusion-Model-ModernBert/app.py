import os
import tempfile
import torch
import gradio as gr
from transformers import AutoModelForMaskedLM
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from tokenizer import get_tokenizer
from create_gif import run_and_collect_frames, create_gif

# ---------------------------------------------------------------------------
# Config — set MODEL_REPO and HF_MODEL_NAME as Space secrets/env variables.
# MODEL_REPO  : HF Hub repo where model.safetensors lives, e.g. "you/ldm-base-sft"
# HF_MODEL_NAME : base architecture used for tokenizer + model init
# ---------------------------------------------------------------------------
MODEL_REPO    = os.environ.get("MODEL_REPO",    "your-username/ldm-modernbert-base-sft")
HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "answerdotai/ModernBERT-base")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device : {DEVICE}")
print(f"Loading weights from {MODEL_REPO} ...")

tokenizer    = get_tokenizer(HF_MODEL_NAME)
weights_path = hf_hub_download(repo_id=MODEL_REPO, filename="model.safetensors")
model        = AutoModelForMaskedLM.from_pretrained(HF_MODEL_NAME, device_map=DEVICE)
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(load_file(weights_path), strict=False)
model.tie_weights()
model.eval()

print("Model ready.")


def generate(prompt, seq_len, num_steps, strategy):

    seq_len   = int(seq_len)
    num_steps = int(num_steps)

    if not prompt.strip():
        input_tokens = torch.full((1, seq_len), tokenizer.mask_token_id,
                                   dtype=torch.long, device=DEVICE)
        mask         = torch.ones((1, seq_len), dtype=torch.bool, device=DEVICE)
        display_prompt = None
    else:
        chat       = [{"role": "user", "content": prompt}]
        prompt_ids = tokenizer.apply_chat_template(chat, tokenize=True,
                                                    add_special_tokens=True,
                                                    add_generation_prompt=True)
        input_tokens = torch.full((1, seq_len), tokenizer.mask_token_id,
                                   dtype=torch.long, device=DEVICE)
        mask         = torch.ones((1, seq_len), dtype=torch.bool, device=DEVICE)
        pt           = torch.tensor(prompt_ids, device=DEVICE)
        input_tokens[0, :len(pt)] = pt
        mask[0, :len(pt)]         = False
        display_prompt = prompt

    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=DEVICE)

    # Target ~30 GIF frames regardless of num_steps
    frame_every = max(1, num_steps // 30)

    frames, final_text = run_and_collect_frames(
        input_tokens, mask, attention_mask,
        model, tokenizer, num_steps,
        remasking=strategy, device=DEVICE,
        frame_every=frame_every,
    )

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        gif_path = f.name
    create_gif(frames, num_steps, gif_path, fps=8)

    if display_prompt:
        final_text = final_text.replace(display_prompt, "").strip()
        # strip role keywords left by chat template
        for kw in ("user", "assistant"):
            final_text = final_text.replace(kw, "").strip()

    return final_text, gif_path


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="What is artificial intelligence?  (leave empty for unconditional generation)",
        ),
        gr.Slider(64, 512, value=128, step=64,  label="Sequence Length"),
        gr.Slider(32, 256, value=64,  step=32,  label="Denoising Steps"),
        gr.Radio(["low_confidence", "random"], value="low_confidence",
                 label="Remasking Strategy"),
    ],
    outputs=[
        gr.Textbox(label="Generated Text"),
        gr.Image(label="Demasking Process", type="filepath"),
    ],
    title="Diffusion Language Model — ModernBERT",
    description=(
        "Masked diffusion language model fine-tuned on Open-Orca, built on ModernBERT-base. "
        "Generation starts from a fully masked sequence and iteratively reveals tokens. "
        "The GIF shows the unmasking process step by step."
    ),
    examples=[
        ["What is artificial intelligence?",  128, 64, "low_confidence"],
        ["Explain how black holes form.",      128, 64, "low_confidence"],
        ["Write a short poem about the ocean.", 128, 64, "low_confidence"],
        ["",                                   128, 64, "random"],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
