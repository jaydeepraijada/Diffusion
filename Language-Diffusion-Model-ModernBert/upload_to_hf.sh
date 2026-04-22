#!/bin/bash
# Upload pretrain + SFT model weights to HF Hub and push the Gradio Space.
# Usage: bash upload_to_hf.sh <hf_username>
#
# After running, go to your Space settings and add these environment variables:
#   MODEL_REPO    = <hf_username>/ldm-modernbert-base-sft
#   HF_MODEL_NAME = answerdotai/ModernBERT-base

set -e

HF_USERNAME=${1:?"Usage: bash upload_to_hf.sh <hf_username>"}

PRETRAIN_REPO="$HF_USERNAME/ldm-modernbert-base-pretrain"
SFT_REPO="$HF_USERNAME/ldm-modernbert-base-sft"
SPACE_REPO="$HF_USERNAME/ldm-modernbert-demo"

echo ">>> Logging in to HuggingFace"
huggingface-cli login

# ── Upload pretrain weights ──────────────────────────────────────────────────
echo ">>> Uploading pretrain model to $PRETRAIN_REPO"
huggingface-cli repo create "$PRETRAIN_REPO" --type model --exist-ok
huggingface-cli upload "$PRETRAIN_REPO" \
    /workspace/experiments/LDM_pretrain_base/final_model/model.safetensors \
    model.safetensors \
    --repo-type model

# ── Upload SFT weights ───────────────────────────────────────────────────────
echo ">>> Uploading SFT model to $SFT_REPO"
huggingface-cli repo create "$SFT_REPO" --type model --exist-ok
huggingface-cli upload "$SFT_REPO" \
    /workspace/experiments/LDM_sft_openorca/final_model/model.safetensors \
    model.safetensors \
    --repo-type model

# ── Push Gradio Space ────────────────────────────────────────────────────────
echo ">>> Creating Space $SPACE_REPO"
huggingface-cli repo create "$SPACE_REPO" --type space --space_sdk gradio --exist-ok

SPACE_DIR=$(mktemp -d)
git clone "https://huggingface.co/spaces/$SPACE_REPO" "$SPACE_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$SCRIPT_DIR/app.py"        "$SPACE_DIR/"
cp "$SCRIPT_DIR/create_gif.py" "$SPACE_DIR/"
cp "$SCRIPT_DIR/tokenizer.py"  "$SPACE_DIR/"
cp "$SCRIPT_DIR/requirements.txt" "$SPACE_DIR/"

cd "$SPACE_DIR"
git add .
git commit -m "Add diffusion LM Space"
git push

echo ""
echo "Done!"
echo "  Pretrain model : https://huggingface.co/$PRETRAIN_REPO"
echo "  SFT model      : https://huggingface.co/$SFT_REPO"
echo "  Space          : https://huggingface.co/spaces/$SPACE_REPO"
echo ""
echo "  Next: in the Space settings, add these env variables:"
echo "    MODEL_REPO    = $SFT_REPO"
echo "    HF_MODEL_NAME = answerdotai/ModernBERT-base"
