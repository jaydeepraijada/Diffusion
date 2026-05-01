from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ["HF_TOKEN"])

# SFT repo
print("Uploading to SFT repo...")
api.upload_file(path_or_fileobj="inference_sft.gif", path_in_repo="inference.gif", repo_id="JaydeepR/ldm-modernbert-base-sft")
api.upload_file(path_or_fileobj="MODEL_CARD.md", path_in_repo="README.md", repo_id="JaydeepR/ldm-modernbert-base-sft")

# Pretrain repo
print("Uploading to pretrain repo...")
api.upload_file(path_or_fileobj="inference_pretrain.gif", path_in_repo="inference.gif", repo_id="JaydeepR/ldm-modernbert-base-pretrain")
api.upload_file(path_or_fileobj="MODEL_CARD_PRETRAIN.md", path_in_repo="README.md", repo_id="JaydeepR/ldm-modernbert-base-pretrain")

print("Done!")
