from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(path_or_fileobj="inference.gif", path_in_repo="inference.gif", repo_id="JaydeepR/ldm-modernbert-base-sft")
api.upload_file(path_or_fileobj="MODEL_CARD.md", path_in_repo="README.md", repo_id="JaydeepR/ldm-modernbert-base-sft")
print("Done!")
