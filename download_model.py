# download_model.py
import os
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

REPO_ID = "Prady029/AyurParam-2.9b-it-gguf"
LOCAL_DIR = "models"

os.makedirs(LOCAL_DIR, exist_ok=True)

api = HfApi(token=HF_TOKEN)
files = api.list_repo_files(REPO_ID)
gguf_files = [f for f in files if f.endswith(".gguf")]

if not gguf_files:
    raise RuntimeError(f"No .gguf file found in {REPO_ID}")

print("Found GGUF file(s):", gguf_files)

target_file = gguf_files[0]
print(f"Downloading: {target_file} ...")

path = hf_hub_download(
    repo_id=REPO_ID,
    filename=target_file,
    local_dir=LOCAL_DIR,
    token=HF_TOKEN,
)

print(f"\nDownloaded to: {path}")
print("Use this exact path in connect_memory_with_llm.py -> MODEL_PATH")