from huggingface_hub import snapshot_download

import config

# Downloads tokenizer files + weights into weights/gpt2/.
path = snapshot_download(
    repo_id=config.HF_REPO_ID,
    local_dir=config.WEIGHTS_DIR,
    allow_patterns=config.DOWNLOAD_PATTERNS,
)
print("Saved to:", path)