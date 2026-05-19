from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="marcosv/ffhq-dataset",
    repo_type="dataset",
    local_dir="ffhq",
    allow_patterns=["Part1/*.png"],
)
