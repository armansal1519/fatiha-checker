"""Pinned model assets shared by the installer and application.

Only the standard library is imported until ensure_model() is called.
"""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL_ID = "obadx/muaalem-model-v3_2"
MODEL_REVISION = "01a1ef9fbe40d144ef845101e89ff924aed3fef5"
MODEL_FILES = (
    "config.json", "model.safetensors", "preprocessor_config.json",
    "added_tokens.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.json",
)


def configure_cache():
    # Set these before importing huggingface_hub, which reads them at import time.
    os.environ.setdefault("HF_HOME", str(BASE / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def complete_snapshot(path):
    return all((Path(path) / name).is_file() and (Path(path) / name).stat().st_size > 0
               for name in MODEL_FILES)


def ensure_model(*, offline=False):
    configure_cache()
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    offline = offline or os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "TRUE", "YES", "ON"}
    path = None
    try:
        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        if not complete_snapshot(path):
            raise LocalEntryNotFoundError("Incomplete cached snapshot")
    except LocalEntryNotFoundError:
        if offline:
            raise RuntimeError("The complete pinned model is not cached. Run python setup.py while online.") from None
        if path is not None:
            # Hub reuses existing files, even empty ones. Repair only those
            # entries; do not redownload a healthy 2.4 GB weight file.
            for name in MODEL_FILES:
                item = Path(path) / name
                if item.is_file() and item.stat().st_size == 0:
                    hf_hub_download(MODEL_ID, filename=name, revision=MODEL_REVISION, force_download=True)
        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=list(MODEL_FILES))
    if not complete_snapshot(path):
        raise RuntimeError("Model download is incomplete. Run python setup.py again.")
    return str(path)
