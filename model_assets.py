"""Pinned model assets shared by the installer and application.

Only the standard library is imported until ensure_model() is called.
"""
import os
import re
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen

BASE = Path(__file__).resolve().parent
MODEL_ID = "obadx/muaalem-model-v3_2"
MODEL_REVISION = "01a1ef9fbe40d144ef845101e89ff924aed3fef5"
MODEL_FILES = (
    "config.json", "model.safetensors", "preprocessor_config.json",
    "added_tokens.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.json",
)
TRANSCRIPTION_MODEL_ID = "openai/whisper-large-v3-turbo"
TRANSCRIPTION_MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
# Sizes and SHA-256 digests of the immutable upstream revision. The weight
# digest is the HF LFS SHA-256; metadata digests cover the exact file bytes.
TRANSCRIPTION_MANIFEST = {
    "added_tokens.json": (34648, "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0"),
    "config.json": (1256, "c5b526b3e3cd64cd8940dabb45e8ba726629e22d8ed389c29b552f9140daf04a"),
    "generation_config.json": (3772, "cce11bfe3aaa6ae9e072ea2637caaec8795e68d9b67e655a5af16ee509681a4c"),
    "merges.txt": (493869, "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6"),
    "model.safetensors": (1617824864, "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1"),
    "normalizer.json": (52666, "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd"),
    "preprocessor_config.json": (340, "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"),
    "special_tokens_map.json": (2186, "baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74"),
    "tokenizer.json": (2710337, "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd"),
    "tokenizer_config.json": (282843, "844b642c73a91359722f47b35705f7174686df33d252695d8572cf9ac03a6389"),
    "vocab.json": (1036558, "e2aa043ef015641d363d8288e7c241c85e36a5c761fb303598e0710233344387"),
}
TRANSCRIPTION_FILES = tuple(TRANSCRIPTION_MANIFEST)
ACTIVITY_MODEL_ID = "snakers4/silero-vad"
ACTIVITY_MODEL_REVISION = "5cd7945676eb32225748052e2e6a0580e4686a08"  # v6.2.3
ACTIVITY_MODEL_SIZE = 2272526
ACTIVITY_MODEL_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
ACTIVITY_MODEL_URL = (
    "https://raw.githubusercontent.com/" + ACTIVITY_MODEL_ID + "/" + ACTIVITY_MODEL_REVISION
    + "/src/silero_vad/data/silero_vad.jit"
)


def configure_cache():
    # Set these before importing huggingface_hub, which reads them at import time.
    os.environ.setdefault("HF_HOME", str(BASE / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def complete_snapshot(path):
    return all((Path(path) / name).is_file() and (Path(path) / name).stat().st_size > 0
               for name in MODEL_FILES)


def hub_cache_directory():
    # Passing this explicitly also works if another library imported HF before
    # configure_cache() could set HF_HOME (HF constants are import-time values).
    configure_cache()
    return os.environ.get("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))


def ensure_model(*, offline=False):
    configure_cache()
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    offline = offline or os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "TRUE", "YES", "ON"}
    path = None
    try:
        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
                                 cache_dir=hub_cache_directory())
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
                    hf_hub_download(MODEL_ID, filename=name, revision=MODEL_REVISION, force_download=True,
                                    cache_dir=hub_cache_directory())
        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=list(MODEL_FILES),
                                 cache_dir=hub_cache_directory())
    if not complete_snapshot(path):
        raise RuntimeError("Model download is incomplete. Run python setup.py again.")
    return str(path)


def invalid_transcription_files(path):
    """Verify every required file, including the full weight SHA-256."""
    invalid = []
    for name, (size, expected_digest) in TRANSCRIPTION_MANIFEST.items():
        item = Path(path) / name
        try:
            if not item.is_file() or item.stat().st_size != size:
                invalid.append(name)
                continue
            digest = sha256()
            with item.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_digest:
                invalid.append(name)
        except OSError:
            invalid.append(name)
    return invalid


def ensure_transcription_model(*, offline=False):
    """Resume HF downloads and never load incomplete or mismatched ASR assets."""
    configure_cache()
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    offline = offline or os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "TRUE", "YES", "ON"}
    path = None
    try:
        path = snapshot_download(TRANSCRIPTION_MODEL_ID, revision=TRANSCRIPTION_MODEL_REVISION,
                                 local_files_only=True, cache_dir=hub_cache_directory())
    except LocalEntryNotFoundError:
        pass
    invalid = invalid_transcription_files(path) if path else list(TRANSCRIPTION_FILES)
    if not invalid:
        return str(path)
    if offline:
        raise RuntimeError("The complete pinned transcription model is not cached or failed integrity verification. Run python setup.py while online.")
    # Only corrupt existing entries need force_download. Missing files use the
    # Hub's resumable .incomplete files and shared per-blob locks normally.
    if path:
        for name in invalid:
            if (Path(path) / name).is_file():
                hf_hub_download(TRANSCRIPTION_MODEL_ID, filename=name,
                                revision=TRANSCRIPTION_MODEL_REVISION, force_download=True,
                                cache_dir=hub_cache_directory())
    path = snapshot_download(TRANSCRIPTION_MODEL_ID, revision=TRANSCRIPTION_MODEL_REVISION,
                             allow_patterns=list(TRANSCRIPTION_FILES), cache_dir=hub_cache_directory())
    invalid = invalid_transcription_files(path)
    if invalid:
        raise RuntimeError("Transcription model integrity verification failed: " + ", ".join(invalid))
    return str(path)


def _valid_activity_model(path):
    try:
        if not path.is_file() or path.stat().st_size != ACTIVITY_MODEL_SIZE:
            return False
        digest = sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == ACTIVITY_MODEL_SHA256
    except OSError:
        return False


def ensure_activity_model(*, offline=False):
    """Return a verified local Silero JIT file; never execute remote hub code.

    Partial transfers survive an interruption. A server that ignores Range is
    restarted atomically, while a malformed partial response is never appended.
    """
    path = BASE / ".cache" / "models" / "silero-vad" / ACTIVITY_MODEL_REVISION / "silero_vad.jit"
    if _valid_activity_model(path):
        return path
    offline = offline or os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "TRUE", "YES", "ON"}
    if offline:
        raise RuntimeError("The pinned speech activity model is not cached or failed integrity verification. Run python setup.py while online.")
    from filelock import FileLock

    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if _valid_activity_model(path):
            return path
        partial = path.with_suffix(path.suffix + ".incomplete")
        if _valid_activity_model(partial):
            partial.replace(path)
            return path
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset >= ACTIVITY_MODEL_SIZE:
            partial.unlink()
            offset = 0
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with urlopen(Request(ACTIVITY_MODEL_URL, headers=headers), timeout=30) as response:
            status = response.status
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise RuntimeError("Speech activity model download used an unexpected content encoding.")
            if status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match or tuple(map(int, match.groups())) != (offset, ACTIVITY_MODEL_SIZE-1, ACTIVITY_MODEL_SIZE):
                    raise RuntimeError("Speech activity model download returned an invalid Content-Range.")
            elif status == 200:
                # GitHub/CDN may ignore Range. Do not append a second full file.
                if response.headers.get("Content-Range"):
                    raise RuntimeError("Speech activity model download returned an unexpected Content-Range.")
                offset = 0
            else:
                raise RuntimeError(f"Speech activity model download returned HTTP {status}.")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdecimal() or int(length) != ACTIVITY_MODEL_SIZE-offset):
                raise RuntimeError("Speech activity model download returned an invalid Content-Length.")
            count = offset
            with partial.open("ab" if offset else "wb") as target:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    count += len(chunk)
                    if count > ACTIVITY_MODEL_SIZE:
                        raise RuntimeError("Speech activity model download exceeded its pinned size.")
                    target.write(chunk)
        if count != ACTIVITY_MODEL_SIZE:
            raise RuntimeError("Speech activity model download is incomplete. Rerun setup.py to resume.")
        if not _valid_activity_model(partial):
            partial.unlink()
            raise RuntimeError("Speech activity model integrity verification failed. Rerun setup.py.")
        partial.replace(path)
        return path
