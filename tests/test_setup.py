"""Installer failure boundaries; no real installs, models, or network requests."""
import json
from io import BytesIO
from hashlib import sha256
from pathlib import Path
import subprocess
from types import SimpleNamespace

import huggingface_hub
from huggingface_hub.errors import LocalEntryNotFoundError
import pytest

import model_assets
import setup as installer


@pytest.fixture
def isolated_installer(monkeypatch, tmp_path):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    monkeypatch.setattr(installer.sys, "version_info", (3, 14, 2))
    monkeypatch.setattr(installer.struct, "calcsize", lambda _: 8)
    return tmp_path


def make_environment(root):
    directory = root / ".venv"
    python = installer.environment_python(directory)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"test interpreter")
    (directory / "pyvenv.cfg").write_text("version = 3.14.2\n", encoding="utf-8")
    return python


def forbid(*args, **kwargs):
    pytest.fail("This operation must not run")


def test_existing_environment_is_reused_without_creation(monkeypatch, isolated_installer):
    python = make_environment(isolated_installer)
    monkeypatch.setattr(installer.venv, "EnvBuilder", forbid)
    monkeypatch.setattr(installer, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=json.dumps([[3, 14], 8, str(python.parent.parent)])))
    assert installer.prepare_environment(offline=True) == python
    assert python.read_bytes() == b"test interpreter"


def test_offline_missing_environment_has_no_side_effects(monkeypatch, isolated_installer):
    monkeypatch.setattr(installer.venv, "EnvBuilder", forbid)
    monkeypatch.setattr(installer, "run", forbid)
    with pytest.raises(RuntimeError, match="first installation needs internet"):
        installer.prepare_environment(offline=True)
    assert not (isolated_installer / ".venv").exists()


@pytest.mark.parametrize("version,bits", [((3, 13, 7), 8), ((3, 14, 2), 4)])
def test_unsupported_interpreter_does_not_touch_environment(monkeypatch, isolated_installer, version, bits):
    monkeypatch.setattr(installer.sys, "version_info", version)
    monkeypatch.setattr(installer.struct, "calcsize", lambda _: bits)
    monkeypatch.setattr(installer.venv, "EnvBuilder", forbid)
    monkeypatch.setattr(installer, "run", forbid)
    with pytest.raises(RuntimeError, match="64-bit Python 3.14"):
        installer.prepare_environment()
    assert not (isolated_installer / ".venv").exists()


def test_unrelated_venv_directory_is_preserved(monkeypatch, isolated_installer):
    directory = isolated_installer / ".venv"
    directory.mkdir()
    marker = directory / "keep.txt"
    marker.write_text("user file", encoding="utf-8")
    monkeypatch.setattr(installer.venv, "EnvBuilder", forbid)
    monkeypatch.setattr(installer, "run", forbid)
    with pytest.raises(RuntimeError, match="incomplete"):
        installer.prepare_environment()
    assert marker.read_text(encoding="utf-8") == "user file"


@pytest.mark.parametrize("version,bits,other_location", [([3, 13], 8, False), ([3, 14], 4, False), ([3, 14], 8, True)])
def test_mismatched_environment_is_not_reinstalled(monkeypatch, isolated_installer, version, bits, other_location):
    python = make_environment(isolated_installer)
    prefix = isolated_installer / ("old-clone" if other_location else ".venv")
    monkeypatch.setattr(installer.venv, "EnvBuilder", forbid)
    monkeypatch.setattr(installer, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=json.dumps([version, bits, str(prefix)])))
    with pytest.raises(RuntimeError, match="different Python version or location"):
        installer.prepare_environment()
    assert python.read_bytes() == b"test interpreter"


def test_offline_dependencies_never_enable_package_indexes(monkeypatch, isolated_installer):
    commands = []
    monkeypatch.setattr(installer, "run", lambda command, **kwargs: commands.append([str(item) for item in command]))
    installer.install_dependencies(Path("test-python"), offline=True)
    installs = [command for command in commands if "install" in command]
    assert installs
    assert all("--no-index" in command for command in installs)
    assert not any("https://" in part or "--index-url" in part for command in commands for part in command)
    assert any(command[-2:] == ["pip", "check"] for command in commands)


@pytest.mark.parametrize("already_installed", [True, False])
def test_cpu_wheel_is_downloaded_only_when_needed(monkeypatch, isolated_installer, already_installed):
    monkeypatch.setattr(installer.sys, "platform", "win32")
    commands = []

    def run(command, **kwargs):
        commands.append([str(item) for item in command])
        if "-c" in command and not already_installed:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(installer, "run", run)
    installer.install_dependencies(Path("test-python"))
    cpu_installs = [command for command in commands if "torch==2.11.0+cpu" in command]
    assert len(cpu_installs) == (0 if already_installed else 1)
    if cpu_installs:
        assert "https://download.pytorch.org/whl/cpu" in cpu_installs[0]
    assert any("requirements.lock.txt" in part for command in commands for part in command)


@pytest.mark.parametrize("phase", ["prepare_environment", "install_dependencies", "verify_runtime", "model"])
def test_setup_failure_stops_following_work_and_never_reports_success(monkeypatch, capsys, phase):
    calls = []
    phases = ("prepare_environment", "install_dependencies", "verify_runtime")
    for name in phases:
        def stage(*args, _name=name, **kwargs):
            calls.append(_name)
            if phase == _name:
                raise subprocess.CalledProcessError(7, ["failed-stage"])
            return Path("test-python")
        monkeypatch.setattr(installer, name, stage)

    def run(command, **kwargs):
        calls.append("model")
        if phase == "model":
            raise subprocess.CalledProcessError(8, command)
        pytest.fail("The app must not start after a failed setup stage")

    monkeypatch.setattr(installer, "run", run)
    assert installer.main(["--run"]) == 1
    assert calls == list((*phases, "model")[:(*phases, "model").index(phase) + 1])
    output = capsys.readouterr()
    assert "Setup finished" not in output.out
    assert "Setup stopped" in output.err


def test_skip_model_cannot_launch_an_unprepared_app(monkeypatch, capsys):
    monkeypatch.setattr(installer, "prepare_environment", forbid)
    with pytest.raises(SystemExit) as exc:
        installer.main(["--skip-model", "--run"])
    assert exc.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_skip_model_completes_dependency_checks_without_download(monkeypatch, capsys):
    checked = []
    monkeypatch.setattr(installer, "prepare_environment", lambda **kwargs: Path("test-python"))
    monkeypatch.setattr(installer, "install_dependencies", lambda *args, **kwargs: checked.append("dependencies"))
    monkeypatch.setattr(installer, "verify_runtime", lambda *args: checked.append("runtime"))
    monkeypatch.setattr(installer, "run", forbid)
    assert installer.main(["--skip-model"]) == 0
    assert checked == ["dependencies", "runtime"]
    assert "Model preparation skipped" in capsys.readouterr().out


@pytest.fixture
def model_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hub-home"))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    return snapshot


def fill_snapshot(snapshot):
    for name in model_assets.MODEL_FILES:
        (snapshot / name).write_bytes(b"test asset")


def test_complete_cached_model_does_not_make_online_request(monkeypatch, model_cache):
    fill_snapshot(model_cache)
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        assert kwargs.get("local_files_only") is True
        return str(model_cache)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert model_assets.ensure_model() == str(model_cache)
    assert len(calls) == 1
    assert calls[0][0] == model_assets.MODEL_ID
    assert calls[0][1]["revision"] == model_assets.MODEL_REVISION


@pytest.mark.parametrize("cache_state", ["missing", "partial", "empty-weight"])
@pytest.mark.parametrize("use_environment_flag", [False, True])
def test_offline_incomplete_model_never_attempts_network(monkeypatch, model_cache, cache_state, use_environment_flag):
    if cache_state == "empty-weight":
        fill_snapshot(model_cache)
        (model_cache / "model.safetensors").write_bytes(b"")
    elif cache_state == "partial":
        (model_cache / "config.json").write_bytes(b"{}")
    if use_environment_flag:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    calls = []

    def download(repo, **kwargs):
        calls.append(kwargs)
        assert kwargs.get("local_files_only") is True, "Offline mode attempted a download"
        if cache_state == "missing":
            raise LocalEntryNotFoundError("Not cached")
        return str(model_cache)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    with pytest.raises(RuntimeError, match="complete pinned model is not cached"):
        model_assets.ensure_model(offline=not use_environment_flag)
    assert len(calls) == 1


def test_partial_online_snapshot_is_repaired_at_same_pinned_revision(monkeypatch, model_cache):
    (model_cache / "config.json").write_bytes(b"{}")
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        if not kwargs.get("local_files_only"):
            fill_snapshot(model_cache)
        return str(model_cache)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert model_assets.ensure_model() == str(model_cache)
    assert len(calls) == 2
    assert all(repo == model_assets.MODEL_ID and kwargs["revision"] == model_assets.MODEL_REVISION for repo, kwargs in calls)
    assert calls[1][1]["allow_patterns"] == list(model_assets.MODEL_FILES)


def test_incomplete_download_is_not_reported_as_ready(monkeypatch, model_cache):
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: str(model_cache))
    with pytest.raises(RuntimeError, match="download is incomplete"):
        model_assets.ensure_model()


def test_empty_cached_file_is_forced_to_download_without_replacing_healthy_files(monkeypatch, model_cache):
    fill_snapshot(model_cache)
    (model_cache / "vocab.json").write_bytes(b"")
    repairs = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: str(model_cache))

    def repair(repo, **kwargs):
        repairs.append((repo, kwargs))
        (model_cache / kwargs["filename"]).write_bytes(b"repaired")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", repair)
    assert model_assets.ensure_model() == str(model_cache)
    assert repairs == [(model_assets.MODEL_ID, dict(filename="vocab.json", revision=model_assets.MODEL_REVISION,
        force_download=True, cache_dir=model_assets.hub_cache_directory()))]
    assert (model_cache / "model.safetensors").read_bytes() == b"test asset"


def test_configured_external_cache_is_respected(monkeypatch, tmp_path):
    custom_cache = str(tmp_path / "existing-cache")
    monkeypatch.setenv("HF_HOME", custom_cache)
    model_assets.configure_cache()
    assert model_assets.os.environ["HF_HOME"] == custom_cache


@pytest.fixture
def transcription_cache(monkeypatch, model_cache):
    contents = {"config.json":b'{"test":true}', "model.safetensors":b"healthy pinned weights"}
    monkeypatch.setattr(model_assets, "TRANSCRIPTION_MANIFEST", {
        name:(len(data),sha256(data).hexdigest()) for name,data in contents.items()})
    monkeypatch.setattr(model_assets, "TRANSCRIPTION_FILES", tuple(contents))
    for name,data in contents.items():
        (model_cache/name).write_bytes(data)
    return model_cache, contents


def test_transcription_cached_assets_verify_full_hash_without_network(monkeypatch, transcription_cache):
    snapshot, _ = transcription_cache
    def local(repo, **kwargs):
        assert repo == model_assets.TRANSCRIPTION_MODEL_ID
        assert kwargs == dict(revision=model_assets.TRANSCRIPTION_MODEL_REVISION, local_files_only=True,
                              cache_dir=model_assets.hub_cache_directory())
        return str(snapshot)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", local)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbid)
    assert model_assets.ensure_transcription_model(offline=True) == str(snapshot)


def test_same_size_corrupt_transcription_weight_is_rejected_offline(monkeypatch, transcription_cache):
    snapshot, contents = transcription_cache
    (snapshot/"model.safetensors").write_bytes(b"x"*len(contents["model.safetensors"]))
    def local(*args, **kwargs):
        assert kwargs.get("local_files_only") is True
        return str(snapshot)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", local)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbid)
    with pytest.raises(RuntimeError, match="failed integrity"):
        model_assets.ensure_transcription_model(offline=True)


def test_corrupt_asr_metadata_repairs_only_that_file_at_pinned_revision(monkeypatch, transcription_cache):
    snapshot, contents = transcription_cache
    (snapshot/"config.json").write_bytes(b"bad")
    repaired = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args,**kwargs:str(snapshot))
    def repair(repo, **kwargs):
        repaired.append((repo,kwargs))
        (snapshot/kwargs["filename"]).write_bytes(contents[kwargs["filename"]])
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", repair)
    assert model_assets.ensure_transcription_model() == str(snapshot)
    assert repaired == [(model_assets.TRANSCRIPTION_MODEL_ID, {
        "filename":"config.json", "revision":model_assets.TRANSCRIPTION_MODEL_REVISION,
        "force_download":True, "cache_dir":model_assets.hub_cache_directory()})]
    assert (snapshot/"model.safetensors").read_bytes() == contents["model.safetensors"]


def test_missing_asr_file_uses_normal_resumable_download(monkeypatch, transcription_cache):
    snapshot, contents = transcription_cache
    (snapshot/"model.safetensors").unlink()
    calls = []
    def snapshot_download(repo, **kwargs):
        calls.append(kwargs)
        if not kwargs.get("local_files_only"):
            (snapshot/"model.safetensors").write_bytes(contents["model.safetensors"])
        return str(snapshot)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbid)
    assert model_assets.ensure_transcription_model() == str(snapshot)
    assert len(calls) == 2 and calls[1]["allow_patterns"] == list(contents)


def test_setup_offline_prepares_speech_and_activity_models_in_same_checked_stage(monkeypatch):
    monkeypatch.setattr(installer, "prepare_environment", lambda **kwargs:Path("test-python"))
    monkeypatch.setattr(installer, "install_dependencies", lambda *args,**kwargs:None)
    monkeypatch.setattr(installer, "verify_runtime", lambda *args:None)
    calls = []
    monkeypatch.setattr(installer, "run", lambda command,**kwargs:calls.append(command))
    assert installer.main(["--offline"]) == 0
    assert len(calls) == 1
    assert "ensure_model(offline=True)" in calls[0][2]
    assert "ensure_transcription_model(offline=True)" in calls[0][2]
    assert "ensure_activity_model(offline=True)" in calls[0][2]


@pytest.fixture
def activity_cache(monkeypatch, tmp_path):
    contents=b"pinned activity model bytes"
    monkeypatch.setattr(model_assets,"BASE",tmp_path)
    monkeypatch.setattr(model_assets,"ACTIVITY_MODEL_SIZE",len(contents))
    monkeypatch.setattr(model_assets,"ACTIVITY_MODEL_SHA256",sha256(contents).hexdigest())
    monkeypatch.delenv("HF_HUB_OFFLINE",raising=False)
    path=tmp_path/".cache"/"models"/"silero-vad"/model_assets.ACTIVITY_MODEL_REVISION/"silero_vad.jit"
    return path,contents


def activity_response(contents, *, status=200, headers=None):
    response=BytesIO(contents)
    response.status=status
    response.headers=headers or {}
    return response


def test_activity_cached_model_is_verified_without_network(monkeypatch,activity_cache):
    path,contents=activity_cache
    path.parent.mkdir(parents=True)
    path.write_bytes(contents)
    monkeypatch.setattr(model_assets,"urlopen",forbid)
    assert model_assets.ensure_activity_model(offline=True)==path


@pytest.mark.parametrize("state",["missing","same_size_corrupt"])
@pytest.mark.parametrize("environment_offline",[False,True])
def test_activity_offline_rejects_missing_or_corrupt_without_download(monkeypatch,activity_cache,state,environment_offline):
    path,contents=activity_cache
    if state=="same_size_corrupt":
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x"*len(contents))
    if environment_offline:monkeypatch.setenv("HF_HUB_OFFLINE","1")
    monkeypatch.setattr(model_assets,"urlopen",forbid)
    with pytest.raises(RuntimeError,match="failed integrity"):
        model_assets.ensure_activity_model(offline=not environment_offline)
    if state=="missing":assert not path.parent.exists()


def test_activity_first_download_is_pinned_verified_and_atomically_published(monkeypatch,activity_cache):
    path,contents=activity_cache
    def download(request,**kwargs):
        assert model_assets.ACTIVITY_MODEL_REVISION in request.full_url
        assert request.get_header("Range") is None
        assert request.get_header("Accept-encoding")=="identity"
        return activity_response(contents,headers={"Content-Length":str(len(contents))})
    monkeypatch.setattr(model_assets,"urlopen",download)
    assert model_assets.ensure_activity_model()==path
    assert path.read_bytes()==contents and not path.with_suffix(".jit.incomplete").exists()


def test_activity_valid_range_resumes_partial_bytes(monkeypatch,activity_cache):
    path,contents=activity_cache
    path.parent.mkdir(parents=True)
    prefix=7
    partial=path.with_suffix(".jit.incomplete");partial.write_bytes(contents[:prefix])
    def download(request,**kwargs):
        assert request.get_header("Range")==f"bytes={prefix}-"
        return activity_response(contents[prefix:],status=206,headers={
            "Content-Range":f"bytes {prefix}-{len(contents)-1}/{len(contents)}",
            "Content-Length":str(len(contents)-prefix)})
    monkeypatch.setattr(model_assets,"urlopen",download)
    assert model_assets.ensure_activity_model().read_bytes()==contents


def test_activity_ignored_range_restarts_without_duplicating_prefix(monkeypatch,activity_cache):
    path,contents=activity_cache
    path.parent.mkdir(parents=True)
    path.with_suffix(".jit.incomplete").write_bytes(contents[:7])
    monkeypatch.setattr(model_assets,"urlopen",lambda *a,**k:activity_response(contents))
    assert model_assets.ensure_activity_model().read_bytes()==contents


@pytest.mark.parametrize("status,headers",[
    (206,{"Content-Range":"bytes 0-3/4"}),
    (206,{}),
    (200,{"Content-Range":"bytes 7-25/26"}),
    (200,{"Content-Length":"3"}),
    (200,{"Content-Encoding":"gzip"}),
])
def test_activity_invalid_response_never_appends_to_partial(monkeypatch,activity_cache,status,headers):
    path,contents=activity_cache
    path.parent.mkdir(parents=True)
    partial=path.with_suffix(".jit.incomplete");partial.write_bytes(contents[:7])
    monkeypatch.setattr(model_assets,"urlopen",lambda *a,**k:activity_response(contents,status=status,headers=headers))
    with pytest.raises(RuntimeError,match="download"):
        model_assets.ensure_activity_model()
    assert partial.read_bytes()==contents[:7] and not path.exists()


def test_activity_truncated_download_preserves_bytes_for_resume(monkeypatch,activity_cache):
    path,contents=activity_cache
    monkeypatch.setattr(model_assets,"urlopen",lambda *a,**k:activity_response(contents[:7]))
    with pytest.raises(RuntimeError,match="incomplete"):
        model_assets.ensure_activity_model()
    assert path.with_suffix(".jit.incomplete").read_bytes()==contents[:7]
    assert not path.exists()


def test_activity_bad_digest_never_replaces_existing_cache(monkeypatch,activity_cache):
    path,contents=activity_cache
    path.parent.mkdir(parents=True)
    path.write_bytes(b"old bad file")
    monkeypatch.setattr(model_assets,"urlopen",lambda *a,**k:activity_response(b"x"*len(contents)))
    with pytest.raises(RuntimeError,match="integrity verification failed"):
        model_assets.ensure_activity_model()
    assert path.read_bytes()==b"old bad file"
    assert not path.with_suffix(".jit.incomplete").exists()
