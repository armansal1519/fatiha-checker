"""Installer failure boundaries; no real installs, models, or network requests."""
import json
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
    assert repairs == [(model_assets.MODEL_ID, dict(filename="vocab.json", revision=model_assets.MODEL_REVISION, force_download=True))]
    assert (model_cache / "model.safetensors").read_bytes() == b"test asset"


def test_configured_external_cache_is_respected(monkeypatch, tmp_path):
    custom_cache = str(tmp_path / "existing-cache")
    monkeypatch.setenv("HF_HOME", custom_cache)
    model_assets.configure_cache()
    assert model_assets.os.environ["HF_HOME"] == custom_cache
