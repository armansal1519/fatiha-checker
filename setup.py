"""Bootstrap Hamkhan. Run with Python 3.14: python setup.py [--run].

This is an application installer, not a setuptools package definition.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
PYTHON_VERSION = (3, 14)


def run(command, *, capture=False):
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    return subprocess.run([str(part) for part in command], cwd=ROOT, env=env,
                          check=True, text=True, encoding="utf-8", capture_output=capture)


def environment_python(directory):
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def prepare_environment(*, offline=False):
    if sys.version_info[:2] != PYTHON_VERSION or struct.calcsize("P") != 8:
        raise RuntimeError("Install 64-bit Python 3.14, then run this script again. The lock file targets Python 3.14.")
    directory = ROOT / ".venv"
    python = environment_python(directory)
    if not directory.exists():
        if offline:
            raise RuntimeError("No .venv exists. The first installation needs internet: python setup.py")
        print("[1/4] Creating .venv ...", flush=True)
        venv.EnvBuilder(with_pip=True).create(directory)
    elif not (directory / "pyvenv.cfg").is_file() or not python.is_file():
        raise RuntimeError(".venv is incomplete or is not a Python environment. Move it aside, then run setup again.")
    else:
        print("[1/4] Reusing .venv ...", flush=True)
    info = json.loads(run([python, "-c", "import json,struct,sys; print(json.dumps([list(sys.version_info[:2]), struct.calcsize('P'), sys.prefix]))"], capture=True).stdout)
    if info[:2] != [list(PYTHON_VERSION), 8] or Path(info[2]).resolve() != directory.resolve():
        raise RuntimeError(".venv uses a different Python version or location. Move it aside and rerun with 64-bit Python 3.14.")
    return python


def install_dependencies(python, *, offline=False):
    print("[2/4] Checking locked dependencies ..." if offline else "[2/4] Installing locked dependencies (CPU) ...", flush=True)
    pip = [python, "-m", "pip", "install"]
    if offline:
        # Never contact an index. A missing or mismatched package fails clearly.
        run([*pip, "--no-index", "-r", ROOT / "requirements.lock.txt"])
    else:
        try:
            run([python, "-c", "import torch; assert torch.__version__.split('+')[0] == '2.11.0' and torch.version.cuda is None"], capture=True)
            cpu_installed = True
        except subprocess.CalledProcessError:
            cpu_installed = False
        if sys.platform != "darwin" and not cpu_installed:
            # Install the CPU wheel explicitly; PyPI defaults differ by platform.
            run([*pip, "--no-deps", "torch==2.11.0+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"])
        run([*pip, "--index-url", "https://pypi.org/simple", "-r", ROOT / "requirements.lock.txt"])
    run([python, "-m", "pip", "check"])


def verify_runtime(python):
    print("[3/4] Checking Python imports and the complete local Hafs corpus ...", flush=True)
    run([python, "-c", "import av, flask, flask_sock, numpy, simple_websocket, torch; "
         "from quran_muaalem import Muaalem; from transformers import WhisperForConditionalGeneration, WhisperProcessor; "
         "from recitation import RecitationService; from quran_corpus import QuranCorpus; "
         "assert torch.version.cuda is None, 'A CPU PyTorch installation is required'; "
         "assert len(RecitationService().references) == 7; assert QuranCorpus().validate() == 6236; "
         "print('Runtime imports, 114 surahs and 6236 ayahs OK.')"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Set up Hamkhan, both pinned speech models and the offline speech activity detector.")
    parser.add_argument("--offline", action="store_true", help="Verify an existing installation/cache without network access.")
    parser.add_argument("--skip-model", action="store_true", help="Install dependencies only; defer the speech models and activity detector (about 4 GB total).")
    parser.add_argument("--run", action="store_true", help="Start the app after successful setup.")
    args = parser.parse_args(argv)
    if args.skip_model and args.run:
        parser.error("--skip-model cannot be combined with --run")
    try:
        python = prepare_environment(offline=args.offline)
        install_dependencies(python, offline=args.offline)
        verify_runtime(python)
        if args.skip_model:
            print("[4/4] Model preparation skipped. Run python setup.py before using the app.")
        else:
            print("[4/4] Preparing pinned speech models and activity detector (about 4 GB on the first download) ...", flush=True)
            run([python, "-c", "from model_assets import ensure_model, ensure_transcription_model, ensure_activity_model; "
                 f"print('Phonetic model cache ready: ' + ensure_model(offline={args.offline!r})); "
                 f"print('Transcription model cache ready: ' + ensure_transcription_model(offline={args.offline!r})); "
                 f"print('Speech activity model cache ready: ' + str(ensure_activity_model(offline={args.offline!r})))"])
        command = "& '" + str(python).replace("'", "''") + "' app.py" if os.name == "nt" else shlex.quote(str(python)) + " app.py"
        print(f'\nSetup finished. Start the app with:\n  {command}\nOpen http://127.0.0.1:5000/live after the model loads.')
        if args.run:
            run([python, "app.py"])
        return 0
    except (RuntimeError, OSError) as exc:
        print(f"\nSetup failed: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"\nSetup stopped: a command failed (exit {exc.returncode}). See its error above, fix it, then rerun setup.py.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Rerun setup.py to reuse completed downloads.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
