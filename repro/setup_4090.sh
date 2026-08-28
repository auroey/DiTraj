#!/usr/bin/env bash
# Linux x86_64 setup only. This script never installs a system driver/toolkit.
set -euo pipefail
umask 077
script_path=${BASH_SOURCE[0]}
script_dir=${script_path%/*}
if [ "$script_dir" = "$script_path" ]; then script_dir=.; fi
repo_root=$(cd -- "$script_dir/.." && pwd -P)
workdir=${DITRAJ_WORKDIR:-"$repo_root/.repro"}
bootstrap_python=${DITRAJ_BOOTSTRAP_PYTHON:-python3.11}

usage() {
    printf '%s\n' \
        'Usage: bash repro/setup_4090.sh [--workdir DIR] [--python PYTHON3.11]' \
        'Creates/reuses DIR/.venv; uses the current checkout, not a second clone.' \
        'Defaults: DITRAJ_WORKDIR or <checkout>/.repro; Python: python3.11.' \
        'Install Python 3.11 separately if needed; no sudo, driver or global pip changes.'
}
while (( $# )); do
    case "$1" in
        --workdir|--python)
            (( $# >= 2 )) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
            if [ "$1" = --workdir ]; then workdir=$2; else bootstrap_python=$2; fi
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
[ -n "$workdir" ] || { printf 'Workdir cannot be empty.\n' >&2; exit 2; }
command -v -- "$bootstrap_python" >/dev/null || {
    printf 'Python 3.11 not found; pass --python /path/to/python3.11.\n' >&2; exit 2;
}
"$bootstrap_python" -I -c '
import platform, sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit("This environment requires Python 3.11")
if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
    raise SystemExit("This installer targets Linux x86_64")
libc, version = platform.libc_ver()
if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 28):
    raise SystemExit("The pinned PyTorch wheels require glibc >= 2.28")
'
workdir=$("$bootstrap_python" -I -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$workdir")
[ "$workdir" != / ] || { printf 'Refusing filesystem root as workdir.\n' >&2; exit 2; }
venv_dir="$workdir/.venv"
mkdir -p "$workdir" "$workdir/logs" "$workdir/outputs" "$workdir/models"
if [ -e "$venv_dir" ] || [ -L "$venv_dir" ]; then
    [ ! -L "$venv_dir" ] && [ -f "$venv_dir/pyvenv.cfg" ] && [ -x "$venv_dir/bin/python" ] || {
        printf 'Existing .venv is not a regular usable venv; choose a new workdir.\n' >&2; exit 2;
    }
else
    "$bootstrap_python" -I -m venv "$venv_dir"
fi
python_exe="$venv_dir/bin/python"
# Validate isolation before any package changes. Do not rely on activated shells.
"$python_exe" -I - "$venv_dir" <<'PY'
from pathlib import Path
import sys
expected = Path(sys.argv[1]).resolve()
if Path(sys.prefix).resolve() != expected or sys.prefix == sys.base_prefix:
    raise SystemExit("Refusing package installation outside the requested private venv")
if sys.version_info[:2] != (3, 11):
    raise SystemExit("The existing private venv must use Python 3.11")
config = (expected / "pyvenv.cfg").read_text().lower()
options = dict(line.split("=", 1) for line in config.splitlines() if "=" in line)
options = {key.strip(): value.strip() for key, value in options.items()}
if options.get("include-system-site-packages") != "false":
    raise SystemExit("Refusing a venv with system site-packages enabled")
PY
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null
unset PIP_TARGET PIP_PREFIX PIP_USER PYTHONPATH PYTHONHOME
"$python_exe" -I -m pip --require-virtualenv install \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.7.1+cu118 torchvision==0.22.1+cu118
"$python_exe" -I -m pip --require-virtualenv install \
    --index-url https://pypi.org/simple -r "$repo_root/requirements-4090.txt"
"$python_exe" -I "$repo_root/repro/apply_wan_patch.py" \
    --repo-root "$repo_root" --venv "$venv_dir"
"$python_exe" -I -m pip --require-virtualenv check
"$python_exe" -I "$repo_root/repro/apply_wan_patch.py" \
    --repo-root "$repo_root" --venv "$venv_dir" --check
printf 'Private runtime ready: %s\n' "$python_exe"
printf 'Weights are not downloaded by setup. Model directory: %s\n' "$workdir/models"
