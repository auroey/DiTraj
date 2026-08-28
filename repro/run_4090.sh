#!/usr/bin/env bash
set -euo pipefail
umask 077
script_path=${BASH_SOURCE[0]}
script_dir=${script_path%/*}
if [ "$script_dir" = "$script_path" ]; then script_dir=.; fi
repo_root=$(cd -- "$script_dir/.." && pwd -P)
workdir=${DITRAJ_WORKDIR:-"$repo_root/.repro"}
model=${DITRAJ_MODEL:-}
python_exe=${DITRAJ_PYTHON:-}
gpu_index=${DITRAJ_GPU:-0}
mode=full
min_free_mib=${DITRAJ_MIN_FREE_MIB:-23000}
output=
dry_run=0
runner_args=()

usage() {
    printf '%s\n' \
        'Usage: bash repro/run_4090.sh [launcher options] -- [runner options]' \
        '  --workdir DIR   Local runtime/cache/output root (default: <checkout>/.repro)' \
        '  --model PATH    Local weights (default: WORKDIR/models/Wan2.1-T2V-1.3B-Diffusers)' \
        '  --python PATH  Runtime Python (default: WORKDIR/.venv/bin/python)' \
        '  --gpu INDEX    Physical nvidia-smi index (default: 0; use --gpu 5 if desired)' \
        '  --mode MODE    full or smoke (default: full)' \
        '  --output FILE  New .mp4 path; never overwrite an existing output/log' \
        '  --min-free-mib N  Conservative free-VRAM gate (default: 23000)' \
        '  --dry-run      Print runner metadata without GPU queries or output creation' \
        'Runner options after -- are forwarded unchanged, e.g. --seed 123 --steps 10.' \
        'Use launcher --model/--output; --repo-root is always the current checkout.' \
        'Inference is offline. The local flock is not a cluster reservation.'
}
while (( $# )); do
    case "$1" in
        --workdir|--model|--python|--gpu|--mode|--output|--min-free-mib)
            (( $# >= 2 )) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
            case "$1" in
                --workdir) workdir=$2 ;; --model) model=$2 ;; --python) python_exe=$2 ;;
                --gpu) gpu_index=$2 ;; --mode) mode=$2 ;; --output) output=$2 ;;
                --min-free-mib) min_free_mib=$2 ;;
            esac
            shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        --) shift; runner_args=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown launcher option %s; put runner options after --.\n' "$1" >&2; exit 2 ;;
    esac
done
[[ "$gpu_index" =~ ^[0-9]+$ ]] || { printf 'GPU must be a nonnegative physical index.\n' >&2; exit 2; }
[[ "$min_free_mib" =~ ^[0-9]+$ ]] || { printf 'min-free-mib must be nonnegative.\n' >&2; exit 2; }
[[ "$mode" = full || "$mode" = smoke ]] || { printf 'Mode must be full or smoke.\n' >&2; exit 2; }
[ -n "$workdir" ] || { printf 'Workdir cannot be empty.\n' >&2; exit 2; }
for argument in "${runner_args[@]}"; do
    case "$argument" in
        --repo-root|--repo-root=*|--output|--output=*|--model|--model=*)
            printf 'Use launcher options for model/output; repo-root is fixed to this checkout.\n' >&2
            exit 2 ;;
        --dry-run) dry_run=1 ;;
    esac
done
if [[ "$workdir" != /* ]]; then workdir="$PWD/$workdir"; fi
if [ -z "$python_exe" ]; then python_exe="$workdir/.venv/bin/python"; fi
command -v -- "$python_exe" >/dev/null || {
    printf 'Runtime Python not found; run setup_4090.sh or pass --python.\n' >&2; exit 2;
}
workdir=$("$python_exe" -I -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$workdir")
[ "$workdir" != / ] || { printf 'Refusing filesystem root as workdir.\n' >&2; exit 2; }
if [ -z "$model" ]; then model="$workdir/models/Wan2.1-T2V-1.3B-Diffusers"; fi
if [ -d "$model" ]; then model=$(cd -- "$model" && pwd -P); fi
if [ -z "$output" ]; then
    run_dir="$workdir/outputs/$mode-gpu$gpu_index-$(date +%Y%m%d-%H%M%S)-$$"
    output="$run_dir/output.mp4"
else
    output=$("$python_exe" -I -c \
        'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$output")
    run_dir=${output%/*}
fi
run_args=()
if [ "$mode" = smoke ]; then
    run_args+=(--num-frames 9 --height 160 --width 288 --steps 2 --mask-step 1 --fix-rope-step 1)
fi
run_args+=("${runner_args[@]}" --repo-root "$repo_root" --model "$model" \
           --local-files-only --output "$output")
if [ "$dry_run" = 1 ]; then
    "$python_exe" "$repo_root/repro/run_low_vram.py" "${run_args[@]}" --dry-run
    exit 0
fi
command -v flock >/dev/null || { printf 'flock is required for the local project lock.\n' >&2; exit 2; }
venv_dir=$("$python_exe" -I -c 'import sys; print(sys.prefix)')
"$python_exe" -I "$repo_root/repro/apply_wan_patch.py" \
    --repo-root "$repo_root" --venv "$venv_dir" --check
mkdir -p "$workdir/locks"
# Coordinates this workdir's launchers only; other users/jobs can still claim the GPU.
exec 9>"$workdir/locks/gpu-$gpu_index.lock"
flock -n 9 || { printf 'Another launcher in this workdir is using GPU %s.\n' "$gpu_index" >&2; exit 3; }
gpu_report=$("$python_exe" -I "$repo_root/repro/gpu_preflight.py" \
    --gpu-index "$gpu_index" --min-free-mib "$min_free_mib")
printf '%s\n' "$gpu_report"
gpu_uuid=$("$python_exe" -I -c 'import json,sys; print(json.loads(sys.argv[1])["uuid"])' "$gpu_report")
log_path="$run_dir/run.log"
"$python_exe" -I - "$output" "$log_path" <<'PY'
from pathlib import Path
import sys
output, log = map(Path, sys.argv[1:])
if output.suffix.lower() != ".mp4":
    raise SystemExit("--output must end in .mp4")
for path in (output, output.with_name(output.stem + "_box.mp4"),
             output.with_suffix(".metrics.json"), log):
    if path.exists() or path.is_symlink():
        raise SystemExit(f"Refusing to overwrite: {path}")
PY
mkdir -p "$run_dir"
# Atomically claim this log; append thereafter rather than truncating existing files.
(set -o noclobber; : > "$log_path") || { printf 'Run log already exists.\n' >&2; exit 3; }
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export HF_HOME="$workdir/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${DITRAJ_CPU_THREADS:-8}
export MKL_NUM_THREADS=${DITRAJ_CPU_THREADS:-8}
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
printf 'Run directory: %s\n' "$run_dir"
"$python_exe" -u "$repo_root/repro/run_low_vram.py" "${run_args[@]}" 2>&1 | tee -a "$log_path"
