#!/usr/bin/env python3
"""Apply/check this checkout's Wan transformer from a private venv.

The original installed module is backed up once. No Diffusers import, CUDA
initialization, package installation, or global site-packages modification
is performed by this helper. Writes are limited to the venv; --check may
read an external editable source tree used by an existing private venv.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile


def private_prefix(expected: Path) -> Path:
    """Require the requested, isolated venv; reject a system interpreter."""
    expected = expected.expanduser().resolve()
    prefix = Path(sys.prefix).resolve()
    if prefix == Path(sys.base_prefix).resolve() or prefix != expected:
        raise RuntimeError("Run this helper with the Python inside the specified private --venv")
    config = prefix / "pyvenv.cfg"
    if not config.is_file():
        raise RuntimeError("The target is not a standard private Python virtual environment")
    options = {}
    for line in config.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            options[name.strip().lower()] = value.strip().lower()
    if options.get("include-system-site-packages") != "false":
        raise RuntimeError("Refusing a venv with access to system site-packages")
    return prefix


def patch_module(repo_root: Path, venv: Path, check_only: bool = False) -> dict:
    prefix = private_prefix(venv)
    source = repo_root.expanduser().resolve() / "module" / "transformer_wan.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing checkout transformer: {source}")
    distribution = importlib.metadata.distribution("diffusers")
    if distribution.version != "0.33.1":
        raise RuntimeError(f"Expected diffusers 0.33.1, found {distribution.version}")
    # Top-level find_spec resolves editable installations without importing the
    # package (or initializing Torch/CUDA). Metadata locate_file alone may point
    # to site-packages rather than an editable package's actual source tree.
    package_spec = importlib.util.find_spec("diffusers")
    if package_spec is None or package_spec.origin is None:
        raise RuntimeError("Cannot locate the installed Diffusers package")
    target = Path(package_spec.origin).parent / "models/transformers/transformer_wan.py"
    if target.is_symlink() or not target.is_file():
        raise RuntimeError("The installed transformer must be a regular, non-symlink file")
    target = target.resolve()
    inside_venv = target.is_relative_to(prefix)
    if not inside_venv and not check_only:
        raise RuntimeError("Refusing to patch global or external editable Diffusers")
    source_bytes = source.read_bytes()
    original_bytes = target.read_bytes()
    expected_hash = hashlib.sha256(source_bytes).hexdigest()
    actual_hash = hashlib.sha256(original_bytes).hexdigest()
    backup = target.with_name(target.name + ".ditraj-original")
    status = "verified" if check_only else "already_patched"
    if actual_hash != expected_hash:
        if check_only:
            raise RuntimeError(
                "Installed Diffusers differs from this checkout; apply the patch "
                "inside a private venv (external editable trees are read-only here)"
            )
        if backup.is_symlink():
            raise RuntimeError("Refusing a symlink at the transformer backup path")
        if not backup.exists():
            with backup.open("xb") as stream:
                stream.write(original_bytes)
        mode = stat.S_IMODE(target.stat().st_mode)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".ditraj-wan-", suffix=".tmp",
                dir=target.parent, delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(source_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(mode)
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        status = "patched"
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError("Post-patch SHA-256 verification failed")
    return {"status": status, "diffusers": distribution.version,
            "source": str(source), "target": str(target),
            "target_scope": "private_venv" if inside_venv else "external_read_only",
            "sha256": expected_hash, "original_backup": str(backup)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--venv", required=True, type=Path)
    parser.add_argument("--check", "--check-only", dest="check_only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = patch_module(args.repo_root, args.venv, args.check_only)
    except (OSError, RuntimeError, importlib.metadata.PackageNotFoundError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
