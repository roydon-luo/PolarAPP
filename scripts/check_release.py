"""Fail when a release tree contains common private or generated artifacts."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


FORBIDDEN_EXTENSIONS = {".ckpt", ".onnx", ".part", ".pt", ".pth", ".zip"}
GENERATED_DIRECTORIES = {
    "datasets",
    "checkpoints",
    "experiments",
    "outputs",
    "runs",
    "wandb",
}
CACHE_DIRECTORIES = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
TEXT_EXTENSIONS = {
    ".cff",
    ".cfg",
    ".gitignore",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PRIVATE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/](?:AllUsers|Users)[\\/]"),
    re.compile(r"/(?:home|Users)/[^/\s]+/"),
)


def _is_placeholder(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    generated_parts = {
        part.lower() for part in relative.parts[:-1] if part.lower() in GENERATED_DIRECTORIES
    }
    return bool(generated_parts) and path.name.lower() in {"readme.md", ".gitkeep"}


def audit(root: Path) -> list[str]:
    root = root.resolve()
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if {part.lower() for part in relative.parts} & CACHE_DIRECTORIES:
            continue
        if path.is_dir():
            continue
        lower_parts = {part.lower() for part in relative.parts[:-1]}
        if lower_parts & GENERATED_DIRECTORIES and not _is_placeholder(path, root):
            errors.append(f"generated/private directory content: {relative}")
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
            errors.append(f"forbidden release artifact: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"file larger than 10 MiB: {relative}")
        if path.suffix.lower() not in TEXT_EXTENSIONS and path.name != ".gitignore":
            continue
        try:
            contents = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"text file is not UTF-8: {relative}")
            continue
        for pattern in PRIVATE_PATH_PATTERNS:
            if pattern.search(contents):
                errors.append(f"machine-specific absolute path: {relative}")
                break
        cuda_visibility_token = "CUDA_" + "VISIBLE_DEVICES"
        if cuda_visibility_token in contents:
            errors.append(f"hard-coded CUDA visibility: {relative}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    errors = audit(args.root)
    if errors:
        print("Release audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Release audit passed: {args.root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
