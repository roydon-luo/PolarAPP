"""Download the PolaRGB dataset with resumable HTTPS requests.

This is a fallback for environments where huggingface_hub cannot validate
metadata returned through a local HTTPS proxy.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests
import urllib3
from PIL import Image, UnidentifiedImageError


REPO_ID = "Mingde/PolaRGB"
API_URL = f"https://huggingface.co/api/datasets/{REPO_ID}"
RESOLVE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (needed by some local proxies).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.insecure:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    manifest_response = requests.get(
        API_URL, verify=not args.insecure, timeout=180
    )
    manifest_response.raise_for_status()
    files = [item["rfilename"] for item in manifest_response.json()["siblings"]]
    print(f"MANIFEST_FILES={len(files)}", flush=True)

    thread_local = threading.local()

    def get_session() -> requests.Session:
        if not hasattr(thread_local, "session"):
            thread_local.session = requests.Session()
            thread_local.session.verify = not args.insecure
        return thread_local.session

    def is_complete(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if path.suffix.lower() != ".png":
            return True
        try:
            with Image.open(path) as image:
                image.verify()
            return True
        except (OSError, UnidentifiedImageError):
            return False

    def download(relative_path: str) -> tuple[str, str, int | str]:
        destination = args.output / Path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if is_complete(destination):
            return "skipped", relative_path, destination.stat().st_size
        destination.unlink(missing_ok=True)

        partial = Path(f"{destination}.part")
        for attempt in range(10):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                url = RESOLVE_URL + quote(relative_path, safe="/") + "?download=true"
                with get_session().get(
                    url, headers=headers, stream=True, timeout=(45, 600)
                ) as response:
                    if offset and response.status_code == 200:
                        partial.unlink(missing_ok=True)
                        offset = 0
                    elif response.status_code == 416:
                        partial.replace(destination)
                        return "downloaded", relative_path, destination.stat().st_size
                    response.raise_for_status()
                    mode = "ab" if offset and response.status_code == 206 else "wb"
                    with partial.open(mode) as output_file:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                output_file.write(chunk)
                partial.replace(destination)
                if not is_complete(destination):
                    destination.unlink(missing_ok=True)
                    raise OSError("Downloaded file failed image-integrity validation")
                return "downloaded", relative_path, destination.stat().st_size
            except Exception as error:  # Retry transient proxy/network failures.
                if attempt == 9:
                    return "error", relative_path, repr(error)
                time.sleep(min(30, 2**attempt))
        raise AssertionError("unreachable")

    started = time.time()
    downloaded = skipped = errors = downloaded_bytes = 0
    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, path) for path in files]
        for index, future in enumerate(as_completed(futures), 1):
            status, path, value = future.result()
            if status == "downloaded":
                downloaded += 1
                downloaded_bytes += int(value)
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                failed.append((path, str(value)))
                print(f"ERROR {path}: {value}", flush=True)
            if index % 100 == 0 or index == len(files):
                rate = index / max(time.time() - started, 1)
                print(
                    f"PROGRESS {index}/{len(files)} downloaded={downloaded} "
                    f"skipped={skipped} errors={errors} "
                    f"new_GB={downloaded_bytes / 1e9:.3f} files_per_s={rate:.2f}",
                    flush=True,
                )

    print(
        f"DOWNLOAD_FINISHED files={len(files)} downloaded={downloaded} "
        f"skipped={skipped} errors={errors} new_bytes={downloaded_bytes}",
        flush=True,
    )
    if failed:
        print("FAILED_FILES_BEGIN", flush=True)
        for path, error in failed:
            print(f"{path}: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
