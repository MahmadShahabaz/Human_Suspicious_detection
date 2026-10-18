"""Provision and verify the original SHAS model files on persistent storage."""
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("SHAS_MODEL_DIR", str(ROOT))).expanduser().resolve()
MANIFEST = json.loads((ROOT / "model_manifest.json").read_text(encoding="utf-8"))
REQUIRED = tuple(MANIFEST)
MAX_ARCHIVE_BYTES = 400 * 1024 * 1024


def verify_model_files(directory):
    """Return names whose size or SHA-256 does not match the checked-in manifest."""
    invalid = []
    for name, expected in MANIFEST.items():
        path = Path(directory) / name
        if not path.is_file() or path.stat().st_size != expected["size"]:
            invalid.append(name)
            continue
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            invalid.append(name)
    return invalid


def download_archive(archive_url, destination):
    headers = {}
    token = os.getenv("SHAS_MODEL_ARCHIVE_TOKEN", "")
    if token:
        headers["Authorization"] = "Bearer " + token
    total = 0
    with requests.get(archive_url, headers=headers, stream=True, timeout=(20, 600)) as response:
        response.raise_for_status()
        with Path(destination).open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    raise ValueError("Model archive exceeds the 400 MiB limit")
                output.write(chunk)


def extract_archive(archive_path, staging_dir):
    seen = set()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if name not in MANIFEST:
                continue
            expected_size = MANIFEST[name]["size"]
            if not member.isfile() or member.size != expected_size or name in seen:
                raise ValueError("Archive contains an invalid model entry")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Archive model entry could not be read")
            with source, (Path(staging_dir) / name).open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            seen.add(name)
    missing = set(REQUIRED) - seen
    if missing:
        raise ValueError("Archive is missing required model files")
    invalid = verify_model_files(staging_dir)
    if invalid:
        raise ValueError("Archive model checksum verification failed")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    invalid = verify_model_files(MODEL_DIR)
    archive_url = os.getenv("SHAS_MODEL_ARCHIVE_URL", "").strip()
    if invalid and archive_url:
        staging_dir = Path(tempfile.mkdtemp(prefix="shas-models-", dir=str(MODEL_DIR)))
        archive_path = staging_dir / "models.tar.gz"
        try:
            download_archive(archive_url, archive_path)
            extract_archive(archive_path, staging_dir)
            for name in REQUIRED:
                os.replace(str(staging_dir / name), str(MODEL_DIR / name))
        except Exception as error:
            # Do not print exception messages: HTTP errors can contain signed archive URLs.
            print("Model asset provisioning failed (" + type(error).__name__ + ").", file=sys.stderr)
            return 1
        finally:
            shutil.rmtree(str(staging_dir), ignore_errors=True)
        invalid = verify_model_files(MODEL_DIR)
    if invalid:
        missing = [name for name in invalid if not (MODEL_DIR / name).is_file()]
        if missing and not archive_url:
            print(
                "Model files are not provisioned. Upload the original assets or configure "
                "SHAS_MODEL_ARCHIVE_URL. Missing: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 0
        print(
            "Model size or SHA-256 verification failed for: " + ", ".join(invalid),
            file=sys.stderr,
        )
        return 1
    print("All required SHAS model files passed size and SHA-256 checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
