"""Optionally provision required model files from a private tar.gz URL."""
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("SHAS_MODEL_DIR", str(ROOT))).expanduser().resolve()
REQUIRED = ("kni_axe.pt", "ACCIDENT.pt", "fire.pt", "yolov5s.pt", "model_new.h5")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED if not (MODEL_DIR / name).is_file()]
    archive_url = os.getenv("SHAS_MODEL_ARCHIVE_URL", "").strip()
    if missing and archive_url:
        headers = {}
        token = os.getenv("SHAS_MODEL_ARCHIVE_TOKEN", "")
        if token:
            headers["Authorization"] = "Bearer " + token
        with tempfile.NamedTemporaryFile(dir=str(MODEL_DIR), suffix=".tar.gz", delete=False) as tmp:
            archive_path = Path(tmp.name)
        try:
            with requests.get(archive_url, headers=headers, stream=True, timeout=(20, 600)) as response:
                response.raise_for_status()
                with archive_path.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            allowed = set(REQUIRED)
            with tarfile.open(archive_path, mode="r:gz") as archive:
                for member in archive.getmembers():
                    filename = Path(member.name).name
                    if filename not in allowed or not member.isfile():
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        continue
                    target = MODEL_DIR / filename
                    with source, target.open("wb") as output:
                        while True:
                            chunk = source.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
        finally:
            archive_path.unlink(missing_ok=True)
    missing = [name for name in REQUIRED if not (MODEL_DIR / name).is_file()]
    if missing:
        print(
            "SHAS model assets are not provisioned. Detection routes will report model errors. "
            "Place these files in SHAS_MODEL_DIR or configure SHAS_MODEL_ARCHIVE_URL to a "
            "private .tar.gz containing the original files: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 0
    print("All required SHAS model assets are present in " + str(MODEL_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
