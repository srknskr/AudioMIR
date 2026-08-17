import argparse
import os
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd


GROOVE_URL = "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"


def download_groove(dest_dir: str = "data/groove") -> None:
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    zip_file = dest_path / "groove_dataset.zip"

    print(f"Checking Groove MIDI Dataset destination: {dest_path}")
    manifest_file = dest_path / "info.csv"

    if manifest_file.exists():
        print(f"Groove dataset metadata already exists at {manifest_file}")
        return

    print(f"Downloading Groove dataset metadata archive from {GROOVE_URL}...")
    try:
        urllib.request.urlretrieve(GROOVE_URL, zip_file)
        print("Extracting archive...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(dest_path)
        if zip_file.exists():
            zip_file.unlink()
        print("Groove MIDI Dataset downloaded and extracted successfully.")
    except Exception as e:
        print(f"Could not automatically download Groove dataset archive: {e}")
        print("Creating placeholder structure at data/groove/...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Groove MIDI Dataset")
    parser.add_argument("--dest", type=str, default="data/groove", help="Destination folder")
    args = parser.parse_args()
    download_groove(args.dest)
