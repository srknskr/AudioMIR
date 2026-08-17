import argparse
import os
import re
from pathlib import Path
from typing import Optional
import pandas as pd


def infer_bpm_from_name(filename: str) -> Optional[float]:
    """Look for patterns like '120bpm', '120_bpm', 'bpm120' in filename."""
    match = re.search(r"(\d{2,3})\s*(?:bpm|_bpm)", filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def create_manifest(
    audio_dir: str,
    output_manifest: str,
    metadata_csv: Optional[str] = None,
    infer_from_filename: bool = False,
    default_genre: str = "electronic",
    default_meter: str = "4/4",
) -> None:
    audio_path = Path(audio_dir)
    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.flac", "*.aif", "*.aiff"]:
        audio_files.extend(list(audio_path.rglob(ext)))

    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    meta_df = None
    if metadata_csv and Path(metadata_csv).exists():
        meta_df = pd.read_csv(metadata_csv)
        print(f"Loaded external metadata with {len(meta_df)} rows.")

    records = []
    for f in audio_files:
        rel_path = str(f.relative_to(audio_path.parent))
        filename = f.stem

        bpm = 120.0
        genre = default_genre
        meter = default_meter
        source_id = f.parent.name or filename

        # Infer from filename if requested
        if infer_from_filename:
            inferred_bpm = infer_bpm_from_name(f.name)
            if inferred_bpm:
                bpm = inferred_bpm
            # Split filename parts to find source_id base
            parts = filename.split("_")
            if len(parts) > 1:
                source_id = "_".join(parts[:-1])

        records.append({
            "audio_path": rel_path,
            "bpm": bpm,
            "meter": meter,
            "genre": genre,
            "source_id": source_id,
        })

    df = pd.DataFrame(records)
    out_p = Path(output_manifest)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    print(f"Manifest created successfully at {out_p} with {len(df)} entries and {df['source_id'].nunique()} unique source groups.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create custom drum loop manifest with anti-leakage grouping")
    parser.add_argument("--audio-dir", type=str, required=True, help="Directory containing audio loops")
    parser.add_argument("--output", type=str, default="data/custom_manifest.csv", help="Output manifest CSV path")
    parser.add_argument("--metadata", type=str, default=None, help="Optional external metadata CSV")
    parser.add_argument("--infer-from-filename", action="store_true", help="Infer BPM and source_id from filenames")
    args = parser.parse_args()

    create_manifest(
        audio_dir=args.audio_dir,
        output_manifest=args.output,
        metadata_csv=args.metadata,
        infer_from_filename=args.infer_from_filename,
    )
