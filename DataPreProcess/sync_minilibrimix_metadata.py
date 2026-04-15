import argparse
import csv
from pathlib import Path

import soundfile as sf


REQUIRED_SUBDIRS = ["mix_both", "mix_clean", "s1", "s2", "noise"]
MIX_TYPES = ["mix_both", "mix_clean"]


def list_wavs(dir_path: Path):
    return {p.name for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".wav"}


def collect_common_ids(dataset_root: Path, split: str):
    split_dir = dataset_root / split
    file_sets = []
    for subdir in REQUIRED_SUBDIRS:
        subdir_path = split_dir / subdir
        if not subdir_path.exists():
            raise FileNotFoundError(f"Missing directory: {subdir_path}")
        file_sets.append(list_wavs(subdir_path))

    common_ids = sorted(set.intersection(*file_sets))
    if not common_ids:
        raise RuntimeError(f"No aligned wav ids found in split: {split}")
    return common_ids


def build_rows(dataset_root: Path, dataset_name: str, split: str, mix_type: str, common_ids):
    rows = []
    for idx, wav_name in enumerate(common_ids):
        rel_prefix = f"{dataset_name}/{split}"
        mix_path = f"{rel_prefix}/{mix_type}/{wav_name}"
        s1_path = f"{rel_prefix}/s1/{wav_name}"
        s2_path = f"{rel_prefix}/s2/{wav_name}"
        noise_path = f"{rel_prefix}/noise/{wav_name}"
        abs_mix_path = dataset_root / split / mix_type / wav_name
        length = float(sf.info(str(abs_mix_path)).frames)
        rows.append([idx, wav_name.replace(".wav", ""), mix_path, s1_path, s2_path, noise_path, length])
    return rows


def write_csv(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "mixture_ID", "mixture_path", "source_1_path", "source_2_path", "noise_path", "length"])
        writer.writerows(rows)


def sync_metadata(dataset_root: Path, metadata_dir: Path, splits):
    dataset_name = dataset_root.name
    for split in splits:
        common_ids = collect_common_ids(dataset_root, split)
        for mix_type in MIX_TYPES:
            rows = build_rows(dataset_root, dataset_name, split, mix_type, common_ids)
            out_csv = metadata_dir / f"mixture_{split}_{mix_type}.csv"
            write_csv(out_csv, rows)
            print(f"Wrote: {out_csv} ({len(rows)} rows)")


def parse_args():
    parser = argparse.ArgumentParser(description="Sync MiniLibriMix metadata CSV files with current dataset files.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=r"D:\Paper\datasets\MiniLibriMix",
        help="MiniLibriMix dataset root containing train/val/test directories.",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default=r"D:\Paper\datasets\MiniLibriMix\metadata",
        help="Output metadata directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        help="Dataset splits to sync metadata for.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sync_metadata(
        dataset_root=Path(args.dataset_root),
        metadata_dir=Path(args.metadata_dir),
        splits=args.splits,
    )
