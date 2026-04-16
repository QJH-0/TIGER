import argparse
import json
import os
from pathlib import Path

import soundfile as sf


DEFAULT_SPLIT_COUNTS = {"train": 20, "val": 10, "test": 10}
DEFAULT_SPEAKERS = ("mix_both", "s1", "s2")


def _sorted_wavs(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix == ".wav")


def _select_aligned_files(split_dir: Path, count: int, speakers: tuple[str, ...]) -> dict[str, list[Path]]:
    selected = {speaker: _sorted_wavs(split_dir / speaker)[:count] for speaker in speakers}

    if any(len(paths) < count for paths in selected.values()):
        raise ValueError(f"Split '{split_dir.name}' does not contain at least {count} wav files per speaker")

    reference_names = [path.name for path in selected[speakers[0]]]
    for speaker in speakers[1:]:
        if [path.name for path in selected[speaker]] != reference_names:
            raise ValueError(f"Split '{split_dir.name}' has misaligned filenames across speakers")

    return selected


def _write_index_file(paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_infos = []
    for wav_path in paths:
        samples = sf.SoundFile(str(wav_path))
        file_infos.append((str(wav_path.resolve()), len(samples)))

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(file_infos, handle, indent=4)


def build_mini_librimix_index(
    in_dir: str,
    out_dir: str,
    split_counts: dict[str, int] | None = None,
    speakers: tuple[str, ...] = DEFAULT_SPEAKERS,
) -> None:
    raw_root = Path(in_dir)
    output_root = Path(out_dir)
    split_counts = DEFAULT_SPLIT_COUNTS if split_counts is None else split_counts

    for split, count in split_counts.items():
        selected = _select_aligned_files(raw_root / split, count, speakers)
        for speaker, paths in selected.items():
            _write_index_file(paths, output_root / split / f"{speaker}.json")


def _parse_args():
    parser = argparse.ArgumentParser("Build a small MiniLibriMix JSON index for smoke tests")
    parser.add_argument("--in_dir", type=str, required=True, help="Raw MiniLibriMix root directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for JSON indexes")
    parser.add_argument("--train_count", type=int, default=DEFAULT_SPLIT_COUNTS["train"])
    parser.add_argument("--val_count", type=int, default=DEFAULT_SPLIT_COUNTS["val"])
    parser.add_argument("--test_count", type=int, default=DEFAULT_SPLIT_COUNTS["test"])
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_mini_librimix_index(
        args.in_dir,
        args.out_dir,
        split_counts={
            "train": args.train_count,
            "val": args.val_count,
            "test": args.test_count,
        },
    )
