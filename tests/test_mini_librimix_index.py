import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DataPreProcess.build_mini_librimix_index import build_mini_librimix_index


def _write_wav(path: Path, length: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(length, dtype=np.float32), 16000)


def test_build_mini_librimix_index_creates_small_aligned_indexes(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "mini_index"

    split_counts = {"train": 24, "val": 12, "test": 12}
    selected_counts = {"train": 20, "val": 10, "test": 10}
    speakers = ("mix_both", "s1", "s2")

    for split, count in split_counts.items():
        for idx in range(count):
            file_name = f"sample_{idx:03d}.wav"
            for speaker in speakers:
                _write_wav(raw_dir / split / speaker / file_name, 1600 + idx)

    build_mini_librimix_index(str(raw_dir), str(out_dir))

    for split, expected_count in selected_counts.items():
        split_entries = {}
        for speaker in speakers:
            json_path = out_dir / split / f"{speaker}.json"
            assert json_path.exists()

            with json_path.open("r", encoding="utf-8") as handle:
                split_entries[speaker] = json.load(handle)

            assert len(split_entries[speaker]) == expected_count
            assert Path(split_entries[speaker][0][0]).is_absolute()

        selected_names = [
            Path(entry[0]).name for entry in split_entries["mix_both"]
        ]
        assert selected_names == [
            f"sample_{idx:03d}.wav" for idx in range(expected_count)
        ]

        for speaker in speakers[1:]:
            assert [Path(entry[0]).name for entry in split_entries[speaker]] == selected_names

        assert [entry[1] for entry in split_entries["mix_both"]] == [
            1600 + idx for idx in range(expected_count)
        ]
