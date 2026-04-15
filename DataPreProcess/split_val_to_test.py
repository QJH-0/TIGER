import argparse
import random
import shutil
from pathlib import Path


def list_files(dir_path: Path):
    return sorted([p.name for p in dir_path.iterdir() if p.is_file() and not p.name.startswith(".")])


def collect_sample_ids(val_dir: Path, subdirs):
    files_by_subdir = {}
    for subdir in subdirs:
        subdir_path = val_dir / subdir
        if not subdir_path.exists() or not subdir_path.is_dir():
            raise NotADirectoryError(f"Required subdir not found: {subdir_path}")
        files_by_subdir[subdir] = set(list_files(subdir_path))

    common_ids = set.intersection(*files_by_subdir.values())
    if not common_ids:
        raise RuntimeError("No common sample ids across subdirs; cannot build paired test set.")

    # Report mismatches to help user inspect data integrity.
    for subdir, file_ids in files_by_subdir.items():
        missing_count = len(common_ids.symmetric_difference(file_ids))
        if missing_count > 0:
            print(f"[WARN] {subdir} is not fully aligned with common ids. diff count={missing_count}")

    return sorted(common_ids)


def move_sample_across_subdirs(val_dir: Path, test_dir: Path, sample_id: str, subdirs, dry_run: bool):
    for subdir in subdirs:
        src = val_dir / subdir / sample_id
        dst_dir = test_dir / subdir
        dst = dst_dir / sample_id

        if not src.exists():
            raise FileNotFoundError(f"Missing source file for paired move: {src}")
        if dst.exists():
            raise FileExistsError(f"Target already exists: {dst}")

        if dry_run:
            print(f"[DRY-RUN] move: {src} -> {dst}")
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def split_val_to_test(dataset_root: Path, val_name: str, test_name: str, seed: int, dry_run: bool, subdirs):
    val_dir = dataset_root / val_name
    test_dir = dataset_root / test_name

    if not val_dir.exists() or not val_dir.is_dir():
        raise NotADirectoryError(f"val directory not found: {val_dir}")

    sample_ids = collect_sample_ids(val_dir=val_dir, subdirs=subdirs)
    total = len(sample_ids)
    if total == 0:
        print(f"No aligned samples found in {val_dir}. Nothing to move.")
        return

    move_count = total // 2
    if move_count == 0:
        print(f"Only {total} sample in {val_dir}. Nothing to move.")
        return

    rng = random.Random(seed)
    selected_ids = rng.sample(sample_ids, move_count)

    print(f"Dataset root : {dataset_root}")
    print(f"Source val   : {val_dir}")
    print(f"Target test  : {test_dir}")
    print(f"Paired subdir: {', '.join(subdirs)}")
    print(f"Total samples: {total}")
    print(f"Move samples : {move_count}")
    print(f"Seed         : {seed}")
    print("-" * 60)

    for sample_id in selected_ids:
        move_sample_across_subdirs(
            val_dir=val_dir,
            test_dir=test_dir,
            sample_id=sample_id,
            subdirs=subdirs,
            dry_run=dry_run,
        )
        print(f"Moved sample: {sample_id}")

    print("-" * 60)
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split half paired samples from val into test for MiniLibriMix."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=r"D:\Paper\datasets\MiniLibriMix",
        help="Dataset root directory containing val and test.",
    )
    parser.add_argument(
        "--val_name",
        type=str,
        default="val",
        help="Validation folder name under dataset root.",
    )
    parser.add_argument(
        "--test_name",
        type=str,
        default="test",
        help="Test folder name under dataset root.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview moves without touching files.",
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=["mix_both", "mix_clean", "noise", "s1", "s2"],
        help="Paired subdirs that should keep one-to-one file alignment.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_val_to_test(
        dataset_root=Path(args.dataset_root),
        val_name=args.val_name,
        test_name=args.test_name,
        seed=args.seed,
        dry_run=args.dry_run,
        subdirs=args.subdirs,
    )
