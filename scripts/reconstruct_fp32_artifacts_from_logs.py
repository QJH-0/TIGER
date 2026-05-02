from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s*/\s*(\d+)\s+"
    r"(?:(?:train/loss:\s+(-?\d+(?:\.\d+)?))\s+)?"
    r"val/loss:\s+(-?\d+(?:\.\d+)?)\s+"
    r"val/si_snr:\s+(-?\d+(?:\.\d+)?)\s+"
    r"epoch:\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE | re.DOTALL,
)
SIZE_RE = re.compile(
    r"\[TIGER\]\s+size_mb\s+fp32_estimated=(\d+(?:\.\d+)?)\s+actual=(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def read_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return strip_ansi(raw)


def parse_epoch_rows(text: str) -> list[dict]:
    rows: list[dict] = []
    for match in EPOCH_RE.finditer(text):
        epoch = int(match.group(1))
        max_epochs = int(match.group(2))
        train_loss = match.group(3)
        val_loss = float(match.group(4))
        val_si_snr = float(match.group(5))
        epoch_float = float(match.group(6))
        rows.append(
            {
                "epoch": epoch,
                "max_epochs": max_epochs,
                "train_loss": float(train_loss) if train_loss is not None else None,
                "val_loss": val_loss,
                "val_si_snr": val_si_snr,
                "epoch_float": epoch_float,
            }
        )
    return rows


def merge_rows(log_paths: list[Path]) -> tuple[list[dict], dict]:
    merged: dict[int, dict] = {}
    meta = {
        "fp32_estimated_size_mb": None,
        "actual_param_size_mb": None,
    }
    for log_path in log_paths:
        text = read_text(log_path)
        size_match = SIZE_RE.search(text)
        if size_match and meta["fp32_estimated_size_mb"] is None:
            meta["fp32_estimated_size_mb"] = float(size_match.group(1))
            meta["actual_param_size_mb"] = float(size_match.group(2))
        for row in parse_epoch_rows(text):
            existing = merged.get(row["epoch"])
            if existing is None:
                merged[row["epoch"]] = row
                continue
            if existing["train_loss"] is None and row["train_loss"] is not None:
                merged[row["epoch"]] = row
    ordered = [merged[k] for k in sorted(merged)]
    return ordered, meta


def discover_logs(exp_dir: Path) -> list[Path]:
    preferred = [
        exp_dir / "output (2).log",
        exp_dir / "output (1).log",
        exp_dir / "output.log",
    ]
    logs = [path for path in preferred if path.exists()]
    if logs:
        return logs
    return sorted(exp_dir.glob("output*.log"))


def read_best_val_loss(exp_dir: Path, rows: list[dict]) -> float:
    best_k_path = exp_dir / "best_k_models.json"
    if best_k_path.exists():
        data = json.loads(best_k_path.read_text(encoding="utf-8"))
        if data:
            return min(float(v) for v in data.values())
    return min(row["val_loss"] for row in rows)


def write_history_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])
        for row in rows:
            train_loss = "" if row["train_loss"] is None else row["train_loss"]
            writer.writerow([row["epoch"], train_loss, row["val_loss"], ""])


def build_kaggle_exp_path(exp_name: str) -> str:
    return f"/kaggle/working/TIGER/Experiments/{exp_name}"


def build_final_metrics_payload(exp_dir: Path, rows: list[dict], meta: dict) -> dict:
    conf_path = exp_dir / "conf.yml"
    exp_name = exp_dir.name
    kaggle_exp_dir = build_kaggle_exp_path(exp_name)
    best_val_loss = read_best_val_loss(exp_dir, rows)
    last_row = rows[-1]
    max_epochs = max(row["max_epochs"] for row in rows)
    payload = {
        "config": {
            "exp_name": exp_name,
            "exp_dir": kaggle_exp_dir,
            "binary_stage_epochs": None,
            "freeze_scope": [],
            "total_epochs_completed": last_row["epoch"],
            "max_epochs": max_epochs,
        },
        "metrics": {
            "val_loss": last_row["val_loss"],
            "best_val_loss": best_val_loss,
            "best_checkpoint_path": f"{kaggle_exp_dir}/checkpoints/best.ckpt",
        },
        "sensitivity": {},
        "efficiency": {},
    }
    if last_row["train_loss"] is not None:
        payload["metrics"]["train_loss"] = last_row["train_loss"]
    if meta.get("fp32_estimated_size_mb") is not None:
        payload["efficiency"]["fp32_estimated_size_mb"] = meta["fp32_estimated_size_mb"]
    if meta.get("actual_param_size_mb") is not None:
        payload["efficiency"]["actual_param_size_mb"] = meta["actual_param_size_mb"]
    if conf_path.exists():
        payload["config"]["conf_path"] = f"{kaggle_exp_dir}/conf.yml"
    return payload


def write_final_metrics(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct FP32 experiment artifacts from training logs.")
    parser.add_argument(
        "--exp-dir",
        default=r"Experiments\TIGER-Small-MiniLibriMix-Kaggle-T4x2",
        help="Experiment directory containing output logs and checkpoints.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    log_paths = discover_logs(exp_dir)
    if not log_paths:
        raise FileNotFoundError(f"No output logs found under: {exp_dir}")

    rows, meta = merge_rows(log_paths)
    if not rows:
        raise RuntimeError(f"No epoch records parsed from logs under: {exp_dir}")

    history_path = exp_dir / "history.csv"
    final_metrics_path = exp_dir / "final_metrics.json"
    write_history_csv(history_path, rows)
    write_final_metrics(final_metrics_path, build_final_metrics_payload(exp_dir, rows, meta))

    print(f"Recovered {len(rows)} epochs from {len(log_paths)} logs.")
    print(f"history.csv -> {history_path}")
    print(f"final_metrics.json -> {final_metrics_path}")
    print(f"Best val_loss: {read_best_val_loss(exp_dir, rows):.8f}")
    print(f"Final epoch: {rows[-1]['epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
