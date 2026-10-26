"""Run the full benchmark and noise study and save the results to results/.

Usage:
    python scripts/run_benchmark.py                 # 5 seeds, ~30-40 minutes on a laptop
    python scripts/run_benchmark.py --seeds 2 --skip-noise

The notebook reads the CSV files written here, so it does not need to retrain
every model.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmlbench import QuantumConfig, noise_study, run_benchmark, summarize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5, help="number of train/test splits")
    parser.add_argument("--noise-seeds", type=int, default=3, help="seeds for the noise study")
    parser.add_argument("--skip-noise", action="store_true", help="skip the VQC noise study")
    parser.add_argument("--noise-only", action="store_true",
                        help="run only the noise study (keeps an existing benchmark.csv)")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    config = QuantumConfig()
    started = time.perf_counter()

    if not args.noise_only:
        results = run_benchmark(seeds=range(args.seeds), config=config, progress=print)
        results.to_csv(args.out / "benchmark.csv", index=False)
        print(summarize(results).to_string())

    if not args.skip_noise:
        # One seed at a time, saved after each, so an interrupted run keeps its progress.
        noise_path = args.out / "noise.csv"
        done = pd.read_csv(noise_path) if noise_path.exists() else pd.DataFrame()
        finished = set(done.seed) if len(done) else set()
        for seed in range(args.noise_seeds):
            if seed in finished:
                print(f"noise seed={seed} already in {noise_path.name}, skipping")
                continue
            part = noise_study(seeds=[seed], config=config, progress=print)
            done = pd.concat([done, part], ignore_index=True)
            done.to_csv(noise_path, index=False)

    info = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_s": round(time.perf_counter() - started, 1),
        "seeds": args.seeds,
        "noise_seeds": None if args.skip_noise else args.noise_seeds,
        "quantum_config": asdict(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in (
            "qiskit", "qiskit-aer", "qiskit-machine-learning", "scikit-learn", "numpy")},
    }
    (args.out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"done in {info['runtime_s']} s -> {args.out}")


if __name__ == "__main__":
    main()
