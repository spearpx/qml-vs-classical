"""Run the tuned benchmark and the VQC noise study, saving results to results/.

Usage:
    python scripts/run_benchmark.py                          # everything, 5 seeds
    python scripts/run_benchmark.py --datasets iris          # one dataset (run several in parallel)
    python scripts/run_benchmark.py --noise-only             # noise study only
    python scripts/run_benchmark.py --untuned --seeds 5      # fixed-default baseline

Every (dataset, seed) result is appended to a per-dataset CSV as soon as it
finishes, and finished pairs are skipped on rerun, so an interrupted run can
simply be restarted. The notebook reads these CSV files.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmlbench import DATASETS, QuantumConfig, noise_study, run_benchmark  # noqa: E402


def _append(path: Path, part: pd.DataFrame) -> None:
    part.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_models(args, config: QuantumConfig) -> None:
    tag = "untuned" if args.untuned else "tuned"
    for dataset in args.datasets:
        path = args.out / f"benchmark_{tag}_{dataset}.csv"
        done = set(pd.read_csv(path).seed) if path.exists() else set()
        for seed in range(args.seeds):
            if seed in done:
                print(f"{dataset} seed={seed} already saved, skipping", flush=True)
                continue
            part = run_benchmark(datasets=[dataset], seeds=[seed], config=config,
                                 tuned=not args.untuned, progress=lambda m: print(m, flush=True))
            _append(path, part)


def tuned_vqc_setting(args) -> tuple[float, int]:
    """Most frequently selected VQC (feature scale, ansatz reps) on moons."""
    path = args.out / "benchmark_tuned_moons.csv"
    if not path.exists():
        raise SystemExit("run the tuned moons benchmark before the noise study")
    df = pd.read_csv(path)
    params = df[df.model == "vqc"].best_params.map(json.loads)
    setting = params.map(lambda p: (p["feature_scale"], p["ansatz_reps"])).mode().iloc[0]
    scale = 1 / np.pi if setting[0] == "[0, 1]" else 1.0
    return scale, int(setting[1])


def run_noise(args, config: QuantumConfig) -> None:
    scale, reps = tuned_vqc_setting(args)
    noise_config = replace(config, ansatz_reps=reps)
    print(f"noise study with tuned VQC: feature_scale={scale:.3f}, ansatz_reps={reps}", flush=True)
    path = args.out / "noise.csv"
    done = set(pd.read_csv(path).seed) if path.exists() else set()
    for seed in range(args.noise_seeds):
        if seed in done:
            print(f"noise seed={seed} already saved, skipping", flush=True)
            continue
        part = noise_study(seeds=[seed], config=noise_config, feature_scale=scale,
                           progress=lambda m: print(m, flush=True))
        _append(path, part)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--seeds", type=int, default=5, help="number of train/test splits")
    parser.add_argument("--untuned", action="store_true", help="use fixed defaults instead of CV tuning")
    parser.add_argument("--noise-seeds", type=int, default=3, help="seeds for the noise study")
    parser.add_argument("--skip-noise", action="store_true", help="skip the VQC noise study")
    parser.add_argument("--noise-only", action="store_true", help="run only the noise study")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    config = QuantumConfig()
    started = time.perf_counter()
    if not args.noise_only:
        run_models(args, config)
    if not args.skip_noise:
        run_noise(args, config)

    info = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_invocation": sys.argv[1:],
        "runtime_s": round(time.perf_counter() - started, 1),
        "quantum_config": asdict(config),
        "cv_folds": 3,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in (
            "qiskit", "qiskit-aer", "qiskit-machine-learning", "scikit-learn", "numpy")},
    }
    (args.out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"done in {info['runtime_s']} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
