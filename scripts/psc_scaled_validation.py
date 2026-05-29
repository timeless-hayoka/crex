"""Validation simulation for the PSC scaled engine V4 upgrades."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psc_scaled import (  # noqa: E402
    DIMENSION_POLARITY,
    PSCBatchEngine,
    benchmark_scale,
)


rng = np.random.default_rng(42)

CRISIS_THRESHOLD = 0.25
DIM = "focus"
DIMS = list(DIMENSION_POLARITY.keys())
N_RUNS = 50


def t1(n: int = 60) -> np.ndarray:
    return np.clip(np.linspace(0.85, 0.10, n) + rng.normal(0, 0.025, n), 0, 1)


def t2(n: int = 60) -> np.ndarray:
    b = np.ones(n) * 0.75
    b[30:] = np.linspace(0.75, 0.05, n - 30) ** 0.5
    return np.clip(b + rng.normal(0, 0.025, n), 0, 1)


def t3(n: int = 60) -> np.ndarray:
    b = np.concatenate([np.ones(25) * 0.72, np.linspace(0.72, 0.08, n - 25)])
    return np.clip(b + rng.normal(0, 0.038, n), 0, 1)


def t4(n: int = 60) -> np.ndarray:
    t_ = np.linspace(0, 4 * np.pi, n)
    a = np.linspace(0.05, 0.25, n)
    return np.clip(
        np.linspace(0.80, 0.15, n) + a * np.sin(t_) + rng.normal(0, 0.025, n),
        0,
        1,
    )


def t5(n: int = 60) -> np.ndarray:
    return np.clip(rng.normal(0.65, 0.075, n), 0, 1)


def t6(n: int = 60) -> np.ndarray:
    b = np.concatenate(
        [
            np.ones(10) * 0.78,
            np.linspace(0.78, 0.35, 8),
            np.linspace(0.35, 0.55, 6),
            np.linspace(0.55, 0.10, n - 24),
        ]
    )
    b[-20:] += 0.06 * np.sin(np.linspace(0, 3 * np.pi, 20))
    return np.clip(b + rng.normal(0, 0.025, n), 0, 1)


TRAJS = {
    "T1 Smooth": t1,
    "T2 Spike": t2,
    "T3 Regime": t3,
    "T4 Feedback": t4,
    "T5 Noisy": t5,
    "T6 Compound": t6,
}


def run_single_dim_sim(traj_fn, n_runs: int = N_RUNS):
    leads, fps, misses, maes = [], [], [], []
    chaos_at_alert, horizon_at_alert = [], []

    for _ in range(n_runs):
        data = traj_fn()
        crisis = next((i for i, v in enumerate(data) if v <= CRISIS_THRESHOLD), None)
        engine = PSCBatchEngine([DIM], policy="SECURITY")
        first_alert = None
        fp = 0
        preds, acts = [], []

        for cycle in range(len(data)):
            engine.push_state({DIM: float(data[cycle])})
            result = engine.run()
            actual = float(data[min(cycle + 5 - 1, len(data) - 1)])
            if result is None:
                continue
            pred_val = float(result.predicted[0])
            preds.append(pred_val)
            acts.append(actual)

            if result.alerted[0]:
                chaos_at_alert.append(float(result.chaos_scores[0]))
                horizon_at_alert.append(int(result.n_steps_used[0]))
                if crisis is not None and cycle < crisis:
                    leads.append(crisis - cycle)
                    if first_alert is None:
                        first_alert = cycle
                elif crisis is None or cycle >= crisis:
                    fp += 1

        miss = int(
            (crisis is not None and first_alert is None)
            or (crisis is not None and first_alert is not None and first_alert >= crisis)
        )
        fps.append(fp)
        misses.append(miss)
        if preds:
            maes.append(float(np.mean(np.abs(np.array(preds) - np.array(acts)))))

    return {
        "lead_mean": np.mean(leads) if leads else 0.0,
        "miss_rate": np.mean(misses),
        "fp_mean": np.mean(fps),
        "mae": np.mean(maes) if maes else 0.0,
        "chaos_at_alert": chaos_at_alert,
        "horizon_at_alert": horizon_at_alert,
    }


def maybe_render_figure(results, bench, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    traj_labels = [t.split(" ", 1)[-1] for t in TRAJS]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "PSC Scaled Engine - Validation Results\nDRIFT V4",
        fontsize=13,
        fontweight="bold",
    )

    colors6 = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    ax = axes[0, 0]
    chaos_by_traj = [results[t]["chaos_at_alert"] for t in TRAJS]
    bp = ax.boxplot(
        [c if c else [0] for c in chaos_by_traj],
        patch_artist=True,
        medianprops=dict(color="white", lw=2),
    )
    for patch, col in zip(bp["boxes"], colors6):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    for elem in ["whiskers", "caps", "fliers"]:
        for item in bp[elem]:
            item.set_color("#888")
    ax.axhline(0.65, color="red", ls="--", lw=1.5, label="Chaos threshold (0.65)")
    ax.axhline(0.20, color="#2ecc71", ls="--", lw=1.5, label="Stable threshold (0.20)")
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(traj_labels, rotation=18, ha="right", fontsize=8, color="#ccc")
    ax.set_ylabel("Chaos Score at Alert", color="#ccc", fontsize=9)
    ax.set_title("Continuous Chaos Score Distribution", color="white", fontsize=10)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#333")

    ax = axes[0, 1]
    horizons_by_traj = [results[t]["horizon_at_alert"] for t in TRAJS]
    bp2 = ax.boxplot(
        [h if h else [5] for h in horizons_by_traj],
        patch_artist=True,
        medianprops=dict(color="white", lw=2),
    )
    for patch, col in zip(bp2["boxes"], colors6):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    for elem in ["whiskers", "caps", "fliers"]:
        for item in bp2[elem]:
            item.set_color("#888")
    ax.axhline(5, color="#f39c12", ls="--", lw=1.5, label="Static N_STEPS=5 baseline")
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(traj_labels, rotation=18, ha="right", fontsize=8, color="#ccc")
    ax.set_ylabel("Prediction Horizon (cycles)", color="#ccc", fontsize=9)
    ax.set_title("Dynamic Prediction Horizon", color="white", fontsize=10)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#333")

    ax = axes[1, 0]
    dim_list = sorted(bench.keys())
    means = [bench[d]["mean_us"] for d in dim_list]
    p99s = [bench[d]["p99_us"] for d in dim_list]
    ax.plot(dim_list, means, "o-", color="#2ecc71", lw=2.5, ms=8, label="Mean latency")
    ax.plot(dim_list, p99s, "s--", color="#e74c3c", lw=2, ms=7, label="p99 latency")
    ax.axvline(16, color="#f39c12", lw=2, ls=":", label="DRIFT dims (16)")
    ax.fill_between(dim_list, means, p99s, alpha=0.1, color="#3498db")
    ax.set_xlabel("Number of Dimensions", color="#ccc", fontsize=9)
    ax.set_ylabel("Latency (us)", color="#ccc", fontsize=9)
    ax.set_title("Scale Benchmark - Latency vs Dimensions", color="white", fontsize=10)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#333")

    ax = axes[1, 1]
    x = np.arange(len(TRAJS))
    w = 0.28
    leads_ = [results[t]["lead_mean"] for t in TRAJS]
    misses_ = [results[t]["miss_rate"] * 10 for t in TRAJS]
    fps_ = [results[t]["fp_mean"] for t in TRAJS]
    ax.bar(x - w, leads_, w, label="Lead time (cycles)", color="#2ecc71", alpha=0.85)
    ax.bar(x, misses_, w, label="Miss rate x10", color="#e74c3c", alpha=0.85)
    ax.bar(x + w, fps_, w, label="False pos/run", color="#f39c12", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(traj_labels, rotation=18, ha="right", fontsize=8, color="#ccc")
    ax.set_title("Scaled Engine: Alert Quality Summary", color="white", fontsize=10)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#333")

    fig.patch.set_facecolor("#0d1117")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "psc_scaled_validation.png"),
        help="figure output path",
    )
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("PSC SCALED ENGINE - VALIDATION SIMULATION")
    print("=" * 70)

    print("\nSECTION 1 - Per-Trajectory Performance (scaled engine, focus dim)")
    print("-" * 70)
    results = {}
    for t_name, t_fn in TRAJS.items():
        r = run_single_dim_sim(t_fn)
        results[t_name] = r
        print(
            f"  {t_name:<18} lead={r['lead_mean']:5.1f}  "
            f"miss={r['miss_rate']:.2f}  FP={r['fp_mean']:4.1f}  MAE={r['mae']:.4f}"
        )

    print("\nSECTION 2 - Continuous Chaos Score: Alert Distribution by Chaos Level")
    print("-" * 70)
    for t_name, r in results.items():
        if r["chaos_at_alert"]:
            print(
                f"  {t_name:<18} chaos@alert mean={np.mean(r['chaos_at_alert']):.3f}  "
                f"range=[{min(r['chaos_at_alert']):.2f}, {max(r['chaos_at_alert']):.2f}]"
            )

    print("\nSECTION 3 - Dynamic N_STEPS: Horizon Used at Alert Time")
    print("-" * 70)
    for t_name, r in results.items():
        if r["horizon_at_alert"]:
            print(
                f"  {t_name:<18} N_steps mean={np.mean(r['horizon_at_alert']):.1f}  "
                f"range=[{min(r['horizon_at_alert'])}, {max(r['horizon_at_alert'])}]"
            )

    print("\nSECTION 4 - SCALE BENCHMARK")
    print("-" * 70)
    bench = benchmark_scale([16, 50, 100, 200, 500], n_cycles=500)
    print(f"\n  {'Dims':>6} {'Mean us':>10} {'p99 us':>10} {'cycles/sec':>14} {'DRIFT fit?':>12}")
    print(f"  {'-' * 58}")
    for dim_count, r in bench.items():
        fps_est = r["cycles_per_sec"]
        fit = "YES (1Hz)" if fps_est > 1 else "NO"
        if fps_est > 10:
            fit = "YES (10Hz)"
        if fps_est > 100:
            fit = "YES (100Hz)"
        print(
            f"  {dim_count:>6} {r['mean_us']:>10.1f} {r['p99_us']:>10.1f} "
            f"{fps_est:>14.0f} {fit:>12}"
        )

    print(f"\n  DRIFT production dims: {len(DIMS)}")
    print(
        f"  At 16 dims: {bench[16]['mean_us']:.0f}us per cycle = "
        f"{bench[16]['cycles_per_sec']:.0f} PSC cycles/sec capacity"
    )

    if not args.no_figure:
        maybe_render_figure(results, bench, Path(args.output))
        print(f"\n[SIM] Figure saved to {args.output}")

    print("=" * 70)


if __name__ == "__main__":
    main()
