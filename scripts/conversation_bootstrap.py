"""Conversation-level paired bootstrap confidence intervals for the transcript benchmark.

Why: the report's limitation 3 notes that attribute decisions within a transcript are
correlated, so 4,050 decisions are not 4,050 independent samples, and its recommended
next steps ask for bootstrap confidence intervals at the conversation level. This script
adds them without changing any existing table, chart or script.

What it reads (all committed):
  data/benchmark_1000/comparison_metrics.json   per-conversation tp/fp/fn/tn for 11 systems
  data/benchmark_1000/jev_noul_test_results.json + results/jev_noul_comparison_1000.json
                                                 Noul probabilities and the validation-selected
                                                 thresholds, to add the calibrated Noul row
  data/benchmark_1000/split.json                 truth for the Noul row

What it does: resamples the 150 locked test conversations with replacement, 1,000 times,
seed 20260919 (the split seed). Every system is scored on the same resamples, so the
difference between two systems is paired. Point estimates equal the published numbers.

Metrics per resample: micro accuracy, precision, recall, F1 over all decisions in the
resample, and exact-match rate (all 27 attributes right) per conversation.

Output: data/benchmark_1000/ci_metrics.json and report/conversation-level-cis.md.
Dependencies: numpy only.

Usage, from the repository root:
    python3 scripts/conversation_bootstrap.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmark_1000"
RESULTS = ROOT / "results"
REPORT = ROOT / "report"

N_RESAMPLES = 1000
SEED = 20260919
ALPHA = 0.05

# Display names and whether the row is a re-runnable model (API or local run with
# committed per-decision outputs). Sol and Luna are the original agent-session
# outputs recorded in the repository; their rows are reported as committed and are
# kept out of the paired-difference table.
SYSTEMS = {
    "sol": ("GPT-5.6 Sol (as reported)", False),
    "jev_noul_calibrated": ("JEV Noul, validation-calibrated", True),
    "luna": ("GPT-5.6 Luna (as reported)", False),
    "modernbert_trained": ("ModernBERT trained heads", True),
    "gliclass_modern_large_v3_opt_threshold": ("GLiClass Modern Large v3, calibrated", True),
    "gliclass_large_v3_opt_threshold": ("GLiClass Large v3, calibrated", True),
    "modernbert_zero_shot_opt_threshold": ("ModernBERT zero-shot, calibrated", True),
    "jev": ("JEV Choice, 0.50", True),
    "gliclass_modern_large_v3": ("GLiClass Modern Large v3, 0.50", True),
    "gliclass_large_v3": ("GLiClass Large v3, 0.50", True),
    "modernbert_zero_shot": ("ModernBERT zero-shot, 0.50", True),
    "deberta_v3_large_zero_shot_c": ("DeBERTa-v3-large zero-shot, 0.50", True),
}


def load_counts() -> tuple[list[int], dict[str, np.ndarray]]:
    """Return test ids and, per system, an array [n_conversations, 4] of tp, fp, fn, tn."""
    comparison = json.loads((DATA / "comparison_metrics.json").read_text())
    ids = [int(i) for i in comparison["metadata"]["test_ids"]]
    rows = comparison["rows"]
    counts: dict[str, np.ndarray] = {}
    for system in comparison["models"]:
        arr = np.zeros((len(ids), 4), dtype=np.int64)
        for k, cid in enumerate(ids):
            cell = rows[str(cid)][system]
            arr[k] = (cell["tp"], cell["fp"], cell["fn"], cell["tn"])
        counts[system] = arr
    counts["jev_noul_calibrated"] = noul_counts(ids)
    return ids, counts


def noul_counts(ids: list[int]) -> np.ndarray:
    """Per-conversation confusion counts for Noul at the validation-selected thresholds."""
    probs = json.loads((DATA / "jev_noul_test_results.json").read_text())["rows"]
    thresholds = json.loads((RESULTS / "jev_noul_comparison_1000.json").read_text())["thresholds"]
    truth = {int(r["id"]): r["truth"] for r in json.loads((DATA / "split.json").read_text())["rows"]}
    arr = np.zeros((len(ids), 4), dtype=np.int64)
    for k, cid in enumerate(ids):
        p = probs[str(cid)]["probabilities"]
        t = truth[cid]
        tp = fp = fn = tn = 0
        for attribute, threshold in thresholds.items():
            pred = int(float(p[attribute]) >= float(threshold))
            y = int(t[attribute])
            tp += y == 1 and pred == 1
            fp += y == 0 and pred == 1
            fn += y == 1 and pred == 0
            tn += y == 0 and pred == 0
        arr[k] = (tp, fp, fn, tn)
    return arr


def metrics_from_counts(c: np.ndarray) -> dict[str, float]:
    """Micro metrics from a [n, 4] count array (tp, fp, fn, tn) plus exact-match rate."""
    tp, fp, fn, tn = (float(x) for x in c.sum(axis=0))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = float(np.mean((c[:, 1] + c[:, 2]) == 0))
    return {
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_rate": exact,
    }


def bootstrap(counts: dict[str, np.ndarray], n_conversations: int) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_conversations, size=(N_RESAMPLES, n_conversations))
    draws: dict[str, dict[str, np.ndarray]] = {}
    for system, c in counts.items():
        per_metric: dict[str, list[float]] = {}
        for b in range(N_RESAMPLES):
            m = metrics_from_counts(c[idx[b]])
            for name, value in m.items():
                per_metric.setdefault(name, []).append(value)
        draws[system] = {k: np.asarray(v) for k, v in per_metric.items()}
    return idx, draws


def interval(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)


def main() -> None:
    ids, counts = load_counts()
    n = len(ids)
    idx, draws = bootstrap(counts, n)

    systems = [s for s in SYSTEMS if s in counts]
    point = {s: metrics_from_counts(counts[s]) for s in systems}
    out = {
        "metadata": {
            "method": "conversation-level percentile bootstrap, paired across systems",
            "resamples": N_RESAMPLES,
            "seed": SEED,
            "unit": "one of the 150 locked test conversations (27 decisions each)",
            "n_conversations": n,
            "confidence": 1 - ALPHA,
            "sources": [
                "data/benchmark_1000/comparison_metrics.json",
                "data/benchmark_1000/jev_noul_test_results.json",
                "results/jev_noul_comparison_1000.json",
                "data/benchmark_1000/split.json",
            ],
            "note": "Sol and Luna rows are the committed agent-session outputs; they are reported with intervals but excluded from paired differences.",
        },
        "systems": {},
        "paired_accuracy_differences_points": {},
    }
    for s in systems:
        out["systems"][s] = {"label": SYSTEMS[s][0], "rerunnable": SYSTEMS[s][1], "metrics": {}}
        for name in ("accuracy", "precision", "recall", "f1", "exact_match_rate"):
            lo, hi = interval(draws[s][name])
            out["systems"][s]["metrics"][name] = {"point": point[s][name], "ci_low": lo, "ci_high": hi}

    rerunnable = [s for s in systems if SYSTEMS[s][1]]
    for a in rerunnable:
        for b in rerunnable:
            if a >= b:
                continue
            diff = 100 * (draws[a]["accuracy"] - draws[b]["accuracy"])
            lo, hi = interval(diff)
            out["paired_accuracy_differences_points"][f"{a} minus {b}"] = {
                "point": 100 * (point[a]["accuracy"] - point[b]["accuracy"]),
                "ci_low": lo,
                "ci_high": hi,
                "separable_at_95": bool(lo > 0 or hi < 0),
            }

    (DATA / "ci_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    write_report(out)
    print(f"wrote {DATA / 'ci_metrics.json'} and {REPORT / 'conversation-level-cis.md'}")


def fmt(m: dict[str, float]) -> str:
    return f"{100 * m['point']:.2f} [{100 * m['ci_low']:.2f}, {100 * m['ci_high']:.2f}]"


def write_report(out: dict) -> None:
    lines = [
        "# Conversation-level bootstrap confidence intervals",
        "",
        f"Percentile bootstrap over the {out['metadata']['n_conversations']} locked test conversations, "
        f"{out['metadata']['resamples']} resamples, seed {out['metadata']['seed']}. Every system is scored on the same "
        "resamples, so differences between systems are paired. Point estimates equal the published tables. "
        "Resampling conversations rather than decisions follows limitation 3 of the main report: the 27 decisions "
        "within a transcript are correlated.",
        "",
        "Sol and Luna are the committed agent-session outputs; their rows carry intervals but are excluded from the "
        "paired-difference table.",
        "",
        "| System | Accuracy % [95% CI] | Micro F1 % [95% CI] | Exact match % [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for s, row in out["systems"].items():
        m = row["metrics"]
        lines.append(f"| {row['label']} | {fmt(m['accuracy'])} | {fmt(m['f1'])} | {fmt(m['exact_match_rate'])} |")
    lines += [
        "",
        "## Paired accuracy differences between neighbours in the ranking, percentage points",
        "",
        "Re-runnable systems only, ordered by accuracy; each row compares a system with the next one down. "
        "A pair is separable at 95% when its interval excludes zero. The full pairwise matrix is in "
        "`data/benchmark_1000/ci_metrics.json`.",
        "",
        "| Higher | Lower | Difference [95% CI] | Separable |",
        "|---|---|---:|:---:|",
    ]
    ranked = sorted(
        (s for s, row in out["systems"].items() if row["rerunnable"]),
        key=lambda s: -out["systems"][s]["metrics"]["accuracy"]["point"],
    )
    diffs = out["paired_accuracy_differences_points"]
    for hi, lo in zip(ranked, ranked[1:]):
        key = f"{hi} minus {lo}" if f"{hi} minus {lo}" in diffs else f"{lo} minus {hi}"
        d = diffs[key]
        sign = 1 if key.startswith(hi) else -1
        point, a, b = sign * d["point"], sign * d["ci_low"], sign * d["ci_high"]
        a, b = min(a, b), max(a, b)
        lines.append(
            f"| {out['systems'][hi]['label']} | {out['systems'][lo]['label']} | "
            f"{point:+.2f} [{a:+.2f}, {b:+.2f}] | {'yes' if d['separable_at_95'] else 'no'} |"
        )
    lines += ["", "Generated by `scripts/conversation_bootstrap.py`."]
    (REPORT / "conversation-level-cis.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
