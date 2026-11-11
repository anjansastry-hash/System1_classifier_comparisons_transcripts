"""Human-review gate economics for the transcript benchmark (report next step 5).

Why: the report's recommended next step 5 asks for "human-review gates and expected
business cost, not just symmetric F1". This script measures, for each encoder with
committed validation and test probabilities, how many decisions must be routed to a human
reviewer to reach a target accuracy on the decisions that are kept, how many of the
errors that catches, and what the trade costs under a stated price of a wrong decision
relative to a review. Everything is fitted on the validation split and reported on the
locked test split. Nothing existing is changed.

Systems gated: GLiClass Modern Large v3, GLiClass Large v3, ModernBERT zero-shot. These
are the three with committed probabilities for both splits. ModernBERT trained heads and
DeBERTa zero-shot have test probabilities only, so a cut cannot be chosen for them without
touching the test split. The hosted system is treated in a diagnostic instead (see below).

Decision rule and gate
  * Decision: the published per-label thresholds (validation-selected, from the
    *_opt_threshold_results.json metadata). Point estimates therefore equal the published
    calibrated rows.
  * Confidence: distance from the label's threshold on the log-odds scale:
        margin = | logit(p) - logit(t) |
    (a monotone distance within each label; on validation it ranked errors better than the
    probability-scale distance for all three encoders)
  * Gate: a decision is routed to review when its margin is below a cut. Two variants:
      - global: one cut for all 27 labels;
      - per-label: a cut per label, relaxed greedily from the global cut, one label at a
        time in attribute order, while the validation selective accuracy stays at or above
        the target.
    Cuts are chosen on validation as the smallest ask rate whose selective accuracy
    (accuracy on the decisions kept) reaches the target. Targets below the system's raw
    validation accuracy are omitted: they are met with no gate.

Reported on the test split, per system and target
  ask rate, selective accuracy, errors caught (errors among routed decisions), residual
  errors (errors among kept decisions), with conversation-level paired bootstrap intervals
  (thresholds and cuts held fixed), plus a whole-pipeline bootstrap at each target in which
  validation and test conversations are resampled and the global cut is refitted inside
  every draw, so the interval also carries the uncertainty of choosing the cut, and the
  share of those draws in which the refitted cut meets the target on test.
  Also: AUROC of the margin as a predictor of correctness, ECE of the positive-class
  probability at 10, 15 and 20 bins, the validation-to-test ask-rate gap, the attributes a
  per-label gate routes more than half the time, and per-label positive and error counts.

Cost view
  expected cost per 1,000 decisions = 1,000 x (ask_rate x c_review + residual_error_rate x
  c_error), reported for ratios c_error / c_review from 1 to 1,000 with c_review = 1, for
  the no-gate baseline and every fitted gate. The ratio is a parameter the reader sets;
  no single operating point is recommended.

Hosted-system diagnostic (JEV Noul at 0.50 and validation-calibrated; JEV Choice at 0.50):
  test errors, share that are false positives, share at least 2 and 4 nats from the
  threshold, errors sitting exactly on the threshold, and the number of distinct
  probability values (the API returns 0.01 steps). These rows are left out of the gate
  table: calibrated Noul makes too few errors to fit or evaluate a gate, the 0.50-rule Noul
  is already repaired by the published calibration, and Choice has no validation
  probabilities. Sol and Luna have no probabilities.

Adjudication sample: twenty test errors routed by the ModernBERT zero-shot per-label gate
at the 99.5 percent target, drawn with a fixed seed, with the transcript span that decides
each one and whether a reader siding with the transcript sides with the gold label. A second
reader's blind first answers on the same rows are recorded beside the verdicts.

Reads (all committed): data/benchmark_1000/{split.json, *_validation_results.json,
*_results.json, *_opt_threshold_results.json, jev_noul_test_results.json, jev_results.json},
results/jev_noul_comparison_1000.json.
Writes: data/benchmark_1000/review_gate_metrics.json, report/review-gate-economics.md,
report/review-gate-adjudication-sample.md. Dependency: numpy only.

Usage, from the repository root:  python3 scripts/review_gate_economics.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmark_1000"
RESULTS = ROOT / "results"
REPORT = ROOT / "report"

SEED = 20260919
N_BOOT = 1000
TARGETS = (0.99, 0.995, 0.999)
RATIOS = (1, 3, 10, 30, 100, 300, 1000)
ECE_BINS = (10, 15, 20)
SAMPLE_FOR_ADJUDICATION = 20
ADJUDICATION_SYSTEM = "modernbert_zero_shot"
ADJUDICATION_TARGET = 0.995

SYSTEMS = {
    "gliclass_modern_large_v3": {
        "label": "GLiClass Modern Large v3",
        "validation": "gliclass_modern_large_v3_validation_results.json",
        "test": "gliclass_modern_large_v3_results.json",
        "thresholds_from": "gliclass_modern_large_v3_opt_threshold_results.json",
    },
    "gliclass_large_v3": {
        "label": "GLiClass Large v3",
        "validation": "gliclass_large_v3_validation_results.json",
        "test": "gliclass_large_v3_results.json",
        "thresholds_from": "gliclass_large_v3_opt_threshold_results.json",
    },
    "modernbert_zero_shot": {
        "label": "ModernBERT zero-shot",
        "validation": "modernbert_zero_shot_validation_results.json",
        "test": "modernbert_zero_shot_results.json",
        "thresholds_from": "modernbert_zero_shot_opt_threshold_results.json",
    },
}

# Reader verdicts for the adjudication sample, keyed by (conversation id, attribute).
# "span" is the transcript text that decides the label under the benchmark's present/absent
# criteria (BOUNDARIES in scripts/jev_noul_benchmark.py); "gold_supported" records whether
# that span supports the gold label; "note" names the criterion where it is not obvious.
# Rows not listed here are left blank in the table.
_PERF_NOT_QUALITY = "names performance; the criteria exclude inferring a service-quality complaint from it"
_QUALITY_NOT_PERF = "names service quality; the criteria exclude inferring a performance complaint from it"
_NO_PHONE = "no mobile or phone service malfunction is reported"
ADJUDICATION: dict[tuple[int, str], dict] = {
    (171, "receiving_error_messages"): {"span": "I keep receiving the message “Service unavailable.”", "gold_supported": True, "note": ""},
    (380, "poor_service_quality_complaint"): {"span": "The overall performance has been poor.", "gold_supported": True, "note": _PERF_NOT_QUALITY},
    (191, "poor_performance_complaint"): {"span": "I am unhappy with the poor service quality.", "gold_supported": True, "note": _QUALITY_NOT_PERF},
    (971, "sales_attempt_by_agent"): {"span": "I can offer a discounted premium-support package.", "gold_supported": True, "note": ""},
    (708, "poor_performance_complaint"): {"span": "I am unhappy with the poor service quality.", "gold_supported": True, "note": _QUALITY_NOT_PERF},
    (577, "poor_performance_complaint"): {"span": "I am unhappy with the poor service quality.", "gold_supported": True, "note": _QUALITY_NOT_PERF + "; glitching is its own attribute"},
    (537, "phone_service_issue"): {"span": "Video calls and online sessions keep lagging.", "gold_supported": True, "note": _NO_PHONE},
    (680, "sales_attempt_by_agent"): {"span": "I can offer a discounted premium-support package.", "gold_supported": True, "note": ""},
    (465, "poor_service_quality_complaint"): {"span": "The overall performance has been poor.", "gold_supported": True, "note": _PERF_NOT_QUALITY},
    (955, "app_issue_mentioned"): {"span": "The account app is not working correctly.", "gold_supported": True, "note": ""},
    (332, "sales_attempt_by_agent"): {"span": "I can offer a discounted premium-support package.", "gold_supported": True, "note": ""},
    (123, "phone_service_issue"): {"span": "Video calls and online sessions keep lagging.", "gold_supported": True, "note": _NO_PHONE},
    (635, "app_issue_mentioned"): {"span": "The account app is not working correctly.", "gold_supported": True, "note": ""},
    (171, "poor_service_quality_complaint"): {"span": "The overall performance has been poor.", "gold_supported": True, "note": _PERF_NOT_QUALITY},
    (591, "poor_service_quality_complaint"): {"span": "The overall performance has been poor.", "gold_supported": True, "note": _PERF_NOT_QUALITY},
    (956, "poor_service_quality_complaint"): {"span": "The overall performance has been poor.", "gold_supported": True, "note": _PERF_NOT_QUALITY + "; the phone-service line is its own attribute"},
    (758, "app_issue_mentioned"): {"span": "The account app is not working correctly.", "gold_supported": True, "note": ""},
    (798, "poor_performance_complaint"): {"span": "I am unhappy with the poor service quality.", "gold_supported": True, "note": _QUALITY_NOT_PERF},
    (659, "app_issue_mentioned"): {"span": "The account app is not working correctly.", "gold_supported": True, "note": ""},
    (425, "sales_attempt_by_agent"): {"span": "I can offer a discounted premium-support package.", "gold_supported": True, "note": ""},
}


# Blind second read of the same sample. The transcript and the criterion were read aloud to a second
# reader; the gold label, the model output and the first reader's verdict were withheld until the
# reader had answered. Value: which side the first answer matched ("key" or "model"); None marks the
# one row where the first reader's verdict was stated first. Every row was then settled as "key right".
SECOND_READ_FIRST_ANSWER: dict[tuple[int, str], str | None] = {
    (171, "receiving_error_messages"): None,
    (380, "poor_service_quality_complaint"): "model",
    (191, "poor_performance_complaint"): "key",
    (971, "sales_attempt_by_agent"): "key",
    (708, "poor_performance_complaint"): "key",
    (577, "poor_performance_complaint"): "key",
    (537, "phone_service_issue"): "model",
    (680, "sales_attempt_by_agent"): "key",
    (465, "poor_service_quality_complaint"): "model",
    (955, "app_issue_mentioned"): "key",
    (332, "sales_attempt_by_agent"): "key",
    (123, "phone_service_issue"): "key",
    (635, "app_issue_mentioned"): "model",
    (171, "poor_service_quality_complaint"): "model",
    (591, "poor_service_quality_complaint"): "model",
    (956, "poor_service_quality_complaint"): "key",
    (758, "app_issue_mentioned"): "key",
    (798, "poor_performance_complaint"): "key",
    (659, "app_issue_mentioned"): "key",
    (425, "sales_attempt_by_agent"): "key",
}
_CALLING = "criterion wording: 'calling problems' can be read to cover 'video calls keep lagging'"
SECOND_READ_NOTE: dict[tuple[int, str], str] = {
    (537, "phone_service_issue"): _CALLING,
    (123, "phone_service_issue"): _CALLING,
    (635, "app_issue_mentioned"): "the app sentence is one short line the call never returns to",
}

# ----------------------------------------------------------------------------- loading

def load_split():
    split = json.loads((DATA / "split.json").read_text())
    attributes = list(split["metadata"]["attributes"])
    rows = {part: [] for part in ("train", "validation", "test")}
    for r in split["rows"]:
        rows[r["split"]].append(r)
    for part in rows:
        rows[part].sort(key=lambda r: int(r["id"]))
    return attributes, rows


def prob_matrix(path: Path, part_rows, attributes) -> np.ndarray:
    payload = json.loads(path.read_text())["rows"]
    out = np.zeros((len(part_rows), len(attributes)))
    for i, r in enumerate(part_rows):
        cell = payload[str(r["id"])]["probabilities"]
        for j, a in enumerate(attributes):
            v = cell[a]
            out[i, j] = float(v["present"]) if isinstance(v, dict) else float(v)
    return out


def truth_matrix(part_rows, attributes) -> np.ndarray:
    return np.array([[int(r["truth"][a]) for a in attributes] for r in part_rows])


def thresholds_vector(path: Path, attributes) -> np.ndarray:
    th = json.loads(path.read_text())["metadata"]["thresholds"]
    return np.array([float(th[a]) for a in attributes])


# ----------------------------------------------------------------------------- gate math

def logit(x: np.ndarray) -> np.ndarray:
    eps = 1e-6
    return np.log((x + eps) / (1 - x + eps))


def margin(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Distance of each probability from its label's threshold, in nats."""
    return np.abs(logit(P) - logit(T)[None, :])


def fit_global_cut(m: np.ndarray, correct: np.ndarray, target: float):
    """Smallest cut (hence smallest ask rate) whose selective accuracy on kept decisions >= target.

    Kept decisions are those with margin >= cut. Only cuts equal to an observed margin are
    considered, and only at the first position of each tie group, so the kept set evaluated
    is exactly the kept set the cut produces.
    """
    order = np.argsort(m.ravel(), kind="stable")
    ms = m.ravel()[order]
    cs = correct.ravel()[order].astype(float)
    n = len(ms)
    kept_correct = np.cumsum(cs[::-1])[::-1]
    kept_n = np.arange(n, 0, -1)
    sa = kept_correct / kept_n
    boundary = np.ones(n, dtype=bool)
    boundary[1:] = ms[1:] != ms[:-1]
    ok = np.where(boundary & (sa >= target - 1e-12))[0]
    if len(ok) == 0:
        return None
    return float(ms[ok[0]])


def fit_per_label_cuts(m: np.ndarray, correct: np.ndarray, target: float, global_cut: float):
    """Greedy per-label relaxation from the global cut: lower one label's cut while SA holds."""
    n_labels = m.shape[1]
    cuts = np.full(n_labels, global_cut)
    grids = [np.unique(m[:, j]) for j in range(n_labels)]

    def sa_for(c):
        kept = m >= c[None, :]
        return correct[kept].mean() if kept.any() else 0.0

    improved = True
    while improved:
        improved = False
        for j in range(n_labels):
            while True:
                lower = grids[j][grids[j] < cuts[j]]
                if len(lower) == 0:
                    break
                trial = cuts.copy()
                trial[j] = lower[-1]
                if sa_for(trial) >= target - 1e-12:
                    cuts = trial
                    improved = True
                else:
                    break
    return cuts


def gate_metrics(m: np.ndarray, correct: np.ndarray, cuts: np.ndarray) -> dict:
    kept = m >= cuts[None, :]
    asked = ~kept
    errors = ~correct
    return {
        "ask_rate": float(asked.mean()),
        "selective_accuracy": float(correct[kept].mean()) if kept.any() else None,
        "errors_total": int(errors.sum()),
        "errors_caught": int((errors & asked).sum()),
        "errors_residual": int((errors & kept).sum()),
        "residual_error_rate": float((errors & kept).sum() / errors.size),
    }


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUROC with average ranks for ties; 'positive' = correct decision, score = margin."""
    s = score.ravel()
    y = positive.ravel().astype(bool)
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg_rank = cum - (counts - 1) / 2.0  # 1-based average rank of each tie group
    ranks = avg_rank[inverse]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auroc_bootstrap(m_t, c_t, rng, n_conv):
    idx = rng.integers(0, n_conv, size=(N_BOOT, n_conv))
    return np.array([auroc(m_t[idx[b]], c_t[idx[b]]) for b in range(N_BOOT)])


def ece(p: np.ndarray, y: np.ndarray, bins: int) -> float:
    p = p.ravel()
    y = y.ravel()
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
        if sel.any():
            total += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(total)


# ----------------------------------------------------------------------------- bootstrap

def paired_bootstrap_fixed(m_t, c_t, cuts, rng, n_conv):
    idx = rng.integers(0, n_conv, size=(N_BOOT, n_conv))
    ask, sa = [], []
    for b in range(N_BOOT):
        mm, cc = m_t[idx[b]], c_t[idx[b]]
        g = gate_metrics(mm, cc, cuts)
        ask.append(g["ask_rate"])
        sa.append(g["selective_accuracy"] if g["selective_accuracy"] is not None else np.nan)
    return np.array(ask), np.array(sa)


def pipeline_bootstrap(m_v, c_v, m_t, c_t, target, rng):
    """Resample validation and test conversations, refit the global cut, score."""
    nv, nt = m_v.shape[0], m_t.shape[0]
    ask, sa, unreach = [], [], 0
    for _ in range(N_BOOT):
        iv = rng.integers(0, nv, nv)
        it = rng.integers(0, nt, nt)
        cut = fit_global_cut(m_v[iv], c_v[iv], target)
        if cut is None:
            unreach += 1
            continue
        g = gate_metrics(m_t[it], c_t[it], np.full(m_t.shape[1], cut))
        ask.append(g["ask_rate"])
        sa.append(g["selective_accuracy"] if g["selective_accuracy"] is not None else np.nan)
    return np.array(ask), np.array(sa), unreach


def ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [None, None]
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


# ----------------------------------------------------------------------------- hosted diagnostic

def hosted_diagnostic(attributes, rows, Yt) -> list[dict]:
    noul = json.loads((DATA / "jev_noul_test_results.json").read_text())["rows"]
    th = json.loads((RESULTS / "jev_noul_comparison_1000.json").read_text())["thresholds"]
    Pn = np.array([[float(noul[str(r["id"])]["probabilities"][a]) for a in attributes] for r in rows["test"]])
    Tn = np.array([float(th[a]) for a in attributes])
    choice = json.loads((DATA / "jev_results.json").read_text())["rows"]
    Pc = np.array([[float(choice[str(r["id"])]["probabilities"][a]["present"]) for a in attributes] for r in rows["test"]])
    half = np.full(len(attributes), 0.5)
    out = []
    for name, P, T, val_probs in (
        ("JEV Noul, 0.50", Pn, half, "committed"),
        ("JEV Noul, validation-calibrated", Pn, Tn, "committed"),
        ("JEV Choice, 0.50", Pc, half, "none"),
    ):
        pred = P >= T[None, :]
        correct = pred == Yt.astype(bool)
        err = ~correct
        m = margin(P, T)
        em = m[err]
        out.append({
            "system": name,
            "test_accuracy": float(correct.mean()),
            "test_errors": int(err.sum()),
            "false_positive_share_of_errors": float((pred & err).sum() / err.sum()) if err.any() else None,
            "errors_at_least_2_nats": int((em >= 2.0).sum()),
            "errors_at_least_4_nats": int((em >= 4.0).sum()),
            "errors_on_threshold": int((P[err] == np.broadcast_to(T[None, :], P.shape)[err]).sum()),
            "distinct_probability_values_test": int(len(np.unique(P))),
            "labels_with_threshold_above_0_50": int((T > 0.5).sum()),
            "validation_probabilities": val_probs,
        })
    return out


# ----------------------------------------------------------------------------- main

def main() -> None:
    attributes, rows = load_split()
    Yv = truth_matrix(rows["validation"], attributes)
    Yt = truth_matrix(rows["test"], attributes)
    out = {
        "metadata": {
            "method": "per-label published thresholds; margin = |logit(p) - logit(t)|; global and per-label review "
                      "cuts fitted on validation to a selective-accuracy target; reported on the locked test split",
            "targets": list(TARGETS),
            "seed": SEED,
            "resamples": N_BOOT,
            "cost_ratios_c_error_over_c_review": list(RATIOS),
            "not_gated": {
                "modernbert_trained_heads": "test probabilities only; no validation probabilities committed",
                "deberta_v3_large_zeroshot": "test probabilities only",
                "jev_noul": "see hosted_diagnostic: 4 calibrated errors; 0.50-rule errors repaired by calibration",
                "jev_choice": "see hosted_diagnostic: no validation probabilities; 0.01-quantized",
                "sol_luna": "no probabilities",
            },
        },
        "systems": {},
    }
    adjudication_rows = []

    for sid, spec in SYSTEMS.items():
        T = thresholds_vector(DATA / spec["thresholds_from"], attributes)
        Pv = prob_matrix(DATA / spec["validation"], rows["validation"], attributes)
        Pt = prob_matrix(DATA / spec["test"], rows["test"], attributes)
        cv = (Pv >= T[None, :]) == Yv.astype(bool)
        ct = (Pt >= T[None, :]) == Yt.astype(bool)
        mv, mt = margin(Pv, T), margin(Pt, T)

        sysout = {
            "label": spec["label"],
            "validation_accuracy": float(cv.mean()),
            "validation_errors": int((~cv).sum()),
            "test_accuracy": float(ct.mean()),
            "test_errors": int((~ct).sum()),
            "auroc_margin_vs_correct_test": auroc(mt, ct),
            "auroc_margin_vs_correct_test_ci": ci(auroc_bootstrap(mt, ct, np.random.default_rng(SEED), mt.shape[0])),
            "test_errors_median_margin_nats": float(np.median(mt[~ct])) if (~ct).any() else None,
            "test_errors_share_at_least_2_nats": float((mt[~ct] >= 2.0).mean()) if (~ct).any() else None,
            "ece_test_by_bins": {str(b): ece(Pt, Yt, b) for b in ECE_BINS},
            "per_label": [],
            "gates": {},
            "cost_per_1000_decisions": {},
        }
        for j, a in enumerate(attributes):
            sysout["per_label"].append({
                "attribute": a,
                "threshold": float(T[j]),
                "validation_positives": int(Yv[:, j].sum()),
                "validation_errors": int((~cv[:, j]).sum()),
                "validation_false_negatives": int((~cv[:, j] & (Yv[:, j] == 1)).sum()),
                "test_positives": int(Yt[:, j].sum()),
                "test_errors": int((~ct[:, j]).sum()),
                "test_false_negatives": int((~ct[:, j] & (Yt[:, j] == 1)).sum()),
            })

        base_err_rate = float((~ct).mean())
        sysout["cost_per_1000_decisions"]["no gate"] = {str(r): 1000 * base_err_rate * r for r in RATIOS}

        for target in TARGETS:
            tkey = f"{100 * target:g}"
            if cv.mean() >= target:
                sysout["gates"][tkey] = {"status": "met on validation with no gate"}
                continue
            gcut = fit_global_cut(mv, cv, target)
            if gcut is None:
                sysout["gates"][tkey] = {"status": "unreachable on validation"}
                continue
            variants = {"global": np.full(len(attributes), gcut)}
            variants["per-label"] = fit_per_label_cuts(mv, cv, target, gcut)
            entry = {"status": "fitted", "global_cut_nats": gcut, "variants": {}}
            for vname, cuts in variants.items():
                gv = gate_metrics(mv, cv, cuts)
                gt = gate_metrics(mt, ct, cuts)
                ask_b, sa_b = paired_bootstrap_fixed(mt, ct, cuts, np.random.default_rng(SEED), mt.shape[0])
                rec = {
                    "validation": gv,
                    "test": gt,
                    "test_ci_fixed_cuts": {"ask_rate": ci(ask_b), "selective_accuracy": ci(sa_b)},
                    "validation_to_test_ask_gap_points": 100 * (gt["ask_rate"] - gv["ask_rate"]),
                }
                if vname == "per-label":
                    rec["cuts_nats"] = {a: float(c) for a, c in zip(attributes, cuts)}
                    rec["distinct_cuts"] = int(len(np.unique(np.round(cuts, 9))))
                    per_label_ask = (mt < cuts[None, :]).mean(axis=0)
                    rec["test_ask_rate_by_label"] = {a: float(x) for a, x in zip(attributes, per_label_ask)}
                    rec["labels_routed_over_half"] = [a for a, x in zip(attributes, per_label_ask) if x > 0.5]
                entry["variants"][vname] = rec
                sysout["cost_per_1000_decisions"][f"{tkey}% {vname}"] = {
                    str(r): 1000 * (gt["ask_rate"] * 1.0 + gt["residual_error_rate"] * r) for r in RATIOS
                }
            ask_p, sa_p, unreach = pipeline_bootstrap(mv, cv, mt, ct, target, np.random.default_rng(SEED + 1))
            entry["pipeline_bootstrap_global"] = {
                "ask_rate_ci": ci(ask_p),
                "selective_accuracy_ci": ci(sa_p),
                "share_of_draws_meeting_target_on_test": float(np.mean(sa_p[~np.isnan(sa_p)] >= target)) if len(sa_p) else None,
                "draws_where_cut_unreachable": unreach,
            }
            sysout["gates"][tkey] = entry

            if target == ADJUDICATION_TARGET and sid == ADJUDICATION_SYSTEM:
                cuts = variants["per-label"]
                asked = mt < cuts[None, :]
                cand = np.argwhere(asked & ~ct)
                pick = cand[np.random.default_rng(SEED + 2).permutation(len(cand))[:SAMPLE_FOR_ADJUDICATION]]
                for (i, j) in pick:
                    r = rows["test"][i]
                    adjudication_rows.append({
                        "system": spec["label"], "conversation_id": int(r["id"]),
                        "attribute": attributes[j], "gold": int(Yt[i, j]),
                        "model_p": float(Pt[i, j]), "threshold": float(T[j]), "margin": float(mt[i, j]),
                        "transcript": r["conversation"],
                    })

        out["systems"][sid] = sysout

    out["hosted_diagnostic"] = hosted_diagnostic(attributes, rows, Yt)
    out["adjudication_sample"] = [
        {
            **{k: v for k, v in r.items() if k != "transcript"},
            "gold_supported": ADJUDICATION.get((r["conversation_id"], r["attribute"]), {}).get("gold_supported"),
            "span": ADJUDICATION.get((r["conversation_id"], r["attribute"]), {}).get("span"),
            "second_reader_blind_first_answer_matched": SECOND_READ_FIRST_ANSWER.get((r["conversation_id"], r["attribute"])),
            "second_reader_final": "key right" if (r["conversation_id"], r["attribute"]) in SECOND_READ_FIRST_ANSWER else None,
            "second_reader_note": SECOND_READ_NOTE.get((r["conversation_id"], r["attribute"])),
        }
        for r in adjudication_rows
    ]

    (DATA / "review_gate_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    write_report(out, attributes)
    write_adjudication(adjudication_rows)
    print("wrote review_gate_metrics.json, review-gate-economics.md, review-gate-adjudication-sample.md")


# ----------------------------------------------------------------------------- report

def pct(x, d=2):
    return "n/a" if x is None else f"{100 * x:.{d}f}"


def ci_pct(c, d=2):
    return "n/a" if c is None or c[0] is None else f"[{100 * c[0]:.{d}f}, {100 * c[1]:.{d}f}]"


def summary_lines(out: dict) -> list[str]:
    """Three data-driven sentences for the top of the report."""
    S = out["systems"]
    zs = S["modernbert_zero_shot"]
    gm, gl = S["gliclass_modern_large_v3"], S["gliclass_large_v3"]
    z995 = zs["gates"]["99.5"]["variants"]
    zg, zp = z995["global"]["test"], z995["per-label"]["test"]
    gm_pl = gm["gates"]["99.5"]["variants"]["per-label"]
    gl_pl = gl["gates"]["99.5"]["variants"]["per-label"]
    gm_g = gm["gates"]["99.5"]["variants"]["global"]["test"]
    gl_g = gl["gates"]["99.5"]["variants"]["global"]["test"]
    hnc = out["hosted_diagnostic"][1]

    def top_routed(rec, k=3):
        items = sorted(rec["test_ask_rate_by_label"].items(), key=lambda kv: -kv[1])[:k]
        return ", ".join(f"`{a}` {100 * x:.0f}" for a, x in items)

    def top_error_share(s, k=3):
        pl = sorted(s["per_label"], key=lambda x: -x["test_errors"])[:k]
        return sum(p["test_errors"] for p in pl) / s["test_errors"]

    return [
        "## Summary",
        "",
        f"A confidence gate works for {zs['label']}: its margin ranks errors well (AUROC {zs['auroc_margin_vs_correct_test']:.2f}), "
        f"so at the 99.5 percent target the global gate routes {pct(zg['ask_rate'], 1)} percent of decisions and catches "
        f"{zg['errors_caught']} of {zg['errors_total']} test errors, and the per-label gate routes {pct(zp['ask_rate'], 1)} percent "
        f"and catches {zp['errors_caught']}. The two GLiClass encoders make fewer errors but confident ones (AUROC "
        f"{gm['auroc_margin_vs_correct_test']:.2f} and {gl['auroc_margin_vs_correct_test']:.2f}; {pct(gm['test_errors_share_at_least_2_nats'], 0)} and "
        f"{pct(gl['test_errors_share_at_least_2_nats'], 0)} percent of their errors sit at least 2 nats from the threshold), so their "
        f"global gates reach 99.5 percent only by routing {pct(gm_g['ask_rate'], 0)} and {pct(gl_g['ask_rate'], 0)} percent of decisions. "
        f"Their errors are concentrated: {pct(top_error_share(gm), 0)} and {pct(top_error_share(gl), 0)} percent of them fall in three "
        f"attributes each, so their per-label gates reach the target by routing a few attributes heavily (percent of each attribute's "
        f"decisions routed, {gm['label']}: {top_routed(gm_pl)}; {gl['label']}: {top_routed(gl_pl)}), which is a staffing decision "
        "about those attributes rather than a confidence gate. "
        f"The hosted system makes {hnc['test_errors']} calibrated test errors, {hnc['errors_on_threshold']} of them exactly on its "
        "decision threshold: too few to gate, and its lever is calibration. "
        "Whether a gate pays depends on the price of a wrong decision relative to a review, which is left as a parameter.",
        "",
    ]


def worked_example(out: dict) -> list[str]:
    """One worked cost line for ModernBERT zero-shot at a ratio of 30, from the computed table."""
    s = out["systems"]["modernbert_zero_shot"]
    g = s["gates"]["99.5"]["variants"]["per-label"]["test"]
    ratio = 30
    base = s["cost_per_1000_decisions"]["no gate"][str(ratio)]
    gated = s["cost_per_1000_decisions"]["99.5% per-label"][str(ratio)]
    review_part = 1000 * g["ask_rate"]
    error_part = 1000 * g["residual_error_rate"] * ratio
    # break-even ratio: where ask_rate = (base_err - residual_err) * ratio
    base_err = s["test_errors"] / 4050
    be = g["ask_rate"] / (base_err - g["residual_error_rate"]) if base_err > g["residual_error_rate"] else float("inf")
    return [
        f"Worked example, {s['label']}, ratio {ratio} (one wrong decision costs as much as {ratio} reviews): with no gate, "
        f"{s['test_errors']} errors in 4,050 decisions cost {base:.1f} per 1,000 decisions. The 99.5 percent per-label gate "
        f"routes {pct(g['ask_rate'])} percent of decisions ({review_part:.1f} reviews per 1,000) and leaves "
        f"{g['errors_residual']} errors ({error_part:.1f} in error cost), total {gated:.1f}. The gate pays for itself once a "
        f"wrong decision costs more than about {be:.0f} reviews; below that, no gate is cheaper.",
    ]


def write_report(out: dict, attributes) -> None:
    L = [
        "# Human-review gate economics",
        "",
        *summary_lines(out),
        "## Method",
        "",
        "Answers the main report's recommended next step 5: human-review gates and expected business cost, "
        "not symmetric F1 alone. A review gate routes a decision to a human when the model's confidence in that "
        "decision is low; the rest are kept. Confidence is the distance of the probability from the label's "
        "published threshold on the log-odds scale, so point estimates equal the published calibrated rows. Cuts "
        "are chosen on the validation split for a target accuracy on kept decisions (selective accuracy) and "
        "reported on the locked test split with conversation-level bootstrap intervals (1,000 resamples, seed "
        "20260919).",
        "",
        "Gated: the three encoders with committed probabilities for both splits (GLiClass Modern Large v3, GLiClass "
        "Large v3, ModernBERT zero-shot). ModernBERT trained heads and DeBERTa have test probabilities only, so no "
        "cut can be chosen for them without touching the test split. The hosted system is covered by the diagnostic "
        "at the end. Targets below a system's raw validation accuracy are omitted: they are met with no gate.",
        "",
        "## Gate table (test split)",
        "",
        "Two variants per target: one cut for all labels (global) and one cut per label (per-label, relaxed from the "
        "global cut on validation). Intervals hold the cuts fixed. The pipeline columns refit the global cut inside "
        "every resample of validation and score it on a resample of test; the last pipeline column is the share of "
        "those draws in which the refitted cut still meets the target on test.",
        "",
        "| System | Target % | Variant | Ask rate % [95% CI] | Selective acc. % [95% CI] | Errors caught / residual / total | "
        "Val→test ask gap (pts) | Pipeline ask % [95% CI] | Pipeline SA % [95% CI] | Pipeline draws meeting target % | "
        "Labels routed >50% |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    routed = []
    for sid, s in out["systems"].items():
        for tkey, g in s["gates"].items():
            if g.get("status") != "fitted":
                L.append(f"| {s['label']} | {tkey} | — | {g['status']} | | | | | | | |")
                continue
            pb = g["pipeline_bootstrap_global"]
            for vname, rec in g["variants"].items():
                t = rec["test"]
                is_global = vname == "global"
                pipe_ask = ci_pct(pb["ask_rate_ci"]) if is_global else ""
                pipe_sa = ci_pct(pb["selective_accuracy_ci"]) if is_global else ""
                pipe_share = pct(pb["share_of_draws_meeting_target_on_test"], 0) if is_global else ""
                n_routed = len(rec.get("labels_routed_over_half", [])) if not is_global else ""
                if not is_global and rec["labels_routed_over_half"]:
                    routed.append((s["label"], tkey, rec["labels_routed_over_half"]))
                L.append(
                    f"| {s['label']} | {tkey} | {vname} | {pct(t['ask_rate'])} {ci_pct(rec['test_ci_fixed_cuts']['ask_rate'])} | "
                    f"{pct(t['selective_accuracy'])} {ci_pct(rec['test_ci_fixed_cuts']['selective_accuracy'])} | "
                    f"{t['errors_caught']} / {t['errors_residual']} / {t['errors_total']} | {rec['validation_to_test_ask_gap_points']:+.2f} | "
                    f"{pipe_ask} | {pipe_sa} | {pipe_share} | {n_routed} |"
                )
    L += [
        "",
        "## Reading the table",
        "",
        "Validation holds 150 conversations, so a target such as 99.9 percent is fitted on a handful of errors; the "
        "pipeline columns show how much the ask rate moves when the cut itself is refitted on a resampled validation "
        "set, and how often the refitted cut still delivers the target on test. A gate whose per-label variant routes "
        "whole attributes is not a confidence gate in practice; it is a decision to hand those attributes to people.",
        "",
    ]
    if routed:
        L += ["Attributes routed more than half the time by a per-label gate:", ""]
        for label, tkey, names in routed:
            L.append(f"- {label}, {tkey} percent: " + ", ".join(f"`{n}`" for n in names))
        L.append("")
    L += [
        "## Ranking quality and calibration (test split)",
        "",
        "| System | Raw accuracy % | Test errors | AUROC of margin vs correctness [95% CI] | Median error margin (nats) | "
        "Share of errors ≥ 2 nats from threshold | ECE at 10 / 15 / 20 bins |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for sid, s in out["systems"].items():
        e = s["ece_test_by_bins"]
        a_ci = s["auroc_margin_vs_correct_test_ci"]
        L.append(
            f"| {s['label']} | {pct(s['test_accuracy'])} | {s['test_errors']} | {s['auroc_margin_vs_correct_test']:.3f} "
            f"[{a_ci[0]:.3f}, {a_ci[1]:.3f}] | {s['test_errors_median_margin_nats']:.2f} | "
            f"{pct(s['test_errors_share_at_least_2_nats'], 0)} | {e['10']:.3f} / {e['15']:.3f} / {e['20']:.3f} |"
        )
    L += [
        "",
        "AUROC here is the probability that a randomly chosen correct decision has a larger margin than a randomly "
        "chosen wrong one; 0.5 means the margin carries no information about correctness. A confidence gate can only "
        "catch errors the model is unsure about. Where most errors sit far from the threshold, the global gate reaches "
        "the target only by routing most decisions, and the per-label gate reaches it by routing whole attributes.",
        "",
        "## Expected cost per 1,000 decisions",
        "",
        "cost = 1,000 × (ask rate × c_review + residual error rate × c_error), with c_review = 1 and the ratio "
        "c_error / c_review a parameter the reader sets. The cheapest setting in each column is in bold. No single "
        "ratio is recommended; the cheapest row moves with it, which is the point.",
        "",
        *worked_example(out),
        "",
    ]
    for sid, s in out["systems"].items():
        L += [f"### {s['label']}", "", "| Setting | " + " | ".join(f"ratio {r}" for r in RATIOS) + " |",
              "|---|" + "---:|" * len(RATIOS)]
        costs = s["cost_per_1000_decisions"]
        mins = {str(r): min(v[str(r)] for v in costs.values()) for r in RATIOS}
        for setting, v in costs.items():
            cells = []
            for r in RATIOS:
                val = v[str(r)]
                cells.append(f"**{val:.1f}**" if abs(val - mins[str(r)]) < 1e-9 else f"{val:.1f}")
            L.append(f"| {setting} | " + " | ".join(cells) + " |")
        L.append("")
    L += [
        "## Per-label support on validation",
        "",
        "Threshold, positive count and errors per label on the 150 validation conversations, then positives and errors on "
        "the 150 test conversations; thresholds with few positives or few errors behind them are the least stable.",
        "",
        "| Attribute | " + " | ".join(f"{s['label']}: thr / val pos / val err / test pos / test err" for s in out["systems"].values()) + " |",
        "|---|" + "---|" * len(out["systems"]),
    ]
    for k, a in enumerate(attributes):
        cells = []
        for s in out["systems"].values():
            pl = s["per_label"][k]
            cells.append(
                f"{pl['threshold']:.2f} / {pl['validation_positives']} / {pl['validation_errors']} / "
                f"{pl['test_positives']} / {pl['test_errors']}"
            )
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    L += ["", "## Where the errors are", ""]
    for s in out["systems"].values():
        top = sorted(s["per_label"], key=lambda x: -x["test_errors"])[:3]
        share = sum(p["test_errors"] for p in top) / s["test_errors"]
        parts = []
        for p in top:
            note = ""
            if p["test_false_negatives"] == p["test_positives"] == p["test_errors"] and p["validation_false_negatives"] == p["validation_positives"] == p["validation_errors"]:
                note = " (every positive missed on both splits)"
            elif p["test_false_negatives"] == p["test_errors"]:
                note = " (all misses)"
            elif p["test_false_negatives"] == 0:
                note = " (all false positives)"
            parts.append(f"`{p['attribute']}` {p['test_errors']}{note}")
        L.append(f"- {s['label']}: {pct(share, 0)} percent of {s['test_errors']} test errors fall in three attributes: " + "; ".join(parts) + ".")
    L += [
        "",
        "An attribute whose positives are all missed, or whose errors are all confident false positives, is a problem of "
        "label wording or model coverage rather than of confidence; no gate recovers it, and the per-label gate's answer "
        "is to hand that attribute to people until the wording or the model changes.",
    ]
    L += [
        "",
        "## Why the hosted system is not in the gate table",
        "",
        "| System | Accuracy % | Test errors | False positives among errors | Errors ≥ 2 nats | Errors ≥ 4 nats | "
        "Errors exactly on threshold | Distinct probability values | Validation probabilities |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for h in out["hosted_diagnostic"]:
        L.append(
            f"| {h['system']} | {pct(h['test_accuracy'])} | {h['test_errors']} | {pct(h['false_positive_share_of_errors'], 0)} | "
            f"{h['errors_at_least_2_nats']} | {h['errors_at_least_4_nats']} | {h['errors_on_threshold']} | "
            f"{h['distinct_probability_values_test']} | {h['validation_probabilities']} |"
        )
    hn0, hnc, hc = out["hosted_diagnostic"]
    fp_phrase = ("every one a false positive" if hn0["false_positive_share_of_errors"] == 1.0
                 else f"{pct(hn0['false_positive_share_of_errors'], 0)} percent of them false positives")
    L += [
        "",
        f"At the 0.50 rule the hosted system makes {hn0['test_errors']} test errors, {fp_phrase}, with "
        f"{hn0['errors_at_least_2_nats']} at least 2 nats (about 7:1 odds) from the threshold. The published "
        f"validation calibration raises the threshold on {hnc['labels_with_threshold_above_0_50']} of the 27 labels "
        f"and removes all but {hnc['test_errors']} of them; "
        f"{hnc['errors_on_threshold']} of those {hnc['test_errors']} sit exactly on the threshold. Calibration, not a "
        "review gate, is this system's lever, and a gate fitted and evaluated on four errors would say nothing. "
        f"The API returns probabilities in 0.01 steps ({hnc['distinct_probability_values_test']} distinct values on "
        f"test), which also limits where a cut can sit. JEV Choice makes {hc['test_errors']} errors with "
        f"{hc['errors_at_least_2_nats']} at least 2 nats from 0.50 and no committed validation probabilities, so a cut "
        "could not be chosen for it without touching the test split.",
        "",
        "A reader-adjudicated sample of errors the gate catches is in `report/review-gate-adjudication-sample.md`.",
        "",
        "Generated by `scripts/review_gate_economics.py`.",
    ]
    (REPORT / "review-gate-economics.md").write_text("\n".join(L) + "\n")


def write_adjudication(rows) -> None:
    n = len(rows)
    verdicts = [ADJUDICATION.get((r["conversation_id"], r["attribute"])) for r in rows]
    n_read = sum(v is not None for v in verdicts)
    n_gold = sum(1 for v in verdicts if v is not None and v["gold_supported"])
    n_boundary = sum(1 for v in verdicts if v is not None and v["note"])
    n_miss_positive = sum(1 for r, v in zip(rows, verdicts) if v is not None and r["gold"] == 1)
    L = [
        "# Review-gate adjudication sample",
        "",
        f"{n} test errors routed to review by the per-label gate (ModernBERT zero-shot, target 99.5 percent), drawn at "
        "random with a fixed seed. Purpose: a routed error only counts as caught if a human reviewer would side with "
        "the gold label; if the gold label were wrong, the gate would be sending reviewers to fix the wrong thing. "
        "The span column quotes the transcript text that decides the attribute under the benchmark's own present/absent "
        "criteria (`BOUNDARIES` in `scripts/jev_noul_benchmark.py`), so any reader can re-check the verdict.",
        "",
    ]
    if n_read:
        L += [
            f"Verdicts: in {n_gold} of {n_read} rows read, the quoted span supports the gold label. The sample splits "
            f"into two kinds of error. {n_miss_positive} rows are missed positives: an explicit phrase (an error message, "
            "an app fault, an upsell offer) that the model scored below the label's threshold; a reviewer fixes these "
            f"from the transcript alone. {n_boundary} rows are false positives on a definitional boundary: the criteria "
            "separate a performance complaint from a service-quality complaint, and a phone-service malfunction from "
            "other symptoms, and forbid inferring one from the other. A reviewer who is not given those criteria could "
            "reasonably side with the model on these rows, so the gate's value for them depends on the review "
            "instructions carrying the criteria, not on the transcript alone.",
            "",
        ]
    keys = [(r["conversation_id"], r["attribute"]) for r in rows]
    blind = [k for k in keys if SECOND_READ_FIRST_ANSWER.get(k) is not None]
    if blind:
        b_key = sum(SECOND_READ_FIRST_ANSWER[k] == "key" for k in blind)
        b_model = len(blind) - b_key
        boundary = [k for k, v in zip(keys, verdicts) if v is not None and v["note"] and k in blind]
        bd_model = sum(SECOND_READ_FIRST_ANSWER[k] == "model" for k in boundary)
        settled = sum(k in SECOND_READ_FIRST_ANSWER for k in keys)
        L += [
            f"Blind second read. A second reader then went through the same {n} rows by ear: the transcript and the "
            "criterion were read aloud, and the gold label, the model output and the first verdict were withheld until "
            f"the reader had answered ({len(blind)} rows; on the remaining row the first verdict was stated first). The "
            f"first answer matched the gold label on {b_key} of {len(blind)} rows and matched the model on {b_model}. "
            f"On the {len(boundary)} definitional-boundary rows the first answer sided with the model on {bd_model}, "
            "each time reasoning from the overall state of the call rather than from the statement the criterion asks "
            "for; once the criterion was applied statement by statement, the reader accepted the gold label on every "
            f"row ({settled} of {n} settled as gold supported). Two notes for the criteria themselves: the phone-service "
            "criterion lists \"calling problems\", which can be read to cover \"video calls keep lagging\"; and a "
            "one-line app complaint that the call never returns to was missed by ear, as it was by the model.",
            "",
        ]
    L += [
        "| # | Conversation | Attribute | Gold | Model p | Threshold | Margin (nats) | Deciding span | Gold supported | Second reader, blind first answer | Note |",
        "|---:|---:|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for k, (r, v) in enumerate(zip(rows, verdicts), 1):
        span = ""
        if v:
            span = v["span"] + (f" ({v['note']})" if v["note"] else "")
        gs = ("yes" if v["gold_supported"] else "no") if v else ""
        key = (r["conversation_id"], r["attribute"])
        fa = SECOND_READ_FIRST_ANSWER.get(key, "")
        first = {"key": "matched gold", "model": "matched model", None: "not blind", "": ""}[fa]
        L.append(
            f"| {k} | {r['conversation_id']} | {r['attribute']} | {r['gold']} | {r['model_p']:.3f} | {r['threshold']:.2f} | "
            f"{r['margin']:.3f} | {span} | {gs} | {first} | {SECOND_READ_NOTE.get(key, '')} |"
        )
    L += ["", "## Transcripts", ""]
    for k, r in enumerate(rows, 1):
        L += [f"### {k}. Conversation {r['conversation_id']} — {r['attribute']} (gold {r['gold']}, model p {r['model_p']:.3f})", "",
              "```", r["transcript"].strip(), "```", ""]
    L += ["Generated by `scripts/review_gate_economics.py`."]
    (REPORT / "review-gate-adjudication-sample.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
