# Human-review gate economics

## Summary

A confidence gate works for ModernBERT zero-shot: its margin ranks errors well (AUROC 0.94), so at the 99.5 percent target the global gate routes 17.9 percent of decisions and catches 133 of 145 test errors, and the per-label gate routes 7.7 percent and catches 128. The two GLiClass encoders make fewer errors but confident ones (AUROC 0.62 and 0.70; 56 and 70 percent of their errors sit at least 2 nats from the threshold), so their global gates reach 99.5 percent only by routing 88 and 93 percent of decisions. Their errors are concentrated: 85 and 80 percent of them fall in three attributes each, so their per-label gates reach the target by routing a few attributes heavily (percent of each attribute's decisions routed, GLiClass Modern Large v3: `agent_attempted_resolution` 63, `poor_performance_complaint` 25, `sales_attempt_by_agent` 23; GLiClass Large v3: `poor_performance_complaint` 72, `multiple_contacts_same_issue` 45, `sales_attempt_by_agent` 25), which is a staffing decision about those attributes rather than a confidence gate. The hosted system makes 4 calibrated test errors, 3 of them exactly on its decision threshold: too few to gate, and its lever is calibration. Whether a gate pays depends on the price of a wrong decision relative to a review, which is left as a parameter.

## Method

Answers the main report's recommended next step 5: human-review gates and expected business cost, not symmetric F1 alone. A review gate routes a decision to a human when the model's confidence in that decision is low; the rest are kept. Confidence is the distance of the probability from the label's published threshold on the log-odds scale, so point estimates equal the published calibrated rows. Cuts are chosen on the validation split for a target accuracy on kept decisions (selective accuracy) and reported on the locked test split with conversation-level bootstrap intervals (1,000 resamples, seed 20260919).

Gated: the three encoders with committed probabilities for both splits (GLiClass Modern Large v3, GLiClass Large v3, ModernBERT zero-shot). ModernBERT trained heads and DeBERTa have test probabilities only, so no cut can be chosen for them without touching the test split. The hosted system is covered by the diagnostic at the end. Targets below a system's raw validation accuracy are omitted: they are met with no gate.

## Gate table (test split)

Two variants per target: one cut for all labels (global) and one cut per label (per-label, relaxed from the global cut on validation). Intervals hold the cuts fixed. The pipeline columns refit the global cut inside every resample of validation and score it on a resample of test; the last pipeline column is the share of those draws in which the refitted cut still meets the target on test.

| System | Target % | Variant | Ask rate % [95% CI] | Selective acc. % [95% CI] | Errors caught / residual / total | Val→test ask gap (pts) | Pipeline ask % [95% CI] | Pipeline SA % [95% CI] | Pipeline draws meeting target % | Labels routed >50% |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GLiClass Modern Large v3 | 99 | global | 87.68 [86.72, 88.49] | 99.40 [98.71, 100.00] | 82 / 3 / 85 | -0.54 | [17.08, 88.79] | [98.14, 100.00] | 70 |  |
| GLiClass Modern Large v3 | 99 | per-label | 3.78 [3.36, 4.22] | 99.10 [98.80, 99.38] | 50 / 35 / 85 | +0.44 |  |  |  | 1 |
| GLiClass Modern Large v3 | 99.5 | global | 88.22 [87.33, 88.99] | 99.37 [98.66, 100.00] | 82 / 3 / 85 | -0.62 | [86.39, 89.90] | [98.58, 100.00] | 46 |  |
| GLiClass Modern Large v3 | 99.5 | per-label | 4.67 [4.17, 5.16] | 99.53 [99.33, 99.72] | 67 / 18 / 85 | +0.30 |  |  |  | 1 |
| GLiClass Modern Large v3 | 99.9 | global | 89.70 [88.91, 90.42] | 100.00 [100.00, 100.00] | 85 / 0 / 85 | -1.11 | [87.68, 90.42] | [98.72, 100.00] | 66 |  |
| GLiClass Modern Large v3 | 99.9 | per-label | 5.73 [5.19, 6.27] | 99.79 [99.66, 99.92] | 77 / 8 / 85 | -0.05 |  |  |  | 1 |
| GLiClass Large v3 | 99 | global | 92.69 [92.02, 93.28] | 98.99 [97.63, 100.00] | 135 / 3 / 138 | -0.35 | [90.99, 96.12] | [96.88, 100.00] | 46 |  |
| GLiClass Large v3 | 99 | per-label | 5.38 [4.89, 5.88] | 99.11 [98.83, 99.40] | 104 / 34 / 138 | +0.17 |  |  |  | 1 |
| GLiClass Large v3 | 99.5 | global | 92.96 [92.30, 93.58] | 99.30 [98.22, 100.00] | 136 / 2 / 138 | -0.35 | [91.56, 96.40] | [97.82, 100.00] | 46 |  |
| GLiClass Large v3 | 99.5 | per-label | 6.69 [6.10, 7.28] | 99.60 [99.39, 99.79] | 123 / 15 / 138 | +0.62 |  |  |  | 1 |
| GLiClass Large v3 | 99.9 | global | 96.07 [95.53, 96.54] | 100.00 [100.00, 100.00] | 138 / 0 / 138 | -0.37 | [92.27, 96.49] | [98.29, 100.00] | 67 |  |
| GLiClass Large v3 | 99.9 | per-label | 8.12 [7.46, 8.79] | 99.84 [99.70, 99.95] | 132 / 6 / 138 | +0.44 |  |  |  | 1 |
| ModernBERT zero-shot | 99 | global | 10.42 [9.55, 11.31] | 98.84 [98.46, 99.21] | 103 / 42 / 145 | -0.02 | [8.32, 13.93] | [98.45, 99.35] | 35 |  |
| ModernBERT zero-shot | 99 | per-label | 6.12 [5.46, 6.74] | 98.87 [98.50, 99.24] | 102 / 43 / 145 | +0.40 |  |  |  | 0 |
| ModernBERT zero-shot | 99.5 | global | 17.90 [16.74, 19.09] | 99.64 [99.43, 99.82] | 133 / 12 / 145 | -0.20 | [12.81, 22.69] | [99.09, 99.91] | 72 |  |
| ModernBERT zero-shot | 99.5 | per-label | 7.65 [6.89, 8.40] | 99.55 [99.33, 99.76] | 128 / 17 / 145 | +0.30 |  |  |  | 0 |
| ModernBERT zero-shot | 99.9 | global | 26.84 [25.36, 28.22] | 99.97 [99.90, 100.00] | 144 / 1 / 145 | -0.54 | [22.79, 32.64] | [99.83, 100.00] | 88 |  |
| ModernBERT zero-shot | 99.9 | per-label | 10.37 [9.51, 11.16] | 99.83 [99.70, 99.95] | 139 / 6 / 145 | -0.20 |  |  |  | 1 |

## Reading the table

Validation holds 150 conversations, so a target such as 99.9 percent is fitted on a handful of errors; the pipeline columns show how much the ask rate moves when the cut itself is refitted on a resampled validation set, and how often the refitted cut still delivers the target on test. A gate whose per-label variant routes whole attributes is not a confidence gate in practice; it is a decision to hand those attributes to people.

Attributes routed more than half the time by a per-label gate:

- GLiClass Modern Large v3, 99 percent: `agent_attempted_resolution`
- GLiClass Modern Large v3, 99.5 percent: `agent_attempted_resolution`
- GLiClass Modern Large v3, 99.9 percent: `agent_attempted_resolution`
- GLiClass Large v3, 99 percent: `poor_performance_complaint`
- GLiClass Large v3, 99.5 percent: `poor_performance_complaint`
- GLiClass Large v3, 99.9 percent: `poor_performance_complaint`
- ModernBERT zero-shot, 99.9 percent: `app_issue_mentioned`

## Ranking quality and calibration (test split)

| System | Raw accuracy % | Test errors | AUROC of margin vs correctness [95% CI] | Median error margin (nats) | Share of errors ≥ 2 nats from threshold | ECE at 10 / 15 / 20 bins |
|---|---:|---:|---:|---:|---:|---|
| GLiClass Modern Large v3 | 97.90 | 85 | 0.623 [0.552, 0.701] | 2.40 | 56 | 0.048 / 0.053 / 0.056 |
| GLiClass Large v3 | 96.59 | 138 | 0.704 [0.653, 0.757] | 3.70 | 70 | 0.064 / 0.065 / 0.065 |
| ModernBERT zero-shot | 96.42 | 145 | 0.943 [0.931, 0.954] | 1.24 | 25 | 0.044 / 0.044 / 0.045 |

AUROC here is the probability that a randomly chosen correct decision has a larger margin than a randomly chosen wrong one; 0.5 means the margin carries no information about correctness. A confidence gate can only catch errors the model is unsure about. Where most errors sit far from the threshold, the global gate reaches the target only by routing most decisions, and the per-label gate reaches it by routing whole attributes.

## Expected cost per 1,000 decisions

cost = 1,000 × (ask rate × c_review + residual error rate × c_error), with c_review = 1 and the ratio c_error / c_review a parameter the reader sets. The cheapest setting in each column is in bold. No single ratio is recommended; the cheapest row moves with it, which is the point.

Worked example, ModernBERT zero-shot, ratio 30 (one wrong decision costs as much as 30 reviews): with no gate, 145 errors in 4,050 decisions cost 1074.1 per 1,000 decisions. The 99.5 percent per-label gate routes 7.65 percent of decisions (76.5 reviews per 1,000) and leaves 17 errors (125.9 in error cost), total 202.5. The gate pays for itself once a wrong decision costs more than about 2 reviews; below that, no gate is cheaper.

### GLiClass Modern Large v3

| Setting | ratio 1 | ratio 3 | ratio 10 | ratio 30 | ratio 100 | ratio 300 | ratio 1000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| no gate | **21.0** | 63.0 | 209.9 | 629.6 | 2098.8 | 6296.3 | 20987.7 |
| 99% global | 877.5 | 879.0 | 884.2 | 899.0 | 950.9 | 1099.0 | 1617.5 |
| 99% per-label | 46.4 | 63.7 | 124.2 | 297.0 | 902.0 | 2630.4 | 8679.8 |
| 99.5% global | 883.0 | 884.4 | 889.6 | 904.4 | 956.3 | 1104.4 | 1623.0 |
| 99.5% per-label | 51.1 | **60.0** | 91.1 | 180.0 | 491.1 | 1380.0 | 4491.1 |
| 99.9% global | 897.0 | 897.0 | 897.0 | 897.0 | 897.0 | 897.0 | **897.0** |
| 99.9% per-label | 59.3 | 63.2 | **77.0** | **116.5** | **254.8** | **649.9** | 2032.6 |

### GLiClass Large v3

| Setting | ratio 1 | ratio 3 | ratio 10 | ratio 30 | ratio 100 | ratio 300 | ratio 1000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| no gate | **34.1** | 102.2 | 340.7 | 1022.2 | 3407.4 | 10222.2 | 34074.1 |
| 99% global | 927.7 | 929.1 | 934.3 | 949.1 | 1001.0 | 1149.1 | 1667.7 |
| 99% per-label | 62.2 | 79.0 | 137.8 | 305.7 | 893.3 | 2572.3 | 8448.9 |
| 99.5% global | 930.1 | 931.1 | 934.6 | 944.4 | 979.0 | 1077.8 | 1423.5 |
| 99.5% per-label | 70.6 | **78.0** | 104.0 | 178.0 | 437.3 | 1178.0 | 3770.6 |
| 99.9% global | 960.7 | 960.7 | 960.7 | 960.7 | 960.7 | 960.7 | **960.7** |
| 99.9% per-label | 82.7 | 85.7 | **96.0** | **125.7** | **229.4** | **525.7** | 1562.7 |

### ModernBERT zero-shot

| Setting | ratio 1 | ratio 3 | ratio 10 | ratio 30 | ratio 100 | ratio 300 | ratio 1000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| no gate | **35.8** | 107.4 | 358.0 | 1074.1 | 3580.2 | 10740.7 | 35802.5 |
| 99% global | 114.6 | 135.3 | 207.9 | 415.3 | 1141.2 | 3215.3 | 10474.6 |
| 99% per-label | 71.9 | 93.1 | 167.4 | 379.8 | 1123.0 | 3246.4 | 10678.5 |
| 99.5% global | 182.0 | 187.9 | 208.6 | 267.9 | 475.3 | 1067.9 | 3142.0 |
| 99.5% per-label | 80.7 | **89.1** | **118.5** | 202.5 | 496.3 | 1335.8 | 4274.1 |
| 99.9% global | 268.6 | 269.1 | 270.9 | 275.8 | 293.1 | **342.5** | **515.3** |
| 99.9% per-label | 105.2 | 108.1 | **118.5** | **148.1** | **251.9** | 548.1 | 1585.2 |

## Per-label support on validation

Threshold, positive count and errors per label on the 150 validation conversations, then positives and errors on the 150 test conversations; thresholds with few positives or few errors behind them are the least stable.

| Attribute | GLiClass Modern Large v3: thr / val pos / val err / test pos / test err | GLiClass Large v3: thr / val pos / val err / test pos / test err | ModernBERT zero-shot: thr / val pos / val err / test pos / test err |
|---|---|---|---|
| reset_equipment_mentioned | 0.50 / 47 / 0 / 50 / 0 | 0.72 / 47 / 0 / 50 / 0 | 0.24 / 47 / 0 / 50 / 0 |
| unable_to_work_due_to_service_issues | 0.50 / 23 / 0 / 18 / 0 | 0.50 / 23 / 0 / 18 / 0 | 0.50 / 23 / 0 / 18 / 0 |
| lagging_issues | 0.77 / 21 / 0 / 23 / 0 | 0.50 / 21 / 0 / 23 / 0 | 0.94 / 21 / 0 / 23 / 0 |
| poor_performance_complaint | 0.97 / 56 / 19 / 54 / 14 | 0.14 / 56 / 36 / 54 / 34 | 0.95 / 56 / 35 / 54 / 34 |
| glitching_issue | 0.50 / 45 / 0 / 31 / 0 | 0.50 / 45 / 0 / 31 / 0 | 0.86 / 45 / 1 / 31 / 0 |
| technician_visits_mentioned | 0.50 / 30 / 0 / 24 / 0 | 0.78 / 30 / 0 / 24 / 0 | 0.50 / 30 / 0 / 24 / 0 |
| poor_service_quality_complaint | 0.99 / 63 / 0 / 63 / 0 | 0.99 / 63 / 11 / 63 / 10 | 0.94 / 63 / 22 / 63 / 19 |
| throughput_issues_mentioned | 0.50 / 21 / 0 / 23 / 0 | 0.88 / 21 / 0 / 23 / 1 | 0.50 / 21 / 0 / 23 / 0 |
| receiving_error_messages | 0.45 / 48 / 5 / 57 / 7 | 0.63 / 48 / 0 / 57 / 1 | 0.61 / 48 / 6 / 57 / 6 |
| remove_phone_lines_requested | 0.50 / 21 / 0 / 13 / 0 | 0.50 / 21 / 0 / 13 / 0 | 0.50 / 21 / 0 / 13 / 0 |
| phone_purchase_upgrade_requested | 0.06 / 16 / 0 / 21 / 0 | 0.01 / 16 / 10 / 21 / 8 | 0.07 / 16 / 7 / 21 / 4 |
| issue_inside_building_mentioned | 0.50 / 60 / 0 / 64 / 0 | 0.50 / 60 / 0 / 64 / 0 | 0.50 / 60 / 0 / 64 / 0 |
| issue_outside_building_mentioned | 0.50 / 36 / 0 / 38 / 0 | 0.50 / 36 / 0 / 38 / 0 | 0.08 / 36 / 7 / 38 / 2 |
| sim_memory_issues_mentioned | 0.50 / 16 / 0 / 15 / 0 | 0.50 / 16 / 0 / 15 / 0 | 0.50 / 16 / 0 / 15 / 0 |
| app_issue_mentioned | 0.50 / 60 / 0 / 51 / 0 | 0.18 / 60 / 0 / 51 / 1 | 0.13 / 60 / 31 / 51 / 27 |
| unknown_charges_on_bill | 0.50 / 35 / 0 / 34 / 0 | 0.50 / 35 / 0 / 34 / 0 | 0.50 / 35 / 0 / 34 / 0 |
| credit_requested | 0.92 / 54 / 0 / 55 / 3 | 0.99 / 54 / 3 / 55 / 2 | 0.08 / 54 / 1 / 55 / 1 |
| credit_applied_by_agent | 0.50 / 28 / 0 / 16 / 0 | 0.50 / 28 / 0 / 16 / 2 | 0.50 / 28 / 0 / 16 / 0 |
| phone_service_issue | 0.99 / 49 / 5 / 45 / 3 | 0.95 / 49 / 2 / 45 / 2 | 0.94 / 49 / 4 / 45 / 9 |
| smart_home_manager_issue | 0.50 / 28 / 0 / 30 / 0 | 0.94 / 28 / 0 / 30 / 0 | 0.50 / 28 / 0 / 30 / 0 |
| sales_attempt_by_agent | 0.50 / 40 / 40 / 37 / 37 | 0.50 / 40 / 40 / 37 / 37 | 0.05 / 40 / 25 / 37 / 31 |
| agent_attempted_resolution | 0.95 / 128 / 19 / 129 / 21 | 0.50 / 128 / 0 / 129 / 0 | 0.95 / 128 / 10 / 129 / 10 |
| multiple_contacts_same_issue | 0.50 / 41 / 0 / 43 / 0 | 0.99 / 41 / 39 / 43 / 40 | 0.92 / 41 / 1 / 43 / 2 |
| fiber_line_issue_mentioned | 0.50 / 19 / 0 / 18 / 0 | 0.50 / 19 / 0 / 18 / 0 | 0.50 / 19 / 0 / 18 / 0 |
| unable_to_activate_internet | 0.50 / 32 / 0 / 31 / 0 | 0.50 / 32 / 0 / 31 / 0 | 0.50 / 32 / 0 / 31 / 0 |
| unable_to_connect_multiple_devices | 0.98 / 29 / 0 / 33 / 0 | 0.50 / 29 / 0 / 33 / 0 | 0.65 / 29 / 0 / 33 / 0 |
| resolution_time_concern | 0.50 / 57 / 0 / 52 / 0 | 0.99 / 57 / 0 / 52 / 0 | 0.50 / 57 / 0 / 52 / 0 |

## Where the errors are

- GLiClass Modern Large v3: 85 percent of 85 test errors fall in three attributes: `sales_attempt_by_agent` 37 (every positive missed on both splits); `agent_attempted_resolution` 21 (all false positives); `poor_performance_complaint` 14.
- GLiClass Large v3: 80 percent of 138 test errors fall in three attributes: `multiple_contacts_same_issue` 40 (all false positives); `sales_attempt_by_agent` 37 (every positive missed on both splits); `poor_performance_complaint` 34 (all false positives).
- ModernBERT zero-shot: 63 percent of 145 test errors fall in three attributes: `poor_performance_complaint` 34 (all false positives); `sales_attempt_by_agent` 31 (all misses); `app_issue_mentioned` 27.

An attribute whose positives are all missed, or whose errors are all confident false positives, is a problem of label wording or model coverage rather than of confidence; no gate recovers it, and the per-label gate's answer is to hand that attribute to people until the wording or the model changes.

## Why the hosted system is not in the gate table

| System | Accuracy % | Test errors | False positives among errors | Errors ≥ 2 nats | Errors ≥ 4 nats | Errors exactly on threshold | Distinct probability values | Validation probabilities |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| JEV Noul, 0.50 | 99.26 | 30 | 100 | 12 | 0 | 3 | 68 | committed |
| JEV Noul, validation-calibrated | 99.90 | 4 | 100 | 0 | 0 | 3 | 68 | committed |
| JEV Choice, 0.50 | 94.40 | 227 | 100 | 155 | 119 | 2 | 91 | none |

At the 0.50 rule the hosted system makes 30 test errors, every one a false positive, with 12 at least 2 nats (about 7:1 odds) from the threshold. The published validation calibration raises the threshold on 2 of the 27 labels and removes all but 4 of them; 3 of those 4 sit exactly on the threshold. Calibration, not a review gate, is this system's lever, and a gate fitted and evaluated on four errors would say nothing. The API returns probabilities in 0.01 steps (68 distinct values on test), which also limits where a cut can sit. JEV Choice makes 227 errors with 155 at least 2 nats from 0.50 and no committed validation probabilities, so a cut could not be chosen for it without touching the test split.

A reader-adjudicated sample of errors the gate catches is in `report/review-gate-adjudication-sample.md`.

Generated by `scripts/review_gate_economics.py`.
