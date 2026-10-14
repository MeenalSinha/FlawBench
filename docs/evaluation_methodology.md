# FlawBench Evaluation Methodology (Phase 5)

## 1. Principles of Scientific Evaluation

The FlawBench evaluation framework establishes a rigorous, reproducible benchmark for contrastive speech flaw detection and temporal localization.

Its architecture is governed by five foundational principles:

1. **Strict Inference Isolation (Label-Blind Execution)**:
   Detectors operate strictly without access to ground-truth labels, injected flaw types, injection severities, or injection time intervals. All predictions are generated independently and frozen into validated artifacts before any ground-truth annotation is read.
2. **Authoritative Temporal Geometry**:
   All speech flaws are localized on the **modified variant recording's timeline**. Temporal bounds are defined authoritatively as half-open integer sample intervals:
   $$[start\_sample, end\_sample) \quad \text{where } 0 \le start\_sample < end\_sample \le total\_samples$$
   Continuous timestamps ($start\_sec, end\_sec$) are calculated strictly using the audio's physical sample rate.
3. **Deterministic One-to-One Matching**:
   A single ground-truth event cannot be credited multiple times by multiple overlapping predictions. Event matching uses a deterministic greedy bipartite assignment algorithm that pairs candidates based on maximum Temporal Intersection over Union (IoU).
4. **Honest Mathematical Reporting**:
   When metrics have zero denominators (e.g. no predictions or no ground-truth events for a specific category), they are formally represented as undefined (`null` in JSON, `"N/A"` in reports) rather than converted into artificial zeros or perfect scores.
5. **Separation of Synthetic Benchmarks from Natural Speech**:
   All benchmark reports explicitly record the provenance of evaluated audio. Testing conducted on synthetic tone fixtures is transparently reported as synthetic verification; no claim of generalizability to natural human speech is made without authenticated speech corpora.

---

## 2. Evaluation Cohorts & Source-Level Splitting

### 2.1 Source-Level Disjoint Splitting
To prevent data leakage caused by acoustic familiarity across different modifications of the same speaker/utterance, FlawBench enforces source-level partitions (`src/flawbench/dataset/splits.py`). All variant recordings derived from reference recording $S_i$ belong exclusively to the same split.

- **Held-Out Test (`--split test`)**: Authoritative benchmark evaluation. Ground truth is never used for detector threshold selection.
- **Validation (`--split validation`)**: Partition reserved for tuning detector sensitivity thresholds.
- **Train (`--split train`)**: Exploratory data partition.
- **Diagnostic (`--split all`)**: Complete cohort evaluation for end-to-end regression auditing.

### 2.2 Cohort Verification
Before evaluation commences, `EvaluationCohort` verifies:
- Registered reference audio exists and matches SHA-256 checksums.
- Generated variant audio exists and matches SHA-256 checksums.
- Manifest annotations are non-corrupted and contain valid sample ranges.
- Detection outputs are either resolved from valid cached artifacts (`detections_manifest.jsonl`) or generated label-blind prior to loading annotations.

---

## 3. Temporal Matching Protocol

### 3.1 Temporal Intersection over Union (IoU)
For a predicted interval $P = [s_P, e_P)$ and a ground-truth interval $G = [s_G, e_G)$ defined in samples on the variant timeline:

$$\text{Intersection}(P, G) = \max(0, \min(e_P, e_G) - \max(s_P, s_G))$$
$$\text{Union}(P, G) = \max(e_P, e_G) - \min(s_P, s_G)$$
$$\text{IoU}(P, G) = \frac{\text{Intersection}(P, G)}{\text{Union}(P, G)}$$

If $\text{Union}(P, G) = 0$, $\text{IoU}(P, G) = 0.0$.

### 3.2 Bipartite Greedy Matching
For each evaluated sample:
1. Candidate pairs $(P_i, G_j)$ are formed where both intervals are non-empty.
2. For each pair, $\text{IoU}(P_i, G_j)$ is computed.
3. Candidate pairs are sorted descending by IoU:
   - Primary key: $\text{IoU}(P_i, G_j)$ (descending).
   - Tie-breaker 1: $\min(s_{P_i}, s_{G_j})$ (earlier start sample first).
   - Tie-breaker 2: $P_i.\text{prediction\_id}$ (deterministic string ordering).
4. Pairs are greedily accepted if:
   - $\text{IoU}(P_i, G_j) \ge \theta_{\text{IoU}}$ (default: $0.50$).
   - Category agreement: $P_i.\text{flaw\_type} == G_j.\text{flaw\_type}$ (when `require_category_match=True`).
   - Neither $P_i$ nor $G_j$ has already been matched.
5. Unmatched predictions are categorized as **False Positives** ($FP$).
6. Unmatched ground-truth annotations are categorized as **False Negatives** ($FN$).
7. All outcomes are written to the audit log `event_matches.jsonl`.

---

## 4. Quantitative Metrics

### 4.1 Classification Metrics
For each flaw category $c \in \{\text{pause\_insertion}, \text{tempo\_change}, \text{energy\_reduction}\}$:

- **True Positives ($TP_c$)**: Count of matched prediction-annotation pairs of category $c$.
- **False Positives ($FP_c$)**: Predictions of category $c$ with no qualifying ground-truth match.
- **False Negatives ($FN_c$)**: Ground-truth events of category $c$ with no qualifying prediction match.

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

### 4.2 Aggregate Metrics
- **Micro-Averaged Metrics**:
  Pooled across all categories, weighting every event equally:
  $$\text{Micro-Precision} = \frac{\sum_c TP_c}{\sum_c TP_c + \sum_c FP_c}$$
  $$\text{Micro-Recall} = \frac{\sum_c TP_c}{\sum_c TP_c + \sum_c FN_c}$$
  $$\text{Micro-}F1 = \frac{2 \cdot \text{Micro-Precision} \cdot \text{Micro-Recall}}{\text{Micro-Precision} + \text{Micro-Recall}}$$

- **Macro-Averaged Metrics**:
  Unweighted arithmetic mean across categories with defined scores:
  $$\text{Macro-}F1 = \frac{1}{|C_{\text{valid}}|} \sum_{c \in C_{\text{valid}}} F1_c$$

### 4.3 Temporal Localization & Boundary Errors
For all matched true-positive pairs $(P, G)$:
- **Mean & Median IoU**: Central tendency of temporal overlap quality.
- **Signed Start Error (seconds)**:
  $$\Delta t_{\text{start}} = \frac{s_P - s_G}{f_s}$$
  A positive error indicates the detector triggered late; negative indicates early trigger.
- **Signed End Error (seconds)**:
  $$\Delta t_{\text{end}} = \frac{e_P - e_G}{f_s}$$
  A positive error indicates the detector ended late; negative indicates early termination.
- **Signed Duration Error (seconds)**:
  $$\Delta t_{\text{dur}} = (e_P - s_P)/f_s - (e_G - s_G)/f_s$$
- **Mean Absolute Errors (MAE)**:
  $$\text{MAE}_{\text{start}} = \frac{1}{N} \sum |\Delta t_{\text{start}}|, \quad \text{MAE}_{\text{end}} = \frac{1}{N} \sum |\Delta t_{\text{end}}|, \quad \text{MAE}_{\text{dur}} = \frac{1}{N} \sum |\Delta t_{\text{dur}}|$$

### 4.4 Localization-Tolerance Curve
Evaluates detector localization robustness across strictness levels by computing event recall at varying IoU thresholds without re-running or re-tuning detectors:
$$\theta_{\text{IoU}} \in \{0.30, 0.50, 0.70, 0.90\}$$
$$\text{Recall}(\theta_{\text{IoU}}) = \frac{\text{Matched Events with IoU} \ge \theta_{\text{IoU}}}{\text{Total Ground-Truth Events}}$$

---

## 5. Confusion & Error Analysis

The event-level confusion matrix categorizes detector behavior:
1. **Diagonal Elements**: Correctly localized and correctly classified flaws.
2. **Off-Diagonal Elements**: Correctly localized intervals ($\text{IoU} \ge \theta_{\text{IoU}}$) assigned to an incorrect flaw category.
3. **Missed Events ($FN$)**: Ground-truth flaws that elicited no matching candidate.
4. **Spurious Detections ($FP$)**: Predicted flaws that correspond to no ground-truth manipulation.

---

## 6. Clean Negative Controls

To assess false-alarm susceptibility in the absence of flaws:
- **Identity Comparison**: When genuine unmanipulated speech recordings are unavailable, the evaluation runner generates an identity control comparing reference recording $R_i$ with itself ($R_i \text{ vs. } R_i$).
- **Expected Outcome**: Empty prediction set ($\emptyset$).
- **False-Positive Audit**:
  $$\text{False-Positive Rate} = \frac{\text{Controls with } \ge 1 \text{ Prediction}}{\text{Total Controls Evaluated}}$$

---

## 7. Statistical Uncertainty: Source-Level Bootstrap

When evaluating speech data, individual variant recordings from the same speaker/reference share acoustic traits. Standard independent-sample bootstrapping underestimates variance.

FlawBench implements **Source-Level Cluster Bootstrap**:
1. Unique source IDs $\{S_1, S_2, \dots, S_K\}$ are resampled with replacement for $B$ replicates (default: 100).
2. All sample evaluations belonging to the resampled sources are aggregated.
3. Micro/macro metrics and mean IoU are computed for each replicate.
4. Empirical 95% confidence intervals are reported from the 2.5th and 97.5th percentiles.
5. If $K < 3$, confidence intervals are flagged as unstable or unavailable due to small cluster sample size.

---

## 8. Reproducibility & Provenance

Every evaluation run creates an immutable artifact directory under `data/evaluations/<evaluation_run_id>/`:
- `evaluation_report.json`: Full quantitative report and configuration hashes.
- `event_matches.jsonl`: Line-by-line event match outcomes.
- `report.md`: Markdown summary.
- Global update to `data/evaluations/evaluations_manifest.jsonl`.
- Validated with `flawbench evaluate validate --run <run_id>`.
