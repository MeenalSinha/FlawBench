"""Event-level categorization confusion matrix construction and error analysis."""

from __future__ import annotations

from flawbench.config import VALID_DETECTION_CATEGORIES
from flawbench.evaluation.schema import ConfusionMatrix, EventMatch


def build_confusion_matrix(
    matches: list[EventMatch],
    categories: list[str] | tuple[str, ...] = VALID_DETECTION_CATEGORIES,
) -> ConfusionMatrix:
    """Construct an event-level confusion matrix from event matching results.

    Rows: True Ground-Truth Category (including "none" for spurious ungrounded predictions).
    Columns: Inferred Category (including "missed" for un-detected ground truth).
    """
    cat_list = list(categories)
    row_keys = cat_list + ["none"]
    col_keys = cat_list + ["missed"]

    matrix: dict[str, dict[str, int]] = {r: {c: 0 for c in col_keys} for r in row_keys}

    for m in matches:
        if m.match_status == "matched":
            r = m.gt_flaw_type if m.gt_flaw_type in matrix else "none"
            c = m.pred_flaw_type if m.pred_flaw_type in col_keys else "missed"
            matrix[r][c] += 1
        elif m.match_status == "unmatched_prediction":
            r = "none"
            c = m.pred_flaw_type if m.pred_flaw_type in col_keys else "missed"
            matrix[r][c] += 1
        elif m.match_status == "unmatched_ground_truth":
            r = m.gt_flaw_type if m.gt_flaw_type in matrix else "none"
            c = "missed"
            matrix[r][c] += 1

    return ConfusionMatrix(categories=cat_list, matrix=matrix)
