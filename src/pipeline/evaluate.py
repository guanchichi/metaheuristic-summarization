import argparse
import csv
import os
import time
from typing import Any, Dict, Iterable, List, Tuple

from src.data.schemas import extract_references
from src.eval.protocol import KNOWN_PROTOCOLS, evaluate_corpus
from src.utils.io import read_jsonl


def align_evaluation_rows(
    prediction_rows: Iterable[Dict[str, Any]],
    gold_rows: Iterable[Dict[str, Any]],
) -> Tuple[List[str], List[List[str]]]:
    """Join predictions to gold references by ID and reject partial alignment."""

    gold_by_id: Dict[str, List[str]] = {}
    for line_number, row in enumerate(gold_rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"gold row {line_number} has no valid 'id'")
        if row_id in gold_by_id:
            raise ValueError(f"duplicate gold id: {row_id}")
        references = extract_references(row)
        if not references:
            raise ValueError(f"gold row {line_number} has no non-empty references")
        gold_by_id[row_id] = references

    predictions: List[str] = []
    references: List[List[str]] = []
    prediction_ids = set()
    for line_number, row in enumerate(prediction_rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"prediction row {line_number} has no valid 'id'")
        if row_id in prediction_ids:
            raise ValueError(f"duplicate prediction id: {row_id}")
        prediction_ids.add(row_id)
        if not isinstance(row.get("summary"), str):
            raise ValueError(f"prediction row {line_number} must contain a string 'summary'")
        if row_id not in gold_by_id:
            raise ValueError(f"prediction id {row_id!r} is missing from the gold dataset")
        predictions.append(row["summary"])
        references.append(gold_by_id[row_id])

    missing_predictions = set(gold_by_id) - prediction_ids
    if missing_predictions:
        preview = sorted(missing_predictions)[:5]
        raise ValueError(
            f"gold/prediction ID mismatch: {len(missing_predictions)} gold IDs have no "
            f"prediction; first IDs: {preview}"
        )
    return predictions, references


def load_evaluation_inputs(
    prediction_path: str, gold_path: str
) -> Tuple[List[str], List[List[str]]]:
    return align_evaluation_rows(read_jsonl(prediction_path), read_jsonl(gold_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions.jsonl path")
    ap.add_argument("--gold", required=True, help="canonical or legacy gold dataset JSONL")
    ap.add_argument("--out", required=True, help="metrics.csv output path")
    ap.add_argument(
        "--protocol",
        required=True,
        choices=KNOWN_PROTOCOLS,
        help="explicit dataset evaluation protocol",
    )
    args = ap.parse_args()

    preds, refs = load_evaluation_inputs(args.pred, args.gold)

    t0 = time.perf_counter()
    m = evaluate_corpus(preds, refs, protocol=args.protocol)
    t1 = time.perf_counter()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["evaluation_protocol", args.protocol])
        # write rouge metrics
        for k, v in m.items():
            w.writerow([k, f"{v:.6f}"])
        # append time statistics
        # selection time (if produced by select_sentences in the same directory)
        sel_time_file = os.path.join(os.path.dirname(args.out), "time_select_seconds.txt")
        if os.path.exists(sel_time_file):
            with open(sel_time_file, "r", encoding="utf-8") as fr:
                val = float((fr.read() or "0").strip())
                w.writerow(["time_select_seconds", f"{val:.6f}"])
        # evaluation time
        w.writerow(["time_eval_seconds", f"{(t1 - t0):.6f}"])
    print(f"ROUGE written to {args.out}")


if __name__ == "__main__":
    main()

