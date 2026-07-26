import argparse
import os
from typing import Dict

from src.data.schemas import build_document_example
from src.utils.io import ensure_dir, write_jsonl

def process_example(ex: Dict, split: str = "test") -> Dict:
    """Convert SciTLDR AIC to the canonical single-document contract.

    ``target`` contains alternative human summaries.  They must remain
    separate references; concatenating them changes both the task and ROUGE.
    """

    return build_document_example(
        example_id=ex["paper_id"],
        split=split,
        documents=[ex["source"]],
        references=ex["target"],
        input_mode="single_document",
        output_mode="single_sentence",
        dataset_name="SciTLDR-AIC",
        metadata={
            "source_labels": ex.get("source_labels", []),
            "rouge_scores": ex.get("rouge_scores", []),
        },
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="dataset split: train, validation, test")
    ap.add_argument("--out", default=None, help="output JSONL path")
    args = ap.parse_args()

    print(f"Loading scitldr (AIC) split: {args.split}...")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Downloading SciTLDR requires the optional 'datasets' package. "
            "Install the full research requirements before running this command."
        ) from exc
    ds = load_dataset("allenai/scitldr", "AIC", split=args.split, trust_remote_code=True)
    
    out_path = args.out or os.path.join("data", "processed", f"scitldr_{args.split}.jsonl")
    ensure_dir(os.path.dirname(out_path))

    rows = []
    for ex in ds:
        rows.append(process_example(ex, split=args.split))
    
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} items to {out_path}")

if __name__ == "__main__":
    main()
