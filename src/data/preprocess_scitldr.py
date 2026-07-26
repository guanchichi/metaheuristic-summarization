import argparse
import os
from typing import List, Dict
from datasets import load_dataset
from src.utils.io import ensure_dir, write_jsonl

def process_example(ex: Dict) -> Dict:
    # scitldr structure:
    # source: list of strings (sentences)
    # source_labels: list of ints
    # rouge_scores: list of floats
    # paper_id: string
    # target: list of strings (summaries)
    
    return {
        "id": ex["paper_id"],
        "sentences": ex["source"],
        "highlights": " ".join(ex["target"]), # Join target sentences into a single string for reference
        "rouge_scores": ex["rouge_scores"],
        "source_labels": ex["source_labels"]
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="dataset split: train, validation, test")
    ap.add_argument("--out", default=None, help="output JSONL path")
    args = ap.parse_args()

    print(f"Loading scitldr (AIC) split: {args.split}...")
    ds = load_dataset("allenai/scitldr", "AIC", split=args.split, trust_remote_code=True)
    
    out_path = args.out or os.path.join("data", "processed", f"scitldr_{args.split}.jsonl")
    ensure_dir(os.path.dirname(out_path))

    rows = []
    for ex in ds:
        rows.append(process_example(ex))
    
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} items to {out_path}")

if __name__ == "__main__":
    main()
