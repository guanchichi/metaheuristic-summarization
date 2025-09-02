import argparse
import csv
import json
import os
import time
from typing import List, Dict

from src.utils.io import read_jsonl
from src.eval.rouge import rouge_scores


def _load_select_time(run_dir: str) -> Dict[str, float]:
    """
    嘗試從 run 目錄讀取選句時間，若不存在則回傳空字典。
    """
    path = os.path.join(run_dir, "time_select_seconds.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 期望包含 {"time_select_seconds": float}
            if isinstance(data, dict) and "time_select_seconds" in data:
                return {"time_select_seconds": float(data["time_select_seconds"])}
        except Exception:
            pass
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions.jsonl path")
    ap.add_argument("--out", required=True, help="metrics.csv output path")
    args = ap.parse_args()

    preds: List[str] = []
    refs: List[str] = []
    for row in read_jsonl(args.pred):
        preds.append(row.get("summary", ""))
        refs.append(row.get("reference", ""))

    # 評估計時（高解析度 perf_counter 適合短時段量測）
    t0 = time.perf_counter()
    m: Dict[str, float] = rouge_scores(preds, refs)
    t_eval = round(time.perf_counter() - t0, 6)

    # 準備輸出資料夾
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # 嘗試帶入選句時間（同一 run 目錄）
    run_dir = os.path.dirname(args.pred) if os.path.isfile(args.pred) else args.pred
    extra_times = _load_select_time(run_dir)

    # 輸出 CSV：既有的 metric/value 之外，追加 time_* 欄位
    # 為了兼容原格式，維持兩欄表頭不變，追加時間以同一規則輸出
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in m.items():
            w.writerow([k, f"{v:.6f}"])

        # 追加時間統計
        w.writerow(["time_eval_seconds", f"{t_eval:.6f}"])
        if "time_select_seconds" in extra_times:
            w.writerow(["time_select_seconds", f"{extra_times['time_select_seconds']:.6f}"])

    print(f"ROUGE written to {args.out}")


if __name__ == "__main__":
    main()
