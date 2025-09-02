import argparse
import os
import sys
import time
import json
import platform
from typing import Dict, List, Optional

import numpy as np

from src.utils.io import load_yaml, ensure_dir, now_stamp, read_jsonl, write_jsonl, set_global_seed
from src.features.tf_isf import sentence_tf_isf_scores
from src.features.length import length_scores
from src.features.position import position_scores
from src.features.compose import combine_scores
from src.representations.sent_vectors import SentenceVectors
from src.representations.similarity import cosine_similarity_matrix
from src.selection.candidate_pool import topk_by_score
from src.models.extractive.greedy import greedy_select
from src.models.extractive.grasp import grasp_select
from src.models.extractive.nsga2 import nsga2_select
from src.models.extractive.supervised import SupervisedScorer


def _write_env_txt(out_dir: str) -> None:
    lines = [
        f"python: {sys.version}".strip(),
        f"os: {platform.platform()}".strip(),
    ]
    # 嘗試寫出關鍵套件版本（若存在）
    try:
        import sklearn  # type: ignore
        lines.append(f"sklearn: {sklearn.__version__}")
    except Exception:
        pass
    try:
        import torch  # type: ignore
        lines.append(f"torch: {torch.__version__}")
    except Exception:
        pass
    try:
        import transformers  # type: ignore
        lines.append(f"transformers: {transformers.__version__}")
    except Exception:
        pass
    try:
        import sentence_transformers  # type: ignore
        lines.append(f"sentence-transformers: {sentence_transformers.__version__}")
    except Exception:
        pass

    with open(os.path.join(out_dir, "env.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_base_scores(sentences: List[str], cfg: Dict) -> List[float]:
    # 基線特徵：TF-ISF、長度、句位，依既有 combine_scores 加權與 normalize
    f_importance = sentence_tf_isf_scores(sentences)
    f_len = length_scores(sentences)
    f_pos = position_scores(sentences)
    weights = {
        "importance": float(cfg.get("objectives", {}).get("lambda_importance", 1.0)),
        "length": 0.3,
        "position": 0.3,
    }
    feats = {"importance": f_importance, "length": f_len, "position": f_pos}
    return combine_scores(feats, weights)


def summarize_one(
    doc: Dict,
    cfg: Dict,
    model_path: Optional[str] = None,
    run_dir: Optional[str] = None,
) -> Dict:
    sentences: List[str] = doc.get("sentences", [])
    highlights: str = doc.get("highlights", "")

    # 1) 句級分數：監督式優先，否則用基線特徵
    if model_path:
        scorer = SupervisedScorer(model_path)
        base_scores = scorer.predict_scores(sentences, cfg)
    else:
        base_scores = build_base_scores(sentences, cfg)

    # 2) 解析優化器/冗餘權重，決定是否需要相似度
    method_opt = cfg.get("optimizer", {}).get("method", "greedy").lower()
    alpha = float(cfg.get("redundancy", {}).get("lambda", 0.7))
    # greedy/grasp 在 alpha<1 時需要相似度；NSGA-II 一律需要相似度作覆蓋/冗餘
    need_sim = (method_opt in ["greedy", "grasp"] and alpha < 1.0) or (method_opt == "nsga2")

    max_tokens = int(cfg.get("length_control", {}).get("max_tokens", 100))

    # 3) 候選池子集法：依 base_scores 取 top-k 作為搜尋空間
    k = int(cfg.get("candidates", {}).get("k", min(15, len(sentences))))
    use_cand = bool(cfg.get("candidates", {}).get("use", True))
    if use_cand:
        cand_idx = topk_by_score(base_scores, k)
    else:
        cand_idx = list(range(len(sentences)))
    cand_idx = sorted(cand_idx)

    # 子集資料
    S_sent = [sentences[i] for i in cand_idx]
    S_score = [base_scores[i] for i in cand_idx]

    # 4) representations.use 閘道：若不需要相似度或 use=false，則跳過，讓選句器走 sim=None 路徑
    rep_cfg = cfg.get("representations", {})
    rep_use = bool(rep_cfg.get("use", True))
    rep_method = rep_cfg.get("method", "tfidf")
    if need_sim and rep_use and len(S_sent) > 0:
        vec = SentenceVectors(method=rep_method)
        X = vec.fit_transform(S_sent)
        S_sim = cosine_similarity_matrix(X)
    else:
        S_sim = None  # 與選句器的冗餘退化相容

    # 5) 子集上呼叫對應優化器
    if method_opt == "greedy":
        picked_sub = greedy_select(S_sent, S_score, S_sim, max_tokens, alpha=alpha)
    elif method_opt == "grasp":
        picked_sub = grasp_select(S_sent, S_score, S_sim, max_tokens, alpha=alpha, iters=10, seed=cfg.get("seed"))
    elif method_opt == "nsga2":
        try:
            adaptive_cfg = cfg.get("adaptive_drop", {})  # 透傳 adaptive drop 參數
            post_thr = float(cfg.get("redundancy", {}).get("post_threshold", 0.9))  # 輸出前冗餘剔除閾值
            # 若啟用 adaptive，為每篇文件寫出一份歷史以利分析（以回呼保存每代資料）
            adaptive_log_path = None
            if adaptive_cfg.get("enabled", False) and run_dir is not None:
                doc_id = str(doc.get("id"))
                adaptive_log_path = os.path.join(run_dir, f"adaptive_history_{doc_id}.json")

            picked_sub = nsga2_select(
                S_sent,
                S_score,
                S_sim,
                max_tokens,
                lambda_importance=float(cfg.get("objectives", {}).get("lambda_importance", 1.0)),
                lambda_coverage=float(cfg.get("objectives", {}).get("lambda_coverage", 0.8)),
                lambda_redundancy=float(cfg.get("objectives", {}).get("lambda_redundancy", 0.7)),
                adaptive_drop_cfg=adaptive_cfg,
                redundancy_threshold=post_thr,
                adaptive_log_path=adaptive_log_path,  # 新增：寫出回呼歷史
            )
        except ImportError as e:
            print(f"Warning: pymoo not available for NSGA-II, falling back to greedy: {e}")
            picked_sub = greedy_select(S_sent, S_score, S_sim, max_tokens, alpha=alpha)
        except Exception as e:
            print(f"Warning: NSGA-II optimization failed, falling back to greedy: {e}")
            picked_sub = greedy_select(S_sent, S_score, S_sim, max_tokens, alpha=alpha)
    elif method_opt == "supervised":
        if not model_path:
            raise RuntimeError("supervised 模式需要提供 model_path")
        picked_sub = greedy_select(S_sent, S_score, S_sim, max_tokens, alpha=alpha)
    else:
        picked_sub = greedy_select(S_sent, S_score, S_sim, max_tokens, alpha=alpha)

    # 6) 子集索引映回原始索引，並以原始順序輸出
    selected = sorted(cand_idx[i] for i in picked_sub)
    summary = " ".join([sentences[i] for i in selected])
    return {
        "id": doc.get("id"),
        "selected_indices": selected,
        "summary": summary,
        "reference": highlights,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config yaml")
    ap.add_argument("--split", required=True, help="dataset split name")
    ap.add_argument("--input", required=True, help="processed jsonl path")
    ap.add_argument("--run_dir", default="runs", help="runs output root")
    ap.add_argument("--stamp", default=None, help="optional fixed stamp for output dir")
    ap.add_argument("--optimizer", default=None, help="override optimizer.method in config")
    ap.add_argument("--model", default=None, help="path to supervised model (joblib)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.optimizer:
        cfg.setdefault("optimizer", {})
        cfg["optimizer"]["method"] = args.optimizer

    set_global_seed(cfg.get("seed"))
    stamp = args.stamp or now_stamp()
    out_dir = os.path.join(args.run_dir, stamp)
    ensure_dir(out_dir)

    # 寫入環境資訊，支援可重現性
    _write_env_txt(out_dir)

    # 選句流程計時
    t0 = time.perf_counter()

    preds_path = os.path.join(out_dir, "predictions.jsonl")
    rows = []
    for doc in read_jsonl(args.input):
        rows.append(summarize_one(doc, cfg, model_path=args.model, run_dir=out_dir))
    write_jsonl(preds_path, rows)

    # also dump the config used
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # 寫入選句耗時（供 evaluate 併入 metrics.csv）
    time_used = round(time.perf_counter() - t0, 6)
    with open(os.path.join(out_dir, "time_select_seconds.json"), "w", encoding="utf-8") as f:
        json.dump({"time_select_seconds": time_used}, f)

    print(f"Wrote predictions to {preds_path}")


if __name__ == "__main__":
    main()
