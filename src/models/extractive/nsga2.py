from typing import List, Optional, Dict
import os
import json
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.core.mutation import Mutation
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.optimize import minimize


class SummarizationProblem(ElementwiseProblem):
    def __init__(self, sentences: List[str], importance: List[float], sim_mat: Optional[np.ndarray], max_tokens: int):
        self.sentences = sentences
        self.importance = np.array(importance, dtype=float)
        self.sim_mat = sim_mat
        self.max_tokens = max_tokens
        self.tok_len = np.array([len(s.split()) for s in sentences], dtype=int)
        n = len(sentences)
        super().__init__(n_var=n, n_obj=3, n_constr=1, xl=0, xu=1, type_var=int)

    def _evaluate(self, x, out, *args, **kwargs):
        idx = np.where(np.asarray(x) > 0)[0]

        imp = float(np.sum(self.importance[idx])) if idx.size > 0 else 0.0

        if self.sim_mat is not None and idx.size > 0:
            sub = self.sim_mat[:, idx]
            cov = float(np.mean(np.max(sub, axis=1)))
        else:
            cov = 0.0

        if self.sim_mat is not None and idx.size > 1:
            S = self.sim_mat[np.ix_(idx, idx)]
            iu = np.triu_indices(len(idx), k=1)
            red = float(np.mean(S[iu])) if S[iu].size > 0 else 0.0
        else:
            red = 0.0

        out["F"] = [-imp, -cov, red]
        total_tokens = int(self.tok_len[idx].sum())
        out["G"] = [total_tokens - self.max_tokens]


class LengthRepair(Repair):
    def _do(self, problem: SummarizationProblem, X, **kwargs):
        imp = problem.importance
        tok_len = problem.tok_len
        max_tokens = problem.max_tokens

        Y = X.copy()
        for r in range(Y.shape[0]):
            sel = np.where(Y[r] > 0)[0]
            if sel.size == 0:
                continue

            cur_tokens = int(tok_len[sel].sum())
            if cur_tokens <= max_tokens:
                continue

            sel_set = set(sel.tolist())
            order = sorted(sel_set, key=lambda i: imp[i])
            for i in order:
                if cur_tokens <= max_tokens:
                    break
                sel_set.remove(i)
                cur_tokens -= int(tok_len[i])

            Y[r, :] = 0
            if len(sel_set) > 0:
                Y[r, np.array(sorted(sel_set), dtype=int)] = 1

        return Y


class AdaptiveDropCallback(Callback):
    def __init__(self,
                 n_sent: int,
                 window: int = 5,
                 min_freq: float = 0.05,
                 initial_drop: float = 0.0,
                 min_drop: float = 0.0,
                 max_drop: float = 0.35,
                 adapt_freq: int = 5,
                 warmup_generations: int = 0):
        super().__init__()
        self.n = n_sent
        self.window = int(window)
        self.min_freq = float(min_freq)
        self.drop = np.full(n_sent, float(initial_drop), dtype=float)
        self.min_drop = float(min_drop)
        self.max_drop = float(max_drop)
        self.adapt_freq = int(adapt_freq)
        self.warm = int(warmup_generations)
        self.hist = []
        self.t = 0
        self.data["drop_history"] = []

    def notify(self, algorithm):
        X = algorithm.pop.get("X")
        if X is None:
            return
        self.t += 1
        freq = X.mean(axis=0).astype(float)
        self.hist.append(freq)

        if self.t <= self.warm:
            return

        if (self.t - self.warm) % self.adapt_freq == 0:
            W = self.hist[-self.window:] if len(self.hist) >= self.window else self.hist
            avg = np.vstack(W).mean(axis=0)
            up_mask = avg < self.min_freq
            down_mask = ~up_mask
            self.drop[up_mask] = np.minimum(self.max_drop, self.drop[up_mask] + 0.05)
            self.drop[down_mask] = np.maximum(self.min_drop, self.drop[down_mask] - 0.02)
            self.data["drop_history"].append(self.drop.copy())


class AdaptiveBitflip(Mutation):
    def __init__(self, callback: AdaptiveDropCallback, base_flip_prob: Optional[float] = None):
        super().__init__()
        self.cb = callback
        self.base = base_flip_prob

    def _do(self, problem, X, **kwargs):
        n = X.shape[1]  # 設計變數數量（句子數）
        base = self.base or (1.0 / max(1, n))
        drop = self.cb.drop if (self.cb is not None) else np.zeros(n, dtype=float)

        # 機率夾住，避免越界
        p10 = np.clip(drop + base, 0.0, 0.9)          # 1->0 機率
        p01 = np.clip(base * (1.0 - drop), 0.0, 0.9)  # 0->1 機率

        Y = X.copy()
        for i in range(Y.shape[0]):  # 個體數量
            ones_idx = np.where(Y[i] == 1)[0]
            zeros_idx = np.where(Y[i] == 0)[0]

            # 1->0 翻轉
            if ones_idx.size > 0:
                flip_10 = (np.random.rand(ones_idx.size) < p10[ones_idx])
                if flip_10.any():
                    Y[i, ones_idx[flip_10]] = 0

            # 0->1 翻轉
            if zeros_idx.size > 0:
                flip_01 = (np.random.rand(zeros_idx.size) < p01[zeros_idx])
                if flip_01.any():
                    Y[i, zeros_idx[flip_01]] = 1

        return Y


def _prune_redundancy_by_sim(indices: List[int],
                             sim: Optional[np.ndarray],
                             importance: np.ndarray,
                             threshold: float = 0.9) -> List[int]:
    if sim is None or len(indices) <= 1:
        return sorted(indices)
    kept: List[int] = []
    for i in sorted(indices, key=lambda k: importance[k], reverse=True):
        if all(sim[i, j] < threshold for j in kept):
            kept.append(i)
    kept.sort()
    return kept


def nsga2_select(
    sentences: List[str],
    importance: List[float],
    sim_mat: Optional[np.ndarray],
    max_tokens: int,
    lambda_importance: float = 1.0,
    lambda_coverage: float = 0.8,
    lambda_redundancy: float = 0.7,
    adaptive_drop_cfg: Optional[Dict] = None,
    redundancy_threshold: float = 0.9,
    adaptive_log_path: Optional[str] = None,
    seed: Optional[int] = None,
    warmup_generations: int = 0,
    pop_size: int = 30,
    n_gen: int = 30,
) -> List[int]:
    n = len(sentences)
    if n == 0:
        return []

    problem = SummarizationProblem(sentences, importance, sim_mat, max_tokens)
    sampling = BinaryRandomSampling()
    crossover = TwoPointCrossover()
    baseline_mut = BitflipMutation()

    cb = None
    mutation = baseline_mut
    if adaptive_drop_cfg and adaptive_drop_cfg.get("enabled", False):
        cb = AdaptiveDropCallback(
            n_sent=n,
            window=int(adaptive_drop_cfg.get("window", 5)),
            min_freq=float(adaptive_drop_cfg.get("min_freq", 0.05)),
            initial_drop=float(adaptive_drop_cfg.get("initial_drop_rate", 0.0)),
            min_drop=float(adaptive_drop_cfg.get("min_drop_rate", 0.0)),
            max_drop=float(adaptive_drop_cfg.get("max_drop_rate", 0.35)),
            adapt_freq=int(adaptive_drop_cfg.get("adaptation_frequency", 5)),
            warmup_generations=int(warmup_generations),
        )
        mutation = AdaptiveBitflip(cb, base_flip_prob=None)

    repair = LengthRepair()
    pop = max(20, min(2 * n, int(pop_size)))

    algorithm = NSGA2(
        pop_size=pop,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
        repair=repair,
    )

    use_seed = 42 if seed is None else int(seed)
    if cb is not None:
        res = minimize(problem, algorithm, ("n_gen", int(n_gen)), verbose=False, seed=use_seed, callback=cb)
    else:
        res = minimize(problem, algorithm, ("n_gen", int(n_gen)), verbose=False, seed=use_seed)

    if adaptive_log_path and cb is not None:
        try:
            os.makedirs(os.path.dirname(adaptive_log_path) or ".", exist_ok=True)
            with open(adaptive_log_path, "w", encoding="utf-8") as f:
                json.dump({"drop_history": [arr.tolist() for arr in cb.data.get("drop_history", [])]}, f, ensure_ascii=False)
        except Exception:
            pass

    if res.X is None:
        return []

    X = np.atleast_2d(res.X)
    best_val = -1e18
    best_idx = -1
    imp_arr = np.array(importance, dtype=float)

    for i, x in enumerate(X):
        idx = np.where(x > 0)[0]
        imp = float(np.sum(imp_arr[idx])) if idx.size > 0 else 0.0
        if sim_mat is not None and idx.size > 0:
            cov = float(np.mean(np.max(sim_mat[:, idx], axis=1)))
        else:
            cov = 0.0

        if sim_mat is not None and idx.size > 1:
            S = sim_mat[np.ix_(idx, idx)]
            iu = np.triu_indices(len(idx), k=1)
            red = float(np.mean(S[iu])) if S[iu].size > 0 else 0.0
        else:
            red = 0.0

        val = lambda_importance * imp + lambda_coverage * cov - lambda_redundancy * red
        if val > best_val:
            best_val = val
            best_idx = i

    if best_idx < 0:
        return []

    chosen = X[best_idx]
    sel = np.where(chosen > 0)[0].tolist()
    sel = _prune_redundancy_by_sim(sel, sim_mat, imp_arr, threshold=redundancy_threshold)
    return sel


if __name__ == "__main__":
    sentences = [
        "The cat sits on the mat.",
        "Dogs are loyal animals.",
        "Artificial intelligence is transforming the world.",
        "The quick brown fox jumps over the lazy dog.",
        "Data science is an interdisciplinary field.",
        "Machine learning is a subset of AI.",
        "Python is widely used for data analysis.",
        "Deep learning uses neural networks.",
        "Natural language processing deals with text data.",
        "Statistics is fundamental for data science."
    ]
    importance = [0.8, 0.6, 0.9, 0.5, 0.7, 0.85, 0.65, 0.9, 0.75, 0.7]

    rng = np.random.default_rng(42)
    sim_mat = rng.random((len(sentences), len(sentences)))
    sim_mat = (sim_mat + sim_mat.T) / 2.0
    np.fill_diagonal(sim_mat, 1.0)

    max_tokens = 20
    lambda_importance = 1.0
    lambda_coverage = 0.9
    lambda_redundancy = 0.7
    post_threshold = 0.95
    warmup_generations = 8
    pop_size = 24
    n_gen = 24

    sel_no_adaptive = nsga2_select(
        sentences=sentences,
        importance=importance,
        sim_mat=sim_mat,
        max_tokens=max_tokens,
        lambda_importance=lambda_importance,
        lambda_coverage=lambda_coverage,
        lambda_redundancy=lambda_redundancy,
        adaptive_drop_cfg=None,
        redundancy_threshold=post_threshold,
        seed=42,
        warmup_generations=warmup_generations,
        pop_size=pop_size,
        n_gen=n_gen,
    )
    print("=== No Adaptive Drop ===")
    print("Selected indices:", sel_no_adaptive)
    print("Selected sentences:", [sentences[i] for i in sel_no_adaptive])

    adaptive_cfg = {
        "enabled": True,
        "window": 11,
        "adaptation_frequency": 15,
        "min_freq": 0.15,
        "initial_drop_rate": 0.0,
        "min_drop_rate": 0.0,
        "max_drop_rate": 0.15,
    }
    adaptive_log_path = "adaptive_history_demo.json"

    sel_adaptive = nsga2_select(
        sentences=sentences,
        importance=importance,
        sim_mat=sim_mat,
        max_tokens=max_tokens,
        lambda_importance=lambda_importance,
        lambda_coverage=lambda_coverage,
        lambda_redundancy=lambda_redundancy,
        adaptive_drop_cfg=adaptive_cfg,
        redundancy_threshold=post_threshold,
        adaptive_log_path=adaptive_log_path,
        seed=42,
        warmup_generations=warmup_generations,
        pop_size=pop_size,
        n_gen=n_gen,
    )
    print("\n=== With Adaptive Drop ===")
    print("Selected indices:", sel_adaptive)
    print("Selected sentences:", [sentences[i] for i in sel_adaptive])
    print(f"Adaptive drop history saved to: {adaptive_log_path}")
