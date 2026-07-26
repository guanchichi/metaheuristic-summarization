# Metaheuristic Extractive Summarization

抽取式摘要研究程式碼。多目標最佳化（NSGA-II）、圖中心性與句向量語意訊號的組合，
目標是在 **zero-training（不做任務微調）** 的條件下研究 quality–cost trade-off。

---

## ⚠️ 專案狀態：重構中，既有結果不可引用

這個 repo 目前正在依 IEEE Access 投稿標準做**正確性重構**。開始使用前請先知道：

| 項目 | 狀態 |
|---|---|
| `runs/` 底下的既有結果 | 🔴 **無效** —— 超參數是在 test set 上選的（test-set overfitting） |
| Stage 2 的 `w_bert` 參數 | 🔴 **命名誤導** —— 它加權的是 TF-IDF 分數，不是 BERT。Stage 2 目前沒有 PLM |
| ROUGE-L | 🟠 舊碼用單序列 `rougeL`；已改為多句適用的 `rougeLsum`，但與 published Perl ROUGE 的 parity 尚未驗證 |
| Baseline（Lead / TextRank / LexRank / PacSum） | 🔴 **尚未實作** —— 舊論文表格的 baseline 數字引用自其他論文，不是本 repo 產出 |
| 三軌候選生成 | 🟠 三軌目前共用同一個預先過濾的候選池，不是各自在完整輸入上排名 |

**簡言之：程式可以跑，但目前的輸出不能當研究結論。**

---

## 📚 研究文件（協作者從這裡開始）

全部在 [`docs/research/`](docs/research/)：

| 文件 | 用途 |
|---|---|
| [`INDEX.md`](docs/research/INDEX.md) | **總索引 + 關鍵數字速查** ← 先看這個 |
| [`ACTION_PLAN.md`](docs/research/ACTION_PLAN.md) | 要做什麼、什麼順序、完成定義 ← 日常執行看這份 |
| [`ARCHITECTURE.md`](docs/research/ARCHITECTURE.md) | Target Architecture v1、schema、模組介面、freeze gate |
| [`paper_revision_plan_IEEE_Access.md`](docs/research/paper_revision_plan_IEEE_Access.md) | 研究流程治理、10 個 P0、投稿合規 |
| [`CODE_AUDIT_IEEE_Access.md`](docs/research/CODE_AUDIT_IEEE_Access.md) | 已驗證的程式缺陷 + 實測數字 |
| [`STRATEGY_ASSESSMENT.md`](docs/research/STRATEGY_ASSESSMENT.md) | 可行性評估、病因診斷、資料集選擇 |
| [`REPO_CLEANUP.md`](docs/research/REPO_CLEANUP.md) | 專案整理計畫 |

AI 協作規則見 repo 根目錄的 [`CLAUDE.md`](CLAUDE.md)。

> 重構前的 legacy 文件（`PIPELINE.md`、`RUNS.md`、`CONFIGS.md`、`PROJECT_STATUS.md` 等）
> 已移至 `docs/_legacy_docs/` 並排除於版本庫外 —— 它們引用的路徑多數已不存在。

---

## 先看哪裡 / 可以先略過哪裡

```
src/          ← 研究主線，看這裡
scripts/audit/← 稽核診斷腳本（Lead 比較、選句位置分析、headroom、PLM 計時）
tests/        ← 單元測試
configs/      ← 實驗設定（歷史檔 _legacy_archive/ 已排除於版本庫外）

frontend/     ← 🚫 展示用 web UI，與論文無關，可以完全略過
backend/      ← 🚫 展示用 API server，與論文無關，可以完全略過
experimental/ ← 🚫 抽象式摘要與 rerank 的探索，與本論文主線無關
notebooks/    ← 🚫 空的
```

> **給協作者**：只需要看 `src/`、`scripts/audit/`、`tests/`、`configs/`。
> `frontend/`、`backend/`、`experimental/` 不用讀，它們不影響任何研究結果。

---

## 安裝

Python 3.10+（開發環境為 3.12）

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Unix
source .venv/bin/activate

pip install -r requirements.txt
```

只想跑展示用的 web app 才需要：

```bash
pip install -r requirements-demo.txt
```

跑測試：

```bash
pytest tests/ -q
```

---

## 模組總覽

| 模組 | 說明 |
|---|---|
| `src/data/` | 前處理（分句、濾短句、CSV/HF → JSONL） |
| `src/features/` | TF-ISF、句長、句位置、TextRank 中心性 |
| `src/representations/` | TF-IDF 向量與相似度矩陣 |
| `src/models/extractive/` | Greedy(MMR)、GRASP、NSGA-II、encoder 排序 |
| `src/pipeline/` | 特徵組合、候選池、optimizer dispatch、選句、評估 |
| `src/eval/` | ROUGE（Lsum + multi-reference）、greedy oracle reference |
| `src/selection/` | 長度控制與候選池工具 |

---

## 快速開始

前處理：

```bash
python -m src.data.preprocess --input data/raw/validation.csv --split validation \
  --out data/processed/validation.jsonl --max_sentences 25
```

選句：

```bash
python -m src.pipeline.select_sentences --config configs/1_Base_NSGA2.yaml \
  --split validation --input data/processed/validation.jsonl --run_dir runs
```

評估（ROUGE-1/2/Lsum）：

```bash
python -m src.pipeline.evaluate --pred runs/<run>/predictions.jsonl --gold data/processed/<dataset>_<split>.jsonl --out runs/<run>/metrics.csv --protocol multisentence_lsum
```

Greedy oracle reference（**不是** exact upper bound）：

```bash
python -m src.eval.oracle --input data/processed/multi_news_test.jsonl --max_words 245 --limit 300
```

---

## 稽核診斷腳本

`scripts/audit/` 底下的腳本用來檢查系統行為，不是產生論文結果：

| 腳本 | 用途 |
|---|---|
| `lead_vs_system.py` | 在同資料、同 evaluator 下比較某個 run 與本地 Lead baseline |
| `selection_diagnostics.py` | 選句位置分布、與 Lead 的重疊率、對 greedy reference 的命中率 |
| `dataset_headroom.py` | 各資料集在 Lead 之上還有多少空間、lead bias 強度 |
| `plm_timing.py` | 拆解 PLM 成本為模型載入 vs 推論 |

用法與已重現的輸出見 `scripts/audit/README.md`。

---

## 已知的實作限制

投稿前必須處理，詳見團隊內部重構計畫：

- `src/data/preprocess.py` 用正則分句，未處理縮寫與小數，會產生過長的偽句子
- `src/data/preprocess_scitldr.py` 已保留 SciTLDR 多個替代 reference；`scitldr_official` 評估在官方 wrapper 通過一致性測試前會拒絕執行
- `src/features/semantic.py` 的 `centrality` 與 `novelty` 數學上完全反相關，同時加權是退化的
- `graph_params.threshold` 只作用於候選池，未套用到 graph 特徵分數
- NSGA-II 的 importance 目標使用總和，與 coverage 一起把解推向長度上限

---

## 授權

尚未指定。在加入 LICENSE 之前，預設保留所有權利。
