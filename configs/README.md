# configs — 實驗設定

> ⚠️ 除 `phase1_mvp_multinews.yaml` 外，**其餘都是 legacy 設定。** 它們產生的結果全部是在 test set 上調參得到的（見
> `docs/research/CODE_AUDIT_IEEE_Access.md` 的 P0-01），**不可用於新論文**。
> 保留它們只為了重現 `runs/` 底下的既有 artifact。
> 新架構以 `phase1_mvp_*.yaml` 命名，且只允許 validation pilot；schema 以 `docs/research/ARCHITECTURE.md` 為準。

歷史設定（`_legacy_archive/`）已排除於版本庫外，只存在原作者本機。

---

## Phase 1 validation MVP

| 檔案 | 用途 | 尚未完成 |
|---|---|---|
| `phase1_mvp_multinews.yaml` | canonical Multi-News 上隔離 lexical + pinned sentence encoder，固定 route/total budget、RRF 與 document guard，deterministic greedy selector | pinned PLM 與 3-row wiring smoke 已過；尚缺正式 cost、baseline reality check、route unique recall、output budget freeze，未過 gate 前不可跑 test |

此設定刻意不含 graph 與 NSGA-II：先確認 lexical + semantic MVP 能否勝過同協定 Lead／PacSum，再決定是否擴張完整架構。

---

## 兩階段流程對照

Stage 1 三軌各跑一次 `select_sentences.py`，輸出各自的 `predictions.jsonl`；
再用 `scripts/utils_fusion.py` 取聯集，Stage 2 在聯集上做最終選句。

| 檔案 | 階段 | optimizer | 說明 |
|---|---|---|---|
| `1_Base_NSGA2.yaml` | Stage 1a | `nsga2` | 統計特徵（TF-ISF 0.8 / 句長 0.2 / 句位 0.2），候選 K=40 |
| `1_LLM_BERT.yaml` | Stage 1b | `bert` | `bert-base-uncased` 重心相似度排序，候選 K=40 |
| `1_Graph_TextRank.yaml` | Stage 1c | `greedy` | 純圖中心性（graph 權重 1.0），候選來源 `[graph]` |
| `2_Fusion_Final.yaml` | Stage 2 | `fast_nsga2` | 當家融合設定 |

### Stage 2 的權重掃描

四個設定只差在 objective 權重與融合權重，其餘相同（`max_tokens: 245`、
`max_sentences: 30`、`pop_size: 100`、`n_gen: 100`、`seed: 2024`）：

| 檔案 | λ_coverage | λ_redundancy | w_base | w_bert |
|---|---|---|---|---|
| `2_Fusion_Final.yaml` | 2.5 | 1.2 | 0.5 | 0.5 |
| `2_Fusion_ExpA.yaml` | 2.5 | 1.2 | 0.3 | **0.7** |
| `2_Fusion_ExpB.yaml` | **4.0** | 1.2 | 0.5 | 0.5 |
| `2_Fusion_ExpC.yaml` | 2.5 | **2.0** | 0.5 | 0.5 |
| `2_Fusion_NoNsga2.yaml` | 2.5 | 1.2 | 0.5 | 0.5 | ← optimizer 改用 `fast_fused`（MMR），NSGA-II ablation |

> 🔴 **`w_bert` 的命名是錯的。** 在 `src/models/extractive/fast_fused.py` 裡，
> 它加權的是 **TF-IDF centroid 相似度**，不是 BERT 分數。Stage 2 整條路徑沒有任何 PLM。
> 這是「移除 BERT 只掉 0.0005 ROUGE-1」的真正原因（F-3）。
> 程式已接受 `w_plm` 作為別名，但語義問題要等重構才解決。

---

## 消融／測試設定

| 檔案 | 變動的單一因素 |
|---|---|
| `test_v2_features.yaml` | 啟用 v2 特徵（TF-ISF 加停用詞與 sublinear TF、position 改 inverse、fusion 加交互項），並開 centrality/novelty |
| `test_v2_tfisf_only.yaml` | 只把 TF-ISF 換成 v2，其餘維持 v1 |
| `test_coverage_set.yaml` | NSGA-II 的 coverage 改用 `set`（貪婪子模） |
| `test_coverage_diversity.yaml` | NSGA-II 的 coverage 改用 `diversity`（coverage 減冗餘懲罰） |

> ⚠️ `centrality` 與 `novelty` 在數學上完全反相關（`centrality_norm = 1 - novelty_norm`），
> 同時給獨立權重是退化的 —— `test_v2_features.yaml` 的那兩個權重實際只有一個自由度。

---

## 使用方式

```bash
python -m src.pipeline.select_sentences \
  --config configs/1_Base_NSGA2.yaml \
  --split validation \
  --input data/processed/<split>.jsonl \
  --run_dir runs \
  --stamp my-run
```

`--optimizer` 可覆寫 config 裡的 `optimizer.method`。

---

## 設定鍵值說明

| 鍵 | 說明 |
|---|---|
| `objectives.lambda_*` | 從 Pareto front 選解時的加權（importance / coverage / redundancy） |
| `objectives.coverage_method` | `max`（預設）/ `set` / `diversity` |
| `features.weights.*` | 各特徵在 base score 的權重 |
| `features.{tf_isf,position,fusion}.version` | `v1`（預設）或 `v2` |
| `graph_params.threshold` | 圖的邊剪枝閾值 τ |
| `candidates.k` / `sources` / `mode` | 候選池大小、來源（`score`/`position`/`centrality`/`graph`）、`hard`/`soft` |
| `candidate_budget.per_route / total` | 新架構的每 route quota 與 selector 最終固定候選數，兩者不可混用 |
| `compute_budget.mode / enabled_routes` | 目前只實作 validation-frozen `fixed`；宣告 `adaptive` 會直接失敗 |
| `routes.semantic.*` | sentence encoder 名稱、固定 revision、batch size、`max_model_tokens`；輸出記錄實際 revision 與截斷率 |
| `routes.graph.*` | 新 graph route 預設 `sparse_knn`；`dense_legacy` 必須明確指定 |
| `coverage_guard.*` | document／section／position strata 保留規則，不屬於語意 route |
| `length_control.unit` | `words`、legacy `tokens`（實為**空白切詞**）或 `sentences`；word budget 不再繞過 selector |
| `optimizer.{method,pop_size,n_gen}` | 選句器與 NSGA-II 參數 |
| `seed` | 全域亂數種子 |

> ⚠️ **`unit: tokens` 其實數的是空白切詞，不是 model token。**
> 新架構會把 `max_words` / `max_sentences` / `max_model_tokens` 拆成不同欄位
> （見 `docs/research/ARCHITECTURE.md` §1.2）。
>
> ℹ️ `pop_size` / `n_gen` / `seed` 過去**未被程式讀取**（實際一律跑 100/100），
> 已在 `optimizer_dispatch.py` 修好。因此 legacy run 的 config_used.json
> 不代表當時真正生效的參數。
