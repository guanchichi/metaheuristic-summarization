# ACTION PLAN —— 到底要做什麼

> 這是**唯一的執行清單**。研究標準以 `paper_revision_plan_IEEE_Access.md` 為準；程式稽核與策略評估的結論全部收斂到這裡。
> 每天工作看這份就好，需要理由再回去翻對應的分析文件。
>
> 版本：2026-07-26 ｜ 進度標記：`[ ]` 未開始 `[~]` 進行中 `[x]` 完成 `[!]` 卡住

---

## 怎麼用這份文件

- 階段是**有順序的**，`Gate` 沒過就不要進下一階段
- 每個任務都有 **DoD（完成定義）** —— 沒達到就不算完成
- 標 🔴 的是**擋路項目**，不做完後面全部白做
- 標 ⏱️ 的是估時（單人工作天）

**總估時：8–12 週**。不建議壓縮，半套修改會再被拒一次。

### 與研究主計畫的對齊

本文件的 **Phase 1–6 編號刻意與 `paper_revision_plan_IEEE_Access.md` §15 完全一致**，
方便兩份文件交叉對照。差別只有：

| | `paper_revision_plan_IEEE_Access.md` §15 | 本文件 | 說明 |
|---|---|---|---|
| Phase −1 | §15（已補入） | 決策與凍結 | legacy Multi-News 診斷顯示舊方法未勝 Lead，先做研究路線決策 |
| Phase 0 | 專案治理與可重現性整理 | 專案整理 | ✅ 相同；清理動作仍須另行確認 |
| Phase 1–6 | ✅ 相同 | ✅ 相同 | 內容以研究主計畫為主幹，插入實測衍生的必做項目 |

**分工原則**：研究主計畫負責「為什麼、要達到什麼標準」，本文件負責「今天做什麼、怎麼算完成」。
兩份有衝突時，以研究主計畫的 gate 為準；數字則一律回到 artifact、程式版本、資料 fingerprint 與 evaluator protocol 驗證，不能由本文件自行取得權威地位。

---

## Phase −1：決策與凍結 ⏱️ 1–2 天

> 動任何程式之前先做完這階段。這裡沒想清楚，後面全是白工。

### 必須先接受的三個事實

- [ ] 🔴 **接受 F-0 的正確範圍**：legacy Multi-News ExpB 沒有贏過同資料同 evaluator 的 Lead（R-2 −0.0048、R-Lsum −0.0021）；CNN/DM 尚無同 split 同 evaluator 的有效勝負
      → 研究主計畫 §6.1「連 Lead 都贏不過就停止投稿並重新設計」的 No-Go 條件**已經觸發**
- [ ] 🔴 **接受 P0-01**：`runs/tuning_experiments/` 全部 11 個 run 都是 test set 調出來的 → **既有結果全部作廢**
- [ ] 🔴 **接受病因假說**：系統選句 61.7% 與 Lead 重疊、22.8% 命中 legacy greedy reference
      → 重現腳本已版本化於 `scripts/audit/selection_diagnostics.py`，數字可重跑確認；
        但仍跑在 test-tuned legacy artifact 上、200 篇抽樣、非 official oracle，須在新 validation pipeline 重做

### 決策

- [ ] 選定研究路線（建議 **研究主計畫路線 A：方法型**）
      → 核心方法貢獻 = **打破候選池的 lead bias + provenance-aware fusion + budget-aware routing**
- [ ] 選定主 benchmark（首選 **GovReport + 原版 Multi-News**；Multi-News bad-retrieval-removed／Multi-News+ 作 paired data-quality sensitivity，PubMed 是需另作資料／成本 pilot 的替代）
      → CNN/DM 降為 sanity check、SciTLDR 降為 stress test
- [ ] 寫下 **Go/No-Go 條件**並簽字（依研究主計畫 §9）
- [ ] 寫下 **canonical method specification**：每條公式對應哪個函式（研究主計畫 §3.1 的產出）

### 凍結

- [ ] 對已核對的 legacy commit `1b9fe6f` 建立 annotated tag `legacy_ict_express`；不可把目前未提交修正誤標成 legacy
- [ ] 在 `runs/README.md` 寫明「以下結果為 test-set 污染，不得用於新論文」

**Gate −1**：路線、主 benchmark、Go/No-Go、method spec 四者都白紙黑字寫定。

---

## Phase 0：專案整理 ⏱️ 1 天

> 詳細指令見 `REPO_CLEANUP.md`。這階段純粹是降低後續的認知負擔。

- [ ] 歸檔 100 個 legacy configs、237 個 archived runs、35 個 archived scripts
- [ ] 將疑似死碼 `src/pipeline/build_features.py` 移入 legacy archive；先做入口與歷史重現檢查，不直接刪除
- [ ] `frontend/`、`backend/`、`experimental/`、`notebooks/` 移出研究主線（另開 repo 或標明與論文無關）
- [ ] 修 `requirements.txt`：`pymoo` 與 `scikit-learn` 各有重複且相容的宣告，需整理成單一明確 lock intent
- [ ] 把 `pytest` 納入 dev dependencies，建立可重現環境並讓完整 `tests/` 能跑

**Gate 0**：`ls configs/` 與 `ls runs/` 一眼看得懂哪些是現行的。

---

## Phase 1：正確性重構 ⏱️ 1–2 週 🔴

> 目標：讓每個數字都可被獨立驗證。這階段不追求分數。

### 1a. Patch 已套用，但尚未完成驗收

- [~] `src/eval/rouge.py` → 已改 ROUGE-Lsum、同一 reference 由最高 R1 選定、長度 mismatch fail；待 pytest 與 published-protocol parity
- [~] `src/eval/oracle.py` → 已有 greedy oracle reference；不得稱 exact upper bound，待加入／重現 SciTLDR official single-sentence oracle
- [~] `src/features/graph.py` → `.copy()` smoke test 通過；待正式 regression test suite
- [~] `src/models/extractive/encoder_rank.py` → 模型快取 patch 已加；待多文件、CPU/GPU、cache-clear 與效能測試
- [~] `src/pipeline/optimizer_dispatch.py` → 參數接線與 no-fallback smoke test 通過；待 pytest regression
- [x] `scripts/audit/` → 稽核診斷腳本已版本化（`lead_vs_system` / `selection_diagnostics` / `dataset_headroom` / `plm_timing`），
      由版本化位置重跑確認 F-0 與 61.7%／22.8% 數字完全一致；用法見 `scripts/audit/README.md`
      **仍是 diagnostic**：跑在 legacy artifact、內部 Lsum 協定、抽樣、未做 paired test

**1a 的共同驗收條件**（全部完成才能把上面的 `[~]` 改成 `[x]`）：

- [ ] `pip install pytest` 並讓 `tests/` 能跑
- [ ] 每個 patch 都有對應的 regression test（見 1e）
- [ ] SciTLDR official single-sentence oracle 重現 R1 ≈ 52.4

### 1b. 資料層

- [ ] 🔴 `preprocess_scitldr.py`：**停止串接 multi-reference**，`references` 存成 list
- [ ] 分句改用 NLTK punkt 或 spaCy
      （現況：正則分句造成 358/37349 個「句子」超過 80 字，最長 855 字）
- [ ] Multi-News 正確處理 `|||||` 分隔符與換行雜訊；修編碼（少數 U+FFFD）
- [ ] 從原始資料重建 Multi-News，保存每個 source document 的 boundary、source order 與 original-to-cleaned mapping；現有扁平 `sentences` 不得進正式實驗
- [ ] 下載並驗證 GovReport 官方資料：split、row count、checksum、license、section metadata 與異常列規則
- [ ] 將 `max_words / max_sentences / max_model_tokens / candidate_budget / compute_budget` 拆成不同欄位；不得再以 `max_tokens` 混用
- [ ] 重建 **CNN/DM 官方 split**：train 287,113 / validation 13,368 / **test 11,490**
- [ ] 資料健檢報告：筆數、空文件、句長分布、reference 數、checksum
      （已知：Multi-News test 有 1 筆零句文件、`test_4241` 有 3,295 句）

### 1c. 候選生成重構 🔴 這是核心

- [ ] 🔴 **lexical、semantic、sparse graph/structure 三路各自在完整輸入上獨立排名**，不可先被共同候選池截斷
- [ ] 🔴 **候選池多來源聯集**：各 route 先獨立 top-K，再以固定 total candidate budget 合併
- [ ] 🔴 加入 position／document／section strata coverage guard；它們不是第四個語意 route
- [ ] candidate record 保留 `sentence_id / original_index / document_id / section_id / route raw score / rank / percentile / model revision / route cost`
- [ ] K 必須在 rank 排序階段截取，不是按原文位置

### 1d. 路由與融合層

- [ ] semantic route 使用明確 sentence-encoder checkpoint、一次載入、batch encode，記錄 `max_model_tokens` 與截斷率
- [ ] graph route 改為有界 sparse kNN／block graph；不得以 dense `N×N` 作長文件預設
- [ ] 先以 rank percentile／reciprocal-rank fusion／route agreement 融合，不直接平均不可比分數
- [ ] 缺失或失敗 route 記為 `null + reason` 並使正式 run fail；不得填 0 或靜默 fallback
- [ ] route budget allocator 第一版只能使用 inference-time 可得的廉價特徵，threshold 在 validation 凍結

### 1e. Objective 與 selector contract

- [ ] 由 task profile 啟用 objective：單句關閉 redundancy 與 subset NSGA-II；多文件才啟用 document-group coverage
- [ ] salience、coverage、redundancy 統一方向與尺度；sum／mean／length-normalized aggregation 由 validation pilot 決定，不得讓集合大小隱性支配
- [ ] hard constraint 明確處理 `min_words / max_words / max_sentences`，空集合不是合法摘要
- [ ] deterministic greedy／MMR 與 NSGA-II 使用完全相同 candidates、objectives 與 constraints

### 1f. 測試

- [ ] TF-ISF / length / position 的手算 golden tests
- [ ] `rougeL` vs `rougeLsum` golden test
- [ ] SciTLDR 官方 reference aggregation golden test：以最高 R1 選一個 reference，R2/RL 必須沿用同一 reference
- [ ] Graph：diagonal、threshold、dangling node
- [ ] 候選 top-K rank 保序測試
- [ ] canonical schema 保存 Multi-News document boundaries；route failure 不得被 0 分掩蓋
- [ ] task-profile matrix 測試：single sentence 不得建立 redundancy objective 或呼叫 subset NSGA-II
- [ ] NSGA-II 參數傳遞、seed 決定性、**no-fallback** 測試
- [ ] 10 篇 toy pipeline snapshot test

**Gate 1**：所有手算測試通過；同 seed 重跑得到相同 indices；故意移除 pymoo 時 run 必須 fail。

---

## Phase 2：Baseline 與 reality check ⏱️ 1 週 🔴

> **這階段的唯一目的：確認新架構真的比 Lead 好。沒過就不要往下走。**

- [ ] 🔴 **先跑 Lead** —— 最便宜的 reality check，優先於一切
      - CNN/DM：Lead-3
      - GovReport：同 word budget 的 Lead
      - Multi-News：同 word budget 的 Lead
      - SciTLDR：Lead-1
- [ ] TextRank、LexRank（本地執行同 pipeline；優先重用官方／可信實作並鎖版本）
- [ ] PacSum
- [ ] Sentence-BERT centroid + MMR
- [ ] Random（固定 seed）
- [ ] Exact extractive oracle（可行時）或明確標示的 greedy reference（不可稱 upper bound）
- [ ] SciTLDR **evaluator conformance test**：使用官方 `files2rouge` wrapper、單句限制，先以最大 R1 選定同一 reference，再報該 reference 的 R1/R2/RL；重現 official oracle R1 ≈ 52.4

**Gate 2**：所有 baseline 在同一 evaluator 下跑出合理數字；SciTLDR 官方 oracle 重現成功。

---

## Phase 3：方法開發（只用 validation） ⏱️ 1–2 週 🔴

> 🚫 **這階段絕對不准碰 test set。**

### 3a. 先看兩個先行指標（比 ROUGE 更早給訊號）

- [ ] 🔴 **候選池對 validated oracle／greedy reference 的 recall@K** —— legacy 值為 **22.8%**，須以 validated oracle 重做
- [ ] 🔴 **選句位置分布與 Lead 的重疊率** —— legacy 值為 **61.7%**

量測工具已存在，改完架構後直接對新 run 重跑同一支腳本即可比較：

```bash
.venv/Scripts/python.exe -m scripts.audit.selection_diagnostics \
  --data <validation.jsonl> --pred runs_v2/<new_run>/predictions.jsonl --budget 245 --limit 200
```

> ### ⚠️ 中途檢查點（最重要的一個）
> 61.7% 與 22.8% 是 legacy baselines。新 validation pipeline 應降低非必要的 lead overlap、
> 提升 validated oracle-candidate recall；若兩者都沒有改善，視為**強烈 redesign 訊號**
> —— 但這是經驗判斷，不是「ROUGE 必然不改善」的數學定理。

### 3b. 方法設計

- [ ] Provenance-preserving fusion（用 route rank/score，不是只看有沒有進 union）
- [ ] Budget-aware adaptive routing（依文件特性決定要不要啟用 PLM）
- [ ] 重新設計 `position` 特徵（legacy greedy-reference 位置中位數探索值為 **0.46**；須在 validation 重做）
- [ ] Objective 正規化：`imp` 不再使用未正規化總和；平均或 length-normalized aggregation 由 validation pilot 決定
- [ ] 依 `ARCHITECTURE.md` 跑 task-profile/objective 啟用矩陣，不按 dataset 名稱偷換公式
- [ ] 明確的 Pareto 選解規則（knee point / reference point，權重只能在 validation 定）

### 3c. Validation 實驗

- [ ] K、graph threshold τ、fusion weights、population/generation 的 sensitivity
- [ ] Optimizer isolation：固定 features/candidates/budget，只換 Greedy / GRASP / NSGA-II
- [ ] Ablation：No-statistical / No-graph / No-PLM / No-provenance / No-routing
- [ ] Route utility：各路 unique candidate recall、quality delta、latency 與 peak memory；無增量效果的 route 刪除
- [ ] 原版 Multi-News 與 bad-retrieval-removed／Multi-News+ 的 paired data-quality sensitivity，不混用 split

**Gate 3** 🔴：
- 在 validation 上，至少一個主 benchmark明顯勝過強 no-task-training baseline；另一個至少 non-inferior 或形成預先定義的 cost Pareto 優勢
- candidate recall 與 lead-overlap 診斷可解釋，且至少一條非 lexical route 有可重現的獨立效益；不要求為了好看而機械式降低 lead overlap
- NSGA-II 只有在 matched-condition 下提供穩定增益、Pareto/hypervolume 優勢或有用 operating point 才留在核心；否則移出標題與主方法
- data schema、budget semantics、objective matrix、route set 與 output policy 全部通過 `ARCHITECTURE.md` freeze gate
- → 通過才 **freeze config**，解鎖 test

---

## Phase 4：正式 test ⏱️ 約 1 週計算

- [ ] 全 dataset、全 seeds（≥5，建議 10）、一次性執行
- [ ] Paired bootstrap（≥10,000 resamples）、95% CI、Holm correction
- [ ] Runtime / memory：模型只載入一次，分開報 cold-start 與 warmed inference
- [ ] Quality–latency Pareto 圖（**用完整 pipeline 成本**，不是單一元件）
- [ ] 產生 immutable artifacts（config hash、commit、data fingerprint）

**Gate 4**：Go / No-Go 決策（研究主計畫 §9）。

---

## Phase 5：分析與寫作 ⏱️ 1–2 週

- [ ] Candidate analysis：各 route 的 recall@K、overlap、unique contribution
      → 這組實驗直接回答「三軌到底互不互補」
- [ ] Qualitative error analysis（成功/失敗各 ≥3 例，選例規則預先定義）
- [ ] BERTScore
- [ ] （加分）Human evaluation 50–100 篇 × 3 人
- [ ] 依研究主計畫 §11 的骨架重寫論文
- [ ] Reviewer response matrix：四位審稿人每一條意見逐項對應
- [ ] Conference extension table（ICACT → IEEE Access 新增了什麼）

---

## Phase 6：投稿前稽核 ⏱️ 2–3 天

- [ ] Equation ↔ code ↔ config ↔ result 全鏈可追溯
- [ ] 從乾淨環境一鍵重現
- [ ] IEEE Access 合規：引用 ICACT、similarity < 35%、AI 揭露、ORCID、biography
- [ ] 文法校對
- [ ] 對照研究主計畫 §16 的最終檢查表逐項打勾

---

## 論文主張紅線（寫作時隨時對照）

### 🚫 不可以寫

- meta-heuristics outperform LM-based methods
- significantly outperforms all extractive baselines（**目前實測不成立**）
- converge to the global optimum
- 3×–170× speedup 概括完整 pipeline
- primary innovation（graph 只是 thresholded TextRank）
- 任何**抄來的 baseline 數字**

### ✅ 可以寫

- under a fixed no-task-training protocol
- provides a statistically supported quality–cost trade-off
- graph centrality acts as a structural complementary signal
- 可審計的 candidate provenance（相對一般端到端 baseline 更直接，但不能宣稱 neural 方法做不到）
- 顯式、可重現的多目標 trade-off 控制
- 以分句／稀疏圖／routing 避免單次全文 512-token 限制；仍須報 sentence encoder 截斷與實測 scaling

---

## 如果 Gate 3 沒過怎麼辦

**不要硬投。** 兩個選項：

1. **改寫成 empirical study / negative result**
   「在 news 領域，lead bias 使 unsupervised 選句方法難以超越 Lead」
   —— 配合 headroom 分析與 oracle gap，這是有價值的發現，但必須有普遍性結論
2. **換到 lead bias 弱的領域重來**（GovReport / PubMed / 法律 / 醫療長文件）

**最糟的選擇是：只修 evaluator、補幾個 baseline、改寫文字就投。**
那會更清楚地顯示輸給 Lead，而且這次是帶著公開程式碼被抓。

---

## 進度追蹤

| Phase | 狀態 | Gate 通過 | 備註 |
|---|---|---|---|
| −1 決策與凍結 | `[ ]` | | |
| 0 專案整理 | `[ ]` | | |
| 1 正確性重構 | `[~]` | | 1a 僅 patch/smoke，未達 DoD |
| 2 Baseline | `[ ]` | | |
| 3 方法開發 | `[ ]` | | 🔴 中途檢查點在這 |
| 4 正式 test | `[ ]` | | |
| 5 分析寫作 | `[ ]` | | |
| 6 投稿稽核 | `[ ]` | | |
