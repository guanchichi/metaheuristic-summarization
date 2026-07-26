# Changelog

All notable changes to the `metaheuristic-summarization` project will be documented in this file.

> ⚠️ **Entries before v0.5.0 contain performance claims that a later audit
> invalidated.** They are kept for history, with corrections noted inline.
> See `docs/research/CODE_AUDIT_IEEE_Access.md`.

## [v0.5.0] - 2026-07-26 (Correctness refactor)
### Fixed
- **ROUGE protocol**: `src/eval/rouge.py` now uses `rougeLsum` for multi-sentence
  summaries (the old single-sequence `rougeL` under-scores extracts), applies the
  same segmentation to prediction and reference, and selects one reference by
  max ROUGE-1 for multi-reference data instead of concatenating references.
- **Similarity matrix corruption**: `src/features/graph.py` copied before
  thresholding; it previously mutated the caller's matrix in place, silently
  truncating the similarities NSGA-II used for coverage/redundancy.
- **Encoder reloading**: `src/models/extractive/encoder_rank.py` caches the
  tokenizer/model instead of calling `from_pretrained` once per document.
- **Unwired hyper-parameters**: `pop_size` / `n_gen` / `seed` are now read from
  config. Every previous NSGA-II run silently used the defaults (100/100)
  regardless of what the YAML declared.
- **Silent fallback removed**: `optimizer_dispatch.py` raises instead of quietly
  running greedy when NSGA-II is unavailable or the method is unknown.

### Added
- `src/eval/oracle.py` — greedy extractive oracle reference (not an exact upper bound).
- `scripts/audit/` — versioned diagnostics: local Lead comparison, selection-position
  analysis, per-dataset headroom, PLM load-vs-inference timing.
- `docs/research/` — audit findings, revision plan, target architecture, action plan.
- Guards on `scripts/quick_tune*.ps1` and `run_missing_experiments.ps1`, which tune
  on the test set.

## [v0.4.0] - 2026-01-14
### Added
- **3-Way Fusion Architecture**: Implemented a multi-view fusion pipeline combining Statistical (Base), Semantic (LLM), and Structural (Graph) scores.
- **Stage 1 Graph**: Added `src.features.graph` module implementing TextRank (PageRank) algorithm.
- **NSGA-II Integration**: Upgraded Stage 1 Base optimizer from `greedy` to `nsga2` for better candidate selection.
- ~~**Multi-News Benchmark**: Achieved **ROUGE-1: 44.32** on Multi-News, surpassing the HiMAP benchmark (44.17).~~
  > 🔴 **RETRACTED.** Three problems: (1) 44.32 does not correspond to any surviving
  > artifact — the runs in `runs/` give 43.52 (ExpB) and 43.37 (full benchmark);
  > (2) the configuration was selected using the **test set**, so the number is a
  > test-tuned artifact, not a valid result; (3) "surpassing HiMAP" compares against
  > a number from another paper computed with a different evaluator and preprocessing.
  > A local, ID-matched Lead baseline under the same evaluator scores
  > 0.4331 / 0.1453 / 0.3901 versus the system's 0.4352 / 0.1405 / 0.3880 —
  > i.e. the system does **not** beat Lead on ROUGE-2 or ROUGE-Lsum.
- ~~**Final Configs**: Standardized best-performing configurations in `configs/final/`.~~
  > That directory no longer exists; the current settings are in `configs/` (see its README).

### Changed
- **Pipeline Update**: `scripts/build_union_stage2.py` now supports 3 inputs: `--base_pred`, `--bert_pred`, and `--graph_pred`.
  > Note: that script now lives in `scripts/_archive/` and is excluded from version control.
- **Optimization**: Tuned `max_tokens` to 245 and `lambda_coverage` to 2.5 for Multi-News dataset.
  > Note: this tuning was performed on the test set (P0-01). `max_tokens` also counts
  > whitespace words, not model tokens.

## [v0.3.0] - 2026-01-10
### Added
- **Multi-Objective Optimization**: Initial implementation of `nsga2` and `fast_nsga2` in Stage 2.
- **Extractive Pipeline**: Built core `select_sentences.py` with modular feature scoring.

## [v0.2.0] - 2025-12-10
### Added
- **Feature Correlation**: Scripts to analyze feature importance.

## [v0.1.0] - 2025-11-28
### Initial Release
- Basic Greedy optimizer.
- TF-ISF scoring.
