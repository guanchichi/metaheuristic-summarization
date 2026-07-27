"""Effective-parameter audit for configs/*.yaml and the selection pipeline.

This answers one bounded question: for the ``optimizer.method`` a config
declares, which selection-related keys can affect that run through either the
shared objective factory, CLI governance, or the matching optimizer branch?

This is a **read-only, non-blocking, reporting-only** check:
    * it does not modify configs/ or src/
    * it is not wired into CI
    * it always exits 0 -- it is not a gate

Scope
-----
The key -> source map below (``METHOD_KEY_MAP``) was hand-transcribed from
``select_sentences.py``, ``objectives/factory.py``, and
``optimizer_dispatch.py``. It is protected by regression tests because a
branch-only map becomes stale whenever shared pipeline wiring changes.

It covers selection/governance sections used by the formal entry point:
``experiment.*``, ``objectives.*``, ``optimizer.{pop_size,n_gen}``, top-level
``seed``, ``bert.*``, ``fusion.*``, ``redundancy.lambda``, ``grasp.*``.
Everything else in a config (``graph_params``, ``features``,
``representations``, ``candidates``, ``candidate_budget``, ``routes``,
``coverage_guard``, ``compute_budget``, ``selector``, ``length_control``,
...) is consumed by *other* modules (``feature_builder.py``,
``candidate_builder.py``, ``select_sentences.py``) and is intentionally
out of this bounded audit. A future schema gate must cover those sections.

``optimizer.method`` itself is deliberately excluded from the scoped-key
checks: it is the dispatch key read by ``select_sentences.py`` to pick a
branch, not a leaf parameter any branch reads from within
``dispatch_optimizer()``.

One key needs a special note: ``objectives.importance_aggregation`` is not
read via a plain ``cfg.get()`` inside this file -- it arrives through
``objective_spec`` (built by ``src/objectives/factory.py`` from
``cfg["objectives"]`` AND the input document's ``task_profile``). Greedy,
GRASP, and NSGA-II all consume that shared evaluator. The ``fast_nsga2`` branch never references
``objective_spec``, so this key is a silent no-op there even though
``fast_nsga2`` does wire up ``objectives.lambda_*``. Because the *default*
that would actually apply also depends on data (whether the input document
has a ``task_profile``), this script can only report the config-declared
value or its static fallback -- it cannot know at audit time whether a
run would hit the profiled or legacy_unprofiled path. See
``docs/research/CODE_AUDIT_IEEE_Access.md`` F-14.

Usage
-----
    python -m scripts.audit.config_key_audit
    python -m scripts.audit.config_key_audit --configs "configs/*.yaml"
    python -m scripts.audit.config_key_audit --configs configs/1_Base_NSGA2.yaml configs/2_Fusion_Final.yaml
    python -m scripts.audit.config_key_audit --json
"""
from __future__ import annotations

import argparse
import glob
import json
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Ground truth: which cfg keys can affect each formal selection path?
# Hand-transcribed from the pipeline/objective/dispatch call chain, read 2026-07-27.
# If that file changes, re-read it and update this table by hand -- do not
# regenerate it from configs or docs.
#
# Each entry: (dotted key path, default-as-shown-string, source line, note)
# ---------------------------------------------------------------------------
KeySpec = Tuple[str, str, str, str]

SHARED_OBJECTIVE_KEYS: List[KeySpec] = [
    (
        "objectives.lambda_importance",
        "1.0",
        "src/objectives/factory.py:128",
        "consumed through objective_spec and the shared evaluator",
    ),
    (
        "objectives.lambda_coverage",
        "0.8",
        "src/objectives/factory.py:129",
        "consumed through objective_spec and the shared evaluator",
    ),
    (
        "objectives.lambda_redundancy",
        "0.7",
        "src/objectives/factory.py:130",
        "consumed through objective_spec and the shared evaluator",
    ),
    (
        "objectives.coverage_method",
        "max",
        "src/objectives/factory.py:107",
        "consumed through objective_spec and the shared evaluator",
    ),
    (
        "objectives.importance_aggregation",
        "mean for profiled multi-sentence; sum for legacy_unprofiled",
        "src/objectives/factory.py:44-55,101",
        "effective default depends on the input task profile; see F-14",
    ),
]

GLOBAL_SEED: KeySpec = (
    "seed",
    "None",
    "src/pipeline/select_sentences.py:set_global_seed",
    "the formal CLI seeds global RNGs before dispatch",
)

METHOD_KEY_MAP: Dict[str, List[KeySpec]] = {
    "greedy": [
        *SHARED_OBJECTIVE_KEYS,
        GLOBAL_SEED,
    ],
    "grasp": [
        *SHARED_OBJECTIVE_KEYS,
        ("grasp.iters", "10", "optimizer_dispatch.py:63", ""),
        ("grasp.rcl_ratio", "0.3", "optimizer_dispatch.py:64", ""),
        GLOBAL_SEED,
    ],
    "nsga2": [
        *SHARED_OBJECTIVE_KEYS,
        ("optimizer.pop_size", "100", "optimizer_dispatch.py:93", ""),
        ("optimizer.n_gen", "100", "optimizer_dispatch.py:94", ""),
        GLOBAL_SEED,
    ],
    "bert": [
        (
            "bert.model_name",
            "None",
            "optimizer_dispatch.py:103",
            "if absent, falls back to a method-specific hardcoded name "
            "(bert-base-uncased / roberta-base / xlnet-base-cased), see "
            "optimizer_dispatch.py:104-105 -- not a single constant default",
        ),
        ("bert.device", "None", "optimizer_dispatch.py:110", ""),
        ("bert.batch_size", "16", "optimizer_dispatch.py:111", ""),
        ("bert.max_model_tokens", "256", "optimizer_dispatch.py:112", ""),
        ("bert.revision", "None", "optimizer_dispatch.py:113", ""),
        GLOBAL_SEED,
    ],
    "fast": [
        ("fusion.w_base", "0.5", "optimizer_dispatch.py:118", ""),
        (
            "fusion.w_bert",
            "0.5",
            "optimizer_dispatch.py:119",
            "fusion.w_plm is NOT recognized on this branch -- only the "
            "fast_nsga2 branch accepts w_plm as an alias",
        ),
        ("redundancy.lambda", "0.7", "optimizer_dispatch.py:120", ""),
        GLOBAL_SEED,
    ],
    "fast_grasp": [
        ("fusion.w_base", "0.5", "optimizer_dispatch.py:129", ""),
        (
            "fusion.w_bert",
            "0.5",
            "optimizer_dispatch.py:130",
            "fusion.w_plm is NOT recognized on this branch",
        ),
        ("redundancy.lambda", "0.7", "optimizer_dispatch.py:131", ""),
        (
            "grasp.iters",
            "15",
            "optimizer_dispatch.py:136",
            "default differs from the plain 'grasp' method's default of 10",
        ),
        ("grasp.rcl_ratio", "0.3", "optimizer_dispatch.py:137", ""),
        GLOBAL_SEED,
    ],
    "fast_nsga2": [
        ("fusion.w_base", "0.5", "optimizer_dispatch.py:143", ""),
        (
            "fusion.w_plm",
            "(falls back to fusion.w_bert, see next row)",
            "optimizer_dispatch.py:150",
            "the only branch that recognizes w_plm at all",
        ),
        (
            "fusion.w_bert",
            "0.5",
            "optimizer_dispatch.py:150",
            "used only as the fallback value when fusion.w_plm is absent",
        ),
        ("objectives.lambda_importance", "1.0", "optimizer_dispatch.py:157", ""),
        ("objectives.lambda_coverage", "0.8", "optimizer_dispatch.py:158", ""),
        ("objectives.lambda_redundancy", "0.7", "optimizer_dispatch.py:159", ""),
        ("optimizer.pop_size", "100", "optimizer_dispatch.py:160", ""),
        ("optimizer.n_gen", "100", "optimizer_dispatch.py:161", ""),
        GLOBAL_SEED,
        # NOTE: unlike nsga2, this branch (optimizer_dispatch.py:141-163)
        # never reads objectives.coverage_method and never references
        # objective_spec (so objectives.importance_aggregation is also a
        # no-op here). See KNOWN_INAPPLICABLE below.
    ],
}

# Methods that share another method's branch verbatim in optimizer_dispatch.py.
METHOD_KEY_MAP["roberta"] = METHOD_KEY_MAP["bert"]  # optimizer_dispatch.py:101
METHOD_KEY_MAP["xlnet"] = METHOD_KEY_MAP["bert"]  # optimizer_dispatch.py:101
METHOD_KEY_MAP["fast_fused"] = METHOD_KEY_MAP["fast"]  # optimizer_dispatch.py:116
METHOD_KEY_MAP["tfidf_fused"] = METHOD_KEY_MAP["fast"]  # optimizer_dispatch.py:116

# Keys that are meaningful for SOME method but explicitly not read on a given
# method's branch -- named here so the report can explain *why* instead of
# just inferring "not in the read list".
KNOWN_INAPPLICABLE: Dict[str, List[Tuple[str, str]]] = {
    "fast_nsga2": [
        (
            "objectives.coverage_method",
            "consumed by the canonical shared evaluator, but "
            "'fast_nsga2' (optimizer_dispatch.py:141-163) never reads it",
        ),
        (
            "objectives.importance_aggregation",
            "consumed by canonical greedy/GRASP/NSGA-II via objective_spec; 'fast_nsga2' "
            "(optimizer_dispatch.py:141-163) never references "
            "objective_spec at all",
        ),
    ],
}

# Sections covered by this audit, and which leaf keys under each section are
# recognized by ANY audited path. A leaf key present in a
# config under one of these sections but NOT in this set is not read by
# an audited path currently coded -- most likely a
# typo (e.g. "pop_zise") or a parameter this script's map doesn't know
# about yet.
SCOPED_SECTIONS: Dict[str, set] = {
    "experiment": {"status", "dataset"},
    "objectives": {
        "lambda_importance",
        "lambda_coverage",
        "lambda_redundancy",
        "coverage_method",
        "importance_aggregation",
    },
    # "method" deliberately excluded: it's the dispatch key read by
    # select_sentences.py, not a leaf parameter of any dispatch_optimizer()
    # branch.
    "optimizer": {"pop_size", "n_gen"},
    "bert": {"model_name", "device", "batch_size", "max_model_tokens", "revision"},
    "fusion": {"w_base", "w_bert", "w_plm"},
    "redundancy": {"lambda"},
    "grasp": {"iters", "rcl_ratio"},
}
TOP_LEVEL_SCALAR_KEYS = {"seed"}

# These keys gate data access and dataset identity before selector dispatch.
# They are optional for legacy reproduction configs, so they should not appear
# in the "missing -> default" section, but declarations must never be reported
# as dead parameters.
ALWAYS_CONSUMED_IF_DECLARED = {
    "experiment.status",
    "experiment.dataset",
}

# Keys under a scoped section that are known and fine to declare but are
# deliberately never checked here -- not typos, just out of scope for this
# script's "is it read by dispatch_optimizer()" question.
EXCLUDED_KEYS = {
    # the dispatch key itself, read by select_sentences.py to choose a
    # branch -- not a leaf parameter any dispatch_optimizer() branch reads.
    "optimizer.method",
}


def _get_nested(cfg: Dict[str, Any], dotted: str) -> Tuple[bool, Any]:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _method_of(cfg: Dict[str, Any]) -> Optional[str]:
    found, value = _get_nested(cfg, "optimizer.method")
    if not found or value is None:
        return None
    return str(value).strip().lower()


def audit_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        return {
            "file": path,
            "error": f"top-level YAML is not a mapping (got {type(cfg).__name__})",
        }

    method = _method_of(cfg)
    method_known = method in METHOD_KEY_MAP if method is not None else False
    read_paths = {spec[0] for spec in METHOD_KEY_MAP.get(method, [])} if method_known else set()

    # --- Category A: declared keys this method's branch will not read ---
    declared_unread: List[Dict[str, str]] = []
    for section, known_leaves in SCOPED_SECTIONS.items():
        found, node = _get_nested(cfg, section)
        if not found or not isinstance(node, dict):
            continue
        for leaf in node:
            dotted = f"{section}.{leaf}"
            if dotted in EXCLUDED_KEYS:
                continue
            if dotted in ALWAYS_CONSUMED_IF_DECLARED:
                continue
            if dotted in read_paths:
                continue
            if leaf not in known_leaves:
                declared_unread.append({
                    "key": dotted,
                    "reason": (
                        "not a recognized dispatch_optimizer() parameter under "
                        "ANY method known to this script -- possible typo, or "
                        "a parameter this script's map does not know about yet"
                    ),
                })
                continue
            # Recognized leaf, just not read for *this* method.
            inapplicable_notes = dict(KNOWN_INAPPLICABLE.get(method or "", []))
            reason = inapplicable_notes.get(dotted)
            if reason is None:
                other_methods = sorted(
                    m for m, specs in METHOD_KEY_MAP.items()
                    if any(s[0] == dotted for s in specs) and m != method
                )
                reason = (
                    f"only read on method(s) {other_methods}; not read for "
                    f"method={method!r}" if other_methods else
                    "not read by any currently-coded method's branch"
                )
            declared_unread.append({"key": dotted, "reason": reason})

    if "seed" in cfg and "seed" not in read_paths:
        other_methods = sorted(
            m for m, specs in METHOD_KEY_MAP.items()
            if any(s[0] == "seed" for s in specs) and m != method
        )
        declared_unread.append({
            "key": "seed",
            "reason": (
                f"optimizer_dispatch.py's method={method!r} branch does not "
                f"call cfg.get('seed') itself (only read on method(s) "
                f"{other_methods}); note this is scoped to dispatch_optimizer() "
                "only -- select_sentences.py may still use seed elsewhere for "
                "global RNG seeding, which is out of scope for this script"
            ),
        })

    # --- Category B: keys this method's branch reads but config omits ---
    missing_default: List[Dict[str, str]] = []
    for path_, default, source, note in METHOD_KEY_MAP.get(method, []):
        found, _ = _get_nested(cfg, path_)
        if not found:
            missing_default.append({
                "key": path_,
                "default": default,
                "source": source,
                "note": note,
            })

    return {
        "file": path,
        "method": method,
        "method_known": method_known,
        "declared_unread": declared_unread,
        "missing_uses_default": missing_default,
    }


def _print_human(report: Dict[str, Any]) -> None:
    print(f"=== {report['file']} ===")
    if "error" in report:
        print(f"  ERROR: {report['error']}")
        print()
        return

    method = report["method"]
    if method is None:
        print("  optimizer.method: <not declared> -- cannot audit, skipping key checks")
        print()
        return
    print(f"  optimizer.method: {method}"
          + ("" if report["method_known"] else "  [unknown to this script's map -- results below are incomplete]"))

    print("  -- declared but NOT read for this method --")
    if report["declared_unread"]:
        for row in report["declared_unread"]:
            print(f"    {row['key']}")
            print(f"        reason: {row['reason']}")
    else:
        print("    (none)")

    print("  -- read by this method but NOT declared (falls back to default) --")
    if report["missing_uses_default"]:
        for row in report["missing_uses_default"]:
            note = f"  ({row['note']})" if row["note"] else ""
            print(f"    {row['key']} -> default={row['default']}  [source: {row['source']}]{note}")
    else:
        print("    (none)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--configs",
        nargs="*",
        default=["configs/*.yaml"],
        help="glob pattern(s) or explicit file path(s); default: configs/*.yaml",
    )
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a table")
    args = ap.parse_args()

    paths: List[str] = []
    for pattern in args.configs:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    # de-duplicate while preserving order
    seen = set()
    unique_paths = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    reports = [audit_config(p) for p in unique_paths]

    if args.json:
        print(json.dumps(reports, indent=2, ensure_ascii=False))
    else:
        print(
            "config-key effective-parameter audit (report-only; does not modify "
            "anything, always exits 0)\n"
        )
        for report in reports:
            _print_human(report)

    raise SystemExit(0)


if __name__ == "__main__":
    main()
