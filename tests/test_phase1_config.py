"""Static guardrails for the validation-only Phase 1 MVP configuration."""

from pathlib import Path

from src.utils.io import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_phase1_mvp_config_keeps_budgets_and_routes_explicit():
    cfg = load_yaml(str(ROOT / "configs" / "phase1_mvp_multinews.yaml"))
    assert cfg["experiment"]["status"] == "validation_pilot_only"
    assert cfg["compute_budget"] == {
        "mode": "fixed",
        "enabled_routes": ["lexical", "semantic"],
    }
    assert cfg["candidate_budget"]["per_route"] > 0
    assert cfg["candidate_budget"]["total"] >= cfg["candidate_budget"]["per_route"]
    assert cfg["length_control"]["max_words"] > 0
    assert cfg["routes"]["semantic"]["revision"] != "main"
    assert len(cfg["routes"]["semantic"]["revision"]) == 40
    assert cfg["optimizer"]["method"] == "greedy"
