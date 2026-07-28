"""Regression tests for the effective config-key audit."""

from io import StringIO
from pathlib import Path

import yaml

from scripts.audit.config_key_audit import audit_config


ROOT = Path(__file__).resolve().parents[1]


def test_phase1_greedy_objectives_are_not_falsely_reported_dead():
    report = audit_config(str(ROOT / "configs" / "phase1_mvp_multinews.yaml"))
    unread = {row["key"] for row in report["declared_unread"]}

    assert not {
        "objectives.lambda_importance",
        "objectives.lambda_coverage",
        "objectives.lambda_redundancy",
        "objectives.coverage_method",
        "objectives.importance_aggregation",
        "experiment.status",
        "experiment.dataset",
        "data_policy.policy_path",
        "data_policy.policy_sha256",
        "data_policy.analysis",
        "seed",
    } & unread
    assert report["declared_unread"] == []


def test_unknown_data_policy_key_is_reported(monkeypatch):
    rendered = yaml.safe_dump(
        {
            "optimizer": {"method": "greedy"},
            "data_policy": {"policy_paht": "typo.json"},
        }
    )
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: StringIO(rendered))
    report = audit_config("bad-data-policy.yaml")
    assert report["declared_unread"][0]["key"] == "data_policy.policy_paht"


def test_canonical_greedy_flags_superseded_legacy_alpha(monkeypatch):
    rendered = yaml.safe_dump(
        {
            "optimizer": {"method": "greedy"},
            "redundancy": {"lambda": 0.7},
        }
    )
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: StringIO(rendered))
    report = audit_config("greedy-with-legacy-alpha.yaml")
    assert {row["key"] for row in report["declared_unread"]} == {
        "redundancy.lambda"
    }


def test_unknown_scoped_key_is_reported(monkeypatch):
    rendered = yaml.safe_dump(
        {
            "optimizer": {"method": "greedy"},
            "objectives": {"lambda_importnace": 1.0},
        }
    )
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: StringIO(rendered))

    report = audit_config("typo.yaml")
    assert report["declared_unread"][0]["key"] == "objectives.lambda_importnace"
