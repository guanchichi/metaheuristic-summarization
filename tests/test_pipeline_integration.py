"""Integration tests for the full pipeline."""

import pytest

from src.data.schemas import build_document_example
from src.pipeline.select_sentences import summarize_one, validate_requested_split


@pytest.fixture
def sample_doc():
    return {
        "id": "test_001",
        "sentences": [
            "This is the first sentence about artificial intelligence.",
            "Machine learning is a subset of AI.",
            "Deep learning has revolutionized many fields.",
            "Natural language processing is important.",
            "Computer vision has many applications.",
        ],
        "highlights": "AI and machine learning are transforming technology.",
    }


@pytest.fixture
def base_config():
    return {
        "optimizer": {"method": "greedy"},
        "length_control": {"unit": "tokens", "max_tokens": 30},
        "redundancy": {"lambda": 0.7},
        "representations": {"use": True, "method": "tfidf"},
        "candidates": {"use": False},
    }


class TestPipelineGreedy:
    def test_basic(self, sample_doc, base_config):
        result = summarize_one(sample_doc, base_config)
        assert "summary" in result
        assert "selected_indices" in result
        assert "id" in result
        assert result["id"] == "test_001"
        assert len(result["summary"]) > 0

    def test_indices_valid(self, sample_doc, base_config):
        result = summarize_one(sample_doc, base_config)
        n = len(sample_doc["sentences"])
        assert all(0 <= i < n for i in result["selected_indices"])

    def test_indices_sorted(self, sample_doc, base_config):
        result = summarize_one(sample_doc, base_config)
        assert result["selected_indices"] == sorted(result["selected_indices"])

    def test_summary_matches_indices(self, sample_doc, base_config):
        result = summarize_one(sample_doc, base_config)
        expected = "\n".join(sample_doc["sentences"][i] for i in result["selected_indices"])
        assert result["summary"] == expected
        assert result["summary_sentences"] == [
            sample_doc["sentences"][i] for i in result["selected_indices"]
        ]


class TestPipelineGrasp:
    def test_grasp(self, sample_doc, base_config):
        base_config["optimizer"]["method"] = "grasp"
        base_config["seed"] = 42
        result = summarize_one(sample_doc, base_config)
        assert len(result["selected_indices"]) > 0


class TestPipelineWithV2Features:
    def test_v2_tf_isf(self, sample_doc, base_config):
        base_config["features"] = {
            "tf_isf": {"version": "v2", "use_stopwords": True},
        }
        result = summarize_one(sample_doc, base_config)
        assert len(result["selected_indices"]) > 0
    def test_v2_position(self, sample_doc, base_config):
        base_config["features"] = {
            "position": {"version": "v2", "method": "inverse"},
        }
        result = summarize_one(sample_doc, base_config)
        assert len(result["selected_indices"]) > 0

    def test_v2_fusion(self, sample_doc, base_config):
        base_config["features"] = {
            "fusion": {"version": "v2"},
            "weights": {"importance": 0.8, "length": 0.2, "position": 0.3},
        }
        result = summarize_one(sample_doc, base_config)
        assert len(result["selected_indices"]) > 0

    def test_semantic_features(self, sample_doc, base_config):
        base_config["features"] = {
            "weights": {
                "importance": 0.6,
                "length": 0.1,
                "position": 0.2,
                "centrality": 0.3,
                "novelty": 0.2,
            },
        }
        result = summarize_one(sample_doc, base_config)
        assert len(result["selected_indices"]) > 0


class TestCandidateBudgetContract:
    def test_pipeline_reports_separate_candidate_budget(self, sample_doc, base_config):
        base_config["candidates"] = {
            "use": True,
            "mode": "hard",
            "sources": ["score"],
        }
        base_config["candidate_budget"] = {"per_route": 4, "total": 2}
        result = summarize_one(sample_doc, base_config)
        assert result["candidate_pool"]["per_route_quota"] == 4
        assert result["candidate_pool"]["total_budget"] == 2
        assert result["candidate_pool"]["actual_size"] == 2

    def test_fixed_compute_budget_controls_enabled_routes(
        self, sample_doc, base_config
    ):
        base_config["candidates"] = {
            "use": True,
            "mode": "hard",
            "sources": ["imaginary_route"],
        }
        base_config["compute_budget"] = {
            "mode": "fixed",
            "enabled_routes": ["lexical"],
        }
        base_config["candidate_budget"] = {"per_route": 2, "total": 2}
        result = summarize_one(sample_doc, base_config)
        assert result["candidate_pool"]["configured_sources"] == ["lexical"]

    def test_unimplemented_adaptive_router_fails_loudly(
        self, sample_doc, base_config
    ):
        base_config["candidates"] = {"use": True}
        base_config["compute_budget"] = {"mode": "adaptive"}
        with pytest.raises(ValueError, match="adaptive routing"):
            summarize_one(sample_doc, base_config)

    def test_word_budget_still_dispatches_configured_selector(
        self, sample_doc, base_config, monkeypatch
    ):
        captured = {}

        def fake_dispatch(*args, **kwargs):
            captured["max_budget"] = args[4]
            captured["unit"] = args[7]
            return [0]

        monkeypatch.setattr(
            "src.pipeline.select_sentences.dispatch_optimizer", fake_dispatch
        )
        base_config["length_control"] = {"unit": "words", "max_words": 17}
        result = summarize_one(sample_doc, base_config)
        assert result["selected_indices"] == [0]
        assert captured == {"max_budget": 17, "unit": "words"}


class TestPipelineEdgeCases:
    def test_empty_sentences(self, base_config):
        doc = {"id": "empty", "sentences": [], "highlights": ""}
        result = summarize_one(doc, base_config)
        assert result["selected_indices"] == []
        assert result["summary"] == ""

    def test_single_sentence(self, base_config):
        doc = {"id": "single", "sentences": ["Hello world."], "highlights": "Hello."}
        result = summarize_one(doc, base_config)
        assert result["selected_indices"] == [0]

    def test_canonical_document_is_supported(self, base_config):
        doc = build_document_example(
            example_id="canonical",
            split="test",
            documents=[["Hello world.", "A second sentence."]],
            references=["First reference.", "Second reference."],
            input_mode="single_document",
            output_mode="single_sentence",
        )
        result = summarize_one(doc, base_config)
        assert result["id"] == "canonical"
        assert "references" not in result
        assert len(result["selected_indices"]) == 1
        assert result["objective_spec"]["active"] == ["salience"]
        assert all(item["document_id"] is not None for item in result["selected_sentences"])

    def test_single_sentence_task_rejects_nsga2(self, base_config):
        doc = build_document_example(
            example_id="single-rank-only",
            split="validation",
            documents=[["One sentence.", "Another sentence."]],
            references=["Reference."],
            input_mode="single_document",
            output_mode="single_sentence",
        )
        base_config["optimizer"]["method"] = "nsga2"
        with pytest.raises(ValueError, match="deterministic ranking"):
            summarize_one(doc, base_config)

    def test_profiled_multi_sentence_defaults_to_mean_importance(self, base_config):
        doc = build_document_example(
            example_id="multi-objectives",
            split="validation",
            documents=[["One sentence.", "Another sentence."]],
            references=["Reference."],
            input_mode="single_document",
            output_mode="multi_sentence",
        )
        result = summarize_one(doc, base_config)
        assert result["objective_spec"]["importance_aggregation"] == "mean"

    def test_profiled_multi_sentence_rejects_sum_importance(self, base_config):
        doc = build_document_example(
            example_id="biased-sum",
            split="validation",
            documents=[["One sentence.", "Another sentence."]],
            references=["Reference."],
            input_mode="single_document",
            output_mode="multi_sentence",
        )
        base_config["objectives"] = {"importance_aggregation": "sum"}
        with pytest.raises(ValueError, match="cardinality bias"):
            summarize_one(doc, base_config)

    def test_gold_reference_cannot_change_selection(self, sample_doc, base_config):
        first = dict(sample_doc, highlights="Completely unrelated gold text.")
        second = dict(sample_doc, highlights="Machine learning AI deep learning.")
        assert summarize_one(first, base_config)["selected_indices"] == summarize_one(
            second, base_config
        )["selected_indices"]

    def test_reference_dependent_candidate_sizing_is_forbidden(self, sample_doc, base_config):
        base_config["candidates"] = {"use": True, "recall_target": 0.9}
        with pytest.raises(ValueError, match="gold references"):
            summarize_one(sample_doc, base_config)

    def test_canonical_split_mismatch_is_forbidden(self):
        doc = build_document_example(
            example_id="validation-row",
            split="validation",
            documents=[["One sentence."]],
            references=["Reference."],
            input_mode="single_document",
            output_mode="single_sentence",
        )
        with pytest.raises(ValueError, match="belongs to split 'validation'"):
            validate_requested_split(doc, "test")
