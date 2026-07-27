"""Snapshot test: ten toy documents through the full selection pipeline.

WHAT THIS IS
------------
Unlike the hand-calculated golden tests, the expected values here were RECORDED
from a run. Nobody can hand-compute the output of
segmentation -> features -> three independent route rankings -> RRF ->
reservations -> coverage guards -> total cap -> selector -> feasibility.

So this test does not claim the output is *correct*. It claims the output has
not *changed*. That is still worth having: the pipeline has many wiring seams
where a change is silent -- a route quietly dropped, tie-breaking reordered, a
guard interacting differently with the cap -- and every one of those would leave
the rest of the suite green.

This is deliberately layered on top of the hand-calculated golden tests
(test_features_golden.py, test_rouge_golden.py). Those pin the feature and
metric definitions by independent arithmetic, so this snapshot cannot silently
bless a wrong formula the way a snapshot of the pre-refactor pipeline would
have blessed rougeL and the TF-ISF length bias.

WHEN THIS FAILS
---------------
Do not just refresh the fixture. Decide first whether the behaviour change was
intended, then regenerate with

    python -m tests.test_pipeline_snapshot --update

and state in the commit message why the output should differ.

WHAT IS RECORDED
----------------
The decision trail, not just the summary text. A change can leave the chosen
sentences identical while altering how they were chosen, and that is exactly
what needs catching: route ranks, inclusion reasons, the budget allocation, the
selector salience source, and the feasibility verdict.

The semantic route is stubbed with a deterministic scorer so this runs in CI
without torch or a model download, and so the snapshot never depends on
checkpoint weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.data.schemas import build_document_example
from src.pipeline.select_sentences import summarize_one

FIXTURE = Path(__file__).parent / "fixtures" / "toy_pipeline_snapshot.json"


# --------------------------------------------------------------------------
# Deterministic stand-in for the sentence encoder.
# --------------------------------------------------------------------------
def _stub_encoder_route_scores(sentences, **kwargs):
    """Score sentences by a fixed, content-derived rule.

    Deliberately anti-correlated with lexical salience for sentences carrying
    the marker token, so the semantic route genuinely disagrees with the
    lexical route and the fusion path is exercised rather than bypassed.
    """
    scores = []
    for index, sentence in enumerate(sentences):
        tokens = sentence.lower().split()
        marker = 1.0 if "signal" in tokens else 0.0
        # Later sentences score slightly higher, opposing any lead prior.
        positional = index / max(1, len(sentences) - 1)
        scores.append(round(0.6 * marker + 0.4 * positional, 6))
    return scores, {
        "model_revision": "stub-encoder:v1",
        "estimated_cost": {"encoded_sentences": len(sentences)},
        "truncation_rate": 0.0,
    }


# --------------------------------------------------------------------------
# Ten toy documents, each chosen for a distinct pipeline shape.
# --------------------------------------------------------------------------
def _toy_documents():
    def doc(example_id, documents, input_mode="multi_document"):
        return build_document_example(
            example_id=example_id,
            split="validation",
            documents=documents,
            references=["A reference summary about the signal topic."],
            input_mode=input_mode,
            output_mode="multi_sentence",
            dataset_name="toy",
        )

    long_sentence = " ".join(["padding"] * 60) + " signal end."

    return [
        # 1. Two source documents, routes broadly agree.
        doc("toy_01", [
            ["Alpha signal opens the story.", "Beta follows with detail.",
             "Gamma adds context."],
            ["Delta signal reports separately.", "Epsilon closes the account."],
        ]),
        # 2. Single document, several sentences.
        doc("toy_02", [
            ["One signal here.", "Two follows.", "Three continues.",
             "Four concludes."],
        ], input_mode="single_document"),
        # 3. Degenerate: a single sentence.
        doc("toy_03", [["Only one signal sentence exists."]],
            input_mode="single_document"),
        # 4. One sentence longer than the whole budget.
        doc("toy_04", [[long_sentence, "Short signal tail."]],
            input_mode="single_document"),
        # 5. Many short sentences: reservation and cap pressure.
        doc("toy_05", [
            [f"Short line {i} signal." if i % 3 == 0 else f"Short line {i}."
             for i in range(12)],
        ], input_mode="single_document"),
        # 6. Near-duplicates: exercises redundancy.
        doc("toy_06", [
            ["The signal repeats often here.", "The signal repeats often here.",
             "The signal repeats often now.", "A different closing remark."],
        ], input_mode="single_document"),
        # 7. Routes disagree: lexical-rich early, marker-rich late.
        doc("toy_07", [
            ["Rare vocabulary dominates this opening clause.",
             "Ordinary filler follows.", "More ordinary filler.",
             "Trailing signal signal signal."],
        ], input_mode="single_document"),
        # 8. Small n, so the total cap cannot be filled.
        doc("toy_08", [["First signal.", "Second line."]],
            input_mode="single_document"),
        # 9. Three source documents: coverage guard across groups.
        doc("toy_09", [
            ["Doc one signal opening.", "Doc one continues."],
            ["Doc two signal opening.", "Doc two continues."],
            ["Doc three signal opening.", "Doc three continues."],
        ]),
        # 10. Abbreviations and decimals, which naive splitters mis-handle.
        doc("toy_10", [
            ["Dr. Smith reported a 3.5 percent signal increase.",
             "The U.S. team confirmed it.", "Analysis continues."],
        ], input_mode="single_document"),
    ]


CONFIG = {
    "seed": 2024,
    "features": {
        "tf_isf": {"version": "v2", "use_stopwords": True,
                   "use_sublinear_tf": True, "use_bigrams": False},
        "weights": {"importance": 1.0, "length": 0.0, "position": 0.0,
                    "graph": 0.0, "centrality": 0.0, "novelty": 0.0},
    },
    "representations": {"use": True, "method": "tfidf"},
    "compute_budget": {"mode": "fixed", "enabled_routes": ["lexical", "semantic"]},
    # min_per_route is 1, not 2, because a reservation larger than the shortest
    # document currently raises. See TestShortDocumentReservation below.
    "candidate_budget": {"route_top_k": 4, "min_per_route": 1, "total": 6},
    "candidates": {"use": True, "mode": "hard", "rrf_constant": 60},
    "routes": {
        "lexical": {"revision": "toy-lexical:v1"},
        "semantic": {"model_name": "stub", "revision": "stub-encoder:v1"},
    },
    "coverage_guard": {"enabled": True, "document": True,
                       "section": False, "position": False},
    "optimizer": {"method": "greedy"},
    "length_control": {"unit": "words", "max_words": 40},
    "redundancy": {"lambda": 0.7},
}


def _record(result: dict) -> dict:
    """Reduce a run to the decisions worth protecting."""
    pool = result["candidate_pool"]
    return {
        "id": result["id"],
        "selected_indices": result["selected_indices"],
        "summary_sentences": result["summary_sentences"],
        "candidate_pool": {
            "configured_sources": pool["configured_sources"],
            "route_top_k": pool["route_top_k"],
            "min_per_route": pool["min_per_route"],
            "total_cap": pool["total_cap"],
            "actual_size": pool["actual_size"],
            "selector_salience_source": pool["selector_salience_source"],
            "route_proposals": pool["route_proposals"],
            "allocation": pool["allocation"],
        },
        "candidates": [
            {
                "original_index": candidate["original_index"],
                "document_id": candidate["document_id"],
                "fused_rank": candidate["fused_rank"],
                "route_agreement": candidate["route_agreement"],
                "selected_by_routes": sorted(candidate["selected_by_routes"]),
                "inclusion_reasons": sorted(candidate["inclusion_reasons"]),
                "route_ranks": {
                    route: score["rank"]
                    for route, score in sorted(candidate["route_scores"].items())
                },
            }
            for candidate in result["candidate_records"]
        ],
        "objective_spec": {
            key: result["objective_spec"][key]
            for key in ("status", "input_mode", "output_mode", "active",
                        "importance_aggregation", "coverage_scope")
            if key in result["objective_spec"]
        },
        "selection_evaluation": result["selection_evaluation"],
        "output_budget": result["output_budget"],
    }


def _run_all(monkeypatch=None) -> list[dict]:
    import src.pipeline.candidate_builder as candidate_builder

    if monkeypatch is not None:
        monkeypatch.setattr(
            candidate_builder, "encoder_route_scores", _stub_encoder_route_scores
        )
        return [_record(summarize_one(doc, CONFIG)) for doc in _toy_documents()]

    original = candidate_builder.encoder_route_scores
    candidate_builder.encoder_route_scores = _stub_encoder_route_scores
    try:
        return [_record(summarize_one(doc, CONFIG)) for doc in _toy_documents()]
    finally:
        candidate_builder.encoder_route_scores = original


class TestToyPipelineSnapshot:
    def test_matches_recorded_snapshot(self, monkeypatch):
        assert FIXTURE.exists(), (
            f"missing fixture {FIXTURE}; regenerate with "
            "python -m tests.test_pipeline_snapshot --update"
        )
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        actual = _run_all(monkeypatch)
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            assert got == want, f"pipeline output changed for {want['id']}"

    def test_every_document_produces_a_feasible_summary(self, monkeypatch):
        """A snapshot alone would happily record an infeasible result, so assert
        the contract independently."""
        for record in _run_all(monkeypatch):
            evaluation = record["selection_evaluation"]
            assert evaluation["feasible"], f"{record['id']} is infeasible"
            assert record["selected_indices"], f"{record['id']} is empty"

    def test_candidates_never_exceed_the_declared_cap(self, monkeypatch):
        for record in _run_all(monkeypatch):
            allocation = record["candidate_pool"]["allocation"]
            assert allocation["actual_size"] <= allocation["candidate_universe_size"]
            cap = record["candidate_pool"]["total_cap"]
            if cap is not None:
                assert allocation["actual_size"] <= cap

    def test_selected_sentences_come_from_the_candidate_pool(self, monkeypatch):
        for record in _run_all(monkeypatch):
            pool = {c["original_index"] for c in record["candidates"]}
            assert set(record["selected_indices"]) <= pool, record["id"]

    def test_both_routes_are_actually_exercised(self, monkeypatch):
        """Guards against a route being silently dropped: at least one document
        must contain a candidate proposed by each route."""
        proposing = set()
        for record in _run_all(monkeypatch):
            for candidate in record["candidates"]:
                proposing.update(candidate["selected_by_routes"])
        assert {"lexical", "semantic"} <= proposing


class TestShortDocumentReservation:
    """Pins current behaviour when a document has fewer sentences than the
    per-route reservation.

    ``route_top_k`` is clamped to the document's sentence count before
    ``min_per_route`` is validated against it, so a short document surfaces as
    a configuration error rather than as a document the reservation cannot be
    satisfied for.

    This matters in production: with the Phase 1 MVP setting
    (``route_top_k: 40, min_per_route: 20``), roughly 9% of Multi-News
    validation documents have fewer than 20 sentences and would abort the run.

    These tests record the behaviour as it stands. If the semantics are changed
    so a reservation is clamped to what the document can supply, update them and
    say so explicitly -- do not let the change pass silently.
    """

    def _single_sentence_doc(self):
        return build_document_example(
            example_id="short_01",
            split="validation",
            documents=[["Only one signal sentence exists."]],
            references=["A reference."],
            input_mode="single_document",
            output_mode="multi_sentence",
            dataset_name="toy",
        )

    def test_reservation_larger_than_the_document_raises(self, monkeypatch):
        import src.pipeline.candidate_builder as candidate_builder

        monkeypatch.setattr(
            candidate_builder, "encoder_route_scores", _stub_encoder_route_scores
        )
        config = json.loads(json.dumps(CONFIG))
        config["candidate_budget"] = {
            "route_top_k": 40, "min_per_route": 20, "total": 60
        }
        with pytest.raises(ValueError, match="exceeds route_top_k"):
            summarize_one(self._single_sentence_doc(), config)

    def test_reservation_within_the_document_succeeds(self, monkeypatch):
        import src.pipeline.candidate_builder as candidate_builder

        monkeypatch.setattr(
            candidate_builder, "encoder_route_scores", _stub_encoder_route_scores
        )
        config = json.loads(json.dumps(CONFIG))
        config["candidate_budget"] = {
            "route_top_k": 40, "min_per_route": 1, "total": 60
        }
        result = summarize_one(self._single_sentence_doc(), config)
        assert result["selected_indices"] == [0]
        assert result["selection_evaluation"]["feasible"]


def _update() -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps(_run_all(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    if "--update" in sys.argv:
        _update()
    else:
        print("pass --update to regenerate the snapshot fixture")
