"""Unit tests for eval/evaluate.py normalization and scoring."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import (  # noqa: E402
    HELDOUT_DEV_IDS,
    FieldScore,
    _cell,
    _summarize_bucket,
    DocResult,
    normalize_container,
    normalize_scalar,
    score_document,
)


# ---------------------------------------------------------------------------
# normalize_scalar
# ---------------------------------------------------------------------------


class TestNormalizeScalar:
    def test_none_returns_empty(self):
        assert normalize_scalar("shipper", None) == ""

    def test_case_and_punctuation(self):
        a = normalize_scalar("vessel_name", "MV Northern Star")
        b = normalize_scalar("vessel_name", "mv northern.star")
        assert a == b

    def test_whitespace_collapsed(self):
        a = normalize_scalar("port_of_loading", "Hong Kong (HKHKG)")
        b = normalize_scalar("port_of_loading", "hongkong hkhkg")
        assert a == b

    def test_date_variants_canonicalize(self):
        values = [
            "03 JUN 2026",
            "03 June 2026",
            "2026-06-03",
            "03/06/2026",
            "June 3, 2026",
            "Jun 3, 2026",
            "03-06-2026",
        ]
        normed = {normalize_scalar("shipped_on_board_date", v) for v in values}
        assert normed == {"2026-06-03"}

    def test_date_fallback_to_text_normalization(self):
        # A string that is not parseable as a date still normalizes (lowercased
        # with punctuation stripped), it just does not canonicalize.
        n = normalize_scalar("shipped_on_board_date", "unknown date")
        assert n == "unknowndate"

    def test_weight_canonical_in_free_text(self):
        a = normalize_scalar("mark", "Gross wt. 25,340 KG")
        b = normalize_scalar("mark", "gross wt 25340 kgs")
        assert a == b

    def test_weight_lbs_to_kg(self):
        a = normalize_scalar("mark", "weight 100 LBS")
        b = normalize_scalar("mark", "weight 45 kg")  # 100 lb ~= 45.36 kg -> 45
        assert a == b


# ---------------------------------------------------------------------------
# normalize_container
# ---------------------------------------------------------------------------


class TestNormalizeContainer:
    def test_uppercase_stripped(self):
        assert normalize_container("mrku-706 178 4") == "MRKU7061784"

    def test_none_empty(self):
        assert normalize_container(None) == ""


# ---------------------------------------------------------------------------
# score_document
# ---------------------------------------------------------------------------


GT = {
    "company_name": "MAERSK LINE",
    "bl_number": "MAER-2026-252721",
    "shipper": "Johnson and Sons, 54357 Alex Summit, Kellyfort, Romania",
    "port_of_loading": "Singapore (SGSIN)",
    "port_of_discharge": "Hong Kong (HKHKG)",
    "place_of_acceptance": "Singapore CY",
    "place_of_delivery": "Hong Kong CY",
    "shipped_on_board_date": "03 JUN 2026",
    "vessel_name": "CMA CGM MARCO POLO",
    "voyage_number": "V-320W",
    "mark": "JOHN/HKG/LOT-B",
    "container_numbers": ["OOLU0054473", "TGHU8672246", "MAEU9829387"],
    "lots": ["15 pallets solar panels", "29 crates furniture", "32 bales cotton fabric"],
}


class TestScoreDocument:
    def test_exact_prediction_full_score(self):
        s = score_document(GT, dict(GT))
        for name, sc in s.items():
            assert sc.strict == 1.0, name
            assert sc.normalized == 1.0, name

    def test_none_prediction_zero_everywhere(self):
        s = score_document(GT, None)
        assert all(v.strict == 0.0 and v.normalized == 0.0 for v in s.values())

    def test_date_wrong_format_strict_zero_normalized_one(self):
        pred = dict(GT)
        pred["shipped_on_board_date"] = "2026-06-03"
        s = score_document(GT, pred)
        assert s["shipped_on_board_date"] == FieldScore(strict=0.0, normalized=1.0)

    def test_container_reordering_scores_full(self):
        pred = dict(GT)
        pred["container_numbers"] = list(reversed(GT["container_numbers"]))
        s = score_document(GT, pred)
        assert s["container_numbers"].strict == 1.0
        assert s["container_numbers"].normalized == 1.0

    def test_container_missing_one_f1(self):
        pred = dict(GT)
        pred["container_numbers"] = GT["container_numbers"][:2]
        s = score_document(GT, pred)
        # tp=2, exp=3, pred=2 -> precision=1, recall=2/3 -> F1 = 0.8
        assert round(s["container_numbers"].strict, 3) == 0.8
        assert round(s["container_numbers"].normalized, 3) == 0.8

    def test_container_extra_wrong_one(self):
        pred = dict(GT)
        pred["container_numbers"] = GT["container_numbers"] + ["BOGU0000000"]
        s = score_document(GT, pred)
        # tp=3, exp=3, pred=4 -> precision=3/4, recall=1 -> F1 = 6/7
        assert round(s["container_numbers"].strict, 3) == round(6 / 7, 3)

    def test_container_normalization_handles_dashes(self):
        pred = dict(GT)
        pred["container_numbers"] = ["oolu-005 4473", "tghu-867 2246", "maeu-982 9387"]
        s = score_document(GT, pred)
        assert s["container_numbers"].strict == 0.0
        assert s["container_numbers"].normalized == 1.0

    def test_scalar_case_insensitive_normalized(self):
        pred = dict(GT)
        pred["company_name"] = "maersk line"
        s = score_document(GT, pred)
        assert s["company_name"].strict == 0.0
        assert s["company_name"].normalized == 1.0

    def test_missing_scalar_scores_zero(self):
        pred = dict(GT)
        pred["bl_number"] = None
        s = score_document(GT, pred)
        assert s["bl_number"] == FieldScore(strict=0.0, normalized=0.0)


class TestHeldoutDev:
    def test_ten_docs_balanced(self):
        assert len(HELDOUT_DEV_IDS) == 10
        # Ground truth filenames align with the holdout ids.
        for doc_id in HELDOUT_DEV_IDS:
            assert doc_id.startswith("bl_")

    def test_balance_across_templates_and_levels(self):
        # Template of bl_<i> is (i % 3) + 1; level is heavy if i is odd.
        idx = {int(d.split("_")[1]) for d in HELDOUT_DEV_IDS}
        by_tpl = {1: 0, 2: 0, 3: 0}
        n_heavy = 0
        for i in idx:
            by_tpl[(i % 3) + 1] += 1
            if i % 2 == 1:
                n_heavy += 1
        assert all(c >= 3 for c in by_tpl.values()), f"template imbalance: {by_tpl}"
        assert n_heavy == 5  # 5 heavy + 5 light


class TestReportCells:
    def test_cell_format(self):
        r1 = DocResult(doc_id="x", split="clean", template=1, llm_s=1.0,
                       prediction={}, scores=score_document(GT, GT))
        r2 = DocResult(doc_id="y", split="clean", template=1, llm_s=2.0,
                       prediction={}, scores=score_document(GT, {**GT, "bl_number": None}))
        s = _summarize_bucket([r1, r2])
        # bl_number: one perfect, one zero -> mean 0.50, perfect 1/2
        assert _cell(s["per_field"]["bl_number"], "strict") == "0.50 (1/2)"
        assert _cell(s["per_field"]["container_numbers"], "strict") == "1.00 (2/2)"
