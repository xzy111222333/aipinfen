import hashlib
import io
import json
import unittest
import urllib.request
from http.client import IncompleteRead
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from urllib.error import URLError

from eval.metrics import compare_real, summarize_stability, variant_result
from eval.prepare_dataset import EVAL, ROOT, read_json
from eval.telemetry import capture_network_usage, track_usage
from scoring.engine import API_URL, MODEL_DEFAULT, RELAY_KEY, RELAY_MODEL, RELAY_URL, call_llm_json, llm_cache_model
from scoring.rubric_cache import load_or_build_rubric


class EvalProtocolTest(unittest.TestCase):
    def test_configured_relay_and_explicit_baseline_use_separate_models_and_keys(self):
        def respond(request, *, timeout):
            if network.call_count == 1:
                raise ConnectionResetError("relay disconnected")
            model = json.loads(request.data)["model"]
            choices = [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]
            return io.BytesIO(json.dumps({"model": model, "choices": choices}).encode())

        with patch("scoring.engine._read_api_setting", side_effect={RELAY_KEY: "relay"}.get):
            self.assertEqual(llm_cache_model(), (RELAY_URL, RELAY_MODEL))
            self.assertEqual(llm_cache_model("baseline"), MODEL_DEFAULT)
            with patch.object(urllib.request, "urlopen", side_effect=respond) as network:
                self.assertEqual(call_llm_json("JSON", {}), {"ok": True})
                self.assertEqual(call_llm_json("JSON", {}, api_key="baseline"), {"ok": True})
        requests = [call.args[0] for call in network.call_args_list]
        self.assertEqual([(r.full_url, r.get_header("Authorization")) for r in requests],
                         [(RELAY_URL, "Bearer relay"), (RELAY_URL, "Bearer relay"), (API_URL, "Bearer baseline")])

    def test_interrupted_response_is_counted_and_retryable(self):
        partial = Mock()
        partial.__enter__ = Mock(return_value=partial)
        partial.__exit__ = Mock(return_value=False)
        partial.read.side_effect = IncompleteRead(b"partial")
        complete = io.BytesIO(b'{"usage":{"prompt_tokens":3,"completion_tokens":2}}')
        with patch.object(urllib.request, "urlopen", side_effect=[partial, complete]):
            with capture_network_usage(), track_usage() as usage:
                request = urllib.request.Request(API_URL)
                with self.assertRaises(URLError):
                    urllib.request.urlopen(request)
                with urllib.request.urlopen(request) as response:
                    self.assertIn(b"usage", response.read())
        self.assertEqual(usage["requests"], 2)
        self.assertEqual(usage["missing_usage"], 1)
        self.assertEqual(usage["input_tokens"], 3)
        self.assertEqual(usage["output_tokens"], 2)

    def test_engine_caches_are_separate(self):
        with TemporaryDirectory() as directory:
            self.assertEqual(load_or_build_rubric(directory, ["q"], lambda: 0, engine="v0"), 0)
            self.assertEqual(load_or_build_rubric(directory, ["q"], lambda: 1, engine="v1"), 1)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)

    def test_frozen_snapshot_matches_hashes(self):
        for line in (EVAL / "v0_frozen.sha256").read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)

    def test_split_is_by_paper(self):
        split = read_json(EVAL / "split.json")["papers"]
        for q in read_json(EVAL / "questions.json") + read_json(EVAL / "real" / "index.json"):
            self.assertEqual(split[q["paper"]], q["split"])
        heldout = [q for q in read_json(EVAL / "real" / "index.json") if q["split"] == "test"]
        self.assertTrue(any(sum(r["paper"] == q["paper"] for r in heldout) > 1 for q in heldout))

    def test_invariance_and_weak_keyword_banding(self):
        variant = {"variant_id": "Q-B-V1", "type": "V1", "expect": {"relation": "equal", "max_gap_ratio": 0.05}}
        self.assertTrue(variant_result({"maximum": 20}, variant, {"score": 15}, {"score": 14})["passed"])
        self.assertFalse(variant_result({"maximum": 20}, variant, {"score": 15}, {"score": 13.5})["passed"])
        variant["expect"] = {"relation": "lower", "min_drop_ratio": 0.2, "band_max": 2}
        self.assertFalse(variant_result({"maximum": 20}, variant, {"score": 20}, {"score": 16, "band": 1})["passed"])
        self.assertTrue(variant_result({"maximum": 20}, variant, {"score": 20}, {"score": 16, "band": 2})["passed"])

    def test_stability_and_constant_baseline(self):
        g = {"question_id": "Q", "answer_id": "A", "maximum": 20, "essay": False}
        rows = [{**g, "rubric_round": 0, "score": s} for s in [10, 10.5, 11, 10, 10]]
        rows.extend([{**g, "rubric_round": 1, "score": 8}, {**g, "rubric_round": 2, "score": 11}])
        metrics = summarize_stability(rows)
        self.assertTrue(metrics["fixed_details"][0]["passed"])
        self.assertFalse(metrics["rebuilt_details"][0]["passed"])
        real = [{"paper": "同卷", "path": str(score), "recorded": score, "prediction": 75} for score in [70, 76]]
        summary = compare_real(real)
        self.assertEqual(summary["M4"]["rate"], 0)
        self.assertEqual(summary["M5"]["mae"], 3)
