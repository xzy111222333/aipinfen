import json
import tempfile
import unittest
from pathlib import Path

from scoring import grade_submission
from scoring.engine import load_question_bundle
from scoring.grid_sheet import measure_benchmark_layout
from scoring.reference_workflow import writing_references


EXAMPLE = Path(__file__).parent / "examples" / "night_market"


class PortableWorkflowTests(unittest.TestCase):
    def test_material_and_references(self):
        question = load_question_bundle(EXAMPLE / "第3题.md")
        self.assertTrue(question["材料段落"])
        self.assertGreaterEqual(len(question["参考答案"]), 6)

    def test_generation_weights(self):
        votes = [(943, 30), (354, 17), (410, 47), (269, 13), (84, 5)]
        rows = [{"正文": str(i), "点赞": up, "点踩": down} for i, (up, down) in enumerate(votes)]
        refs = writing_references(rows)
        self.assertAlmostEqual(refs[0]["写作参考权重"], 0.3356, places=4)
        self.assertAlmostEqual(sum(r["写作参考权重"] for r in refs), 1.0, places=4)

    def test_verified_layout(self):
        answer = json.loads((EXAMPLE / "标杆答案.json").read_text(encoding="utf-8"))
        self.assertEqual(answer["格数"], 424)
        self.assertEqual(measure_benchmark_layout(answer["正文"], 25), 494)

    def test_frozen_fingerprint(self):
        old = json.loads((EXAMPLE / "复评分.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as cache:
            result = grade_submission(EXAMPLE / "冻结细则.json", "", cache)
        self.assertEqual(result["总分"], 0)
        self.assertEqual(result["细则指纹"], old["细则指纹"])
