import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from .__main__ import build, check_analysis, check_reference_decisions, check_rubric, collect_reference_candidates, grade
from .prompts import ANALYZE, COMPILE, GRADE, REFERENCES


def fixture():
    source = {"题干": "概括做法。", "材料": [{"全局段号": 1, "原文": "设立客运专线。"}],
              "满分": 2, "字数上限": 100, "字数下限": 0}
    analysis = {"题型": 1, "任务": [{"编号": "T1", "题干引文": "概括做法", "交付": "做法",
                                   "判定": "具备具体动作", "所属": "内容"}],
                "材料提取": [{"段号": 1, "原句": "设立客运专线", "对应任务": ["T1"]}]}
    item = {"编号": "S1", "名称": "专线", "组": "做法", "类别": "内容", "任务": ["T1"],
            "依据": [1], "等级": [{"分值": 0, "条件": "未表达"}, {"分值": 2, "条件": "设立专线"}],
            "有效写法": "增设专项线路", "误判边界": "仅说改善交通"}
    rubric = {"任务覆盖": {"T1": "S1"}, "阅卷规则": "按要点给分", "计分理由": "唯一任务",
              "评分项": [item], "档位": [],
              "样例": [{"名称": str(i), "答卷": "设立客运专线", "预期区间": [2, 2], "说明": "采分"} for i in range(4)]}
    return source, analysis, rubric


class WorkshopTests(unittest.TestCase):
    def test_valid_rubric(self):
        source, analysis, rubric = fixture()
        check_analysis(analysis, source)
        check_rubric(rubric, source, analysis)

    def test_rejects_invented_material(self):
        source, analysis, _ = fixture()
        analysis["材料提取"][0]["原句"] = "关闭客运专线"
        with self.assertRaisesRegex(ValueError, "材料引文错误"):
            check_analysis(analysis, source)

    def test_rejects_wrong_75_25(self):
        source, analysis, rubric = fixture()
        analysis["题型"] = 2
        with self.assertRaisesRegex(ValueError, "分值分配错误"):
            check_rubric(rubric, source, analysis)

    def test_rejects_unlisted_partial_score(self):
        source, analysis, rubric = fixture()
        result = {"判卷": [{"明细": [{"编号": "S1", "分值": 1, "引文": "专线", "理由": "任意给分"}]}]}
        with patch("rubric_workshop.__main__.call_llm_json", return_value=result):
            with self.assertRaisesRegex(ValueError, "未授权分值"):
                grade(source, analysis, rubric, ["设立客运专线"])

    def test_sums_locked_levels(self):
        source, analysis, rubric = fixture()
        result = {"判卷": [{"明细": [{"编号": "S1", "分值": 2, "引文": "客运专线", "理由": "命中"}]}]}
        with patch("rubric_workshop.__main__.call_llm_json", return_value=result):
            self.assertEqual(grade(source, analysis, rubric, ["设立客运专线"])[0]["得分"], 2)

    def test_group_cap_is_enforced_after_per_item_grading(self):
        source, analysis, rubric = fixture()
        alternate = deepcopy(rubric["评分项"][0])
        alternate["编号"] = "S2"
        rubric["评分项"].append(alternate)
        rubric["组封顶"] = {"做法": 2}
        check_rubric(rubric, source, analysis)
        marks = [{"编号": i, "分值": 2, "引文": "客运专线", "理由": "命中"} for i in ("S1", "S2")]
        with patch("rubric_workshop.__main__.call_llm_json", return_value={"判卷": [{"明细": marks}]}):
            self.assertEqual(grade(source, analysis, rubric, ["设立客运专线"])[0]["得分"], 2)

    def test_rejects_essay_higher_than_triggered_cap(self):
        source, analysis, rubric = fixture()
        analysis["题型"] = 3
        rubric["档位"] = [{"名称": "一类上", "最低": 30, "最高": 35}]
        rubric["限档"] = [{"编号": "C1", "条件": "偏题", "最高档": "三类上"}]
        mark = {"档位": "一类上", "分值": 31, "引文": "客运专线", "理由": "合格",
                "限档判断": [{"编号": "C1", "命中": True, "理由": "偏题"}]}
        with patch("rubric_workshop.__main__.call_llm_json", return_value={"判卷": [mark]}):
            with self.assertRaisesRegex(ValueError, "超过细则允许的最高档"):
                grade(source, analysis, rubric, ["设立客运专线"])

    def test_build_checks_references_and_saves_both_formats(self):
        source, analysis, rubric = fixture()
        analysis["材料提取"][0].update({"大意": "设线", "作用": "支持措施", "限定": "客运"})
        bundle = {"题干与要求": "概括做法（2分），不超过100字。", "材料段落": source["材料"],
                  "参考答案": [{"来源": f"来源{i}", "正文": "设立客运专线。", "格数": 8}
                           for i in range(6)]}
        candidate = {"摘录": "设立客运专线", "归纳": "设线", "处理": "采纳", "依据": [1],
                     "理由": "有原文", "任务": ["T1"]}
        rubric["候选裁决"] = {"R1": {"处理": "计分", "对应细则": ["S1"], "理由": "有原文"}}
        mark = {"明细": [{"编号": "S1", "分值": 2, "引文": "客运专线", "理由": "命中"}]}
        outputs = {ANALYZE: analysis, COMPILE: rubric, GRADE: {"判卷": [mark] * 4}}

        def reply(prompt, payload, **kwargs):
            if prompt == ANALYZE:
                self.assertNotIn("参考答案", payload)
            if prompt == GRADE:
                self.assertNotIn("样例", payload["细则"])
                self.assertNotIn("候选裁决", payload["细则"])
            if prompt == REFERENCES:
                self.assertTrue(all("来源" not in row for row in payload["参考答案"]))
                return {"参考核对": [{"编号": row["编号"], "候选": [candidate]} for row in payload["参考答案"]]}
            if prompt == COMPILE:
                self.assertEqual(len(payload["参考候选"]), 1)
                self.assertEqual(len(payload["参考候选"][0]["观察"]), 6)
            return deepcopy(outputs[prompt])

        with TemporaryDirectory() as root:
            with patch("rubric_workshop.__main__.load_question_bundle", return_value=bundle), \
                 patch("rubric_workshop.__main__.call_llm_json", side_effect=reply) as model:
                path = build(Path("第1题.md"), Path(root))
                self.assertEqual(model.call_count, 5)
                self.assertIn("已锁定", path.read_text(encoding="utf-8"))
                self.assertIn("客运专线", path.with_suffix(".md").read_text(encoding="utf-8"))

    def test_reference_disagreement_survives_first_review(self):
        _, _, rubric = fixture()
        included = {"摘录": "设立客运专线", "归纳": "设线", "处理": "采纳",
                    "依据": [1], "理由": "有原文", "任务": ["T1"]}
        excluded = {**included, "处理": "排除", "理由": "初判有争议"}
        other = {**excluded, "归纳": "关闭线路", "摘录": "关闭线路"}
        candidates = collect_reference_candidates(
            [{"编号": 1, "候选": [included]}, {"编号": 2, "候选": [excluded, other]}],
        )
        self.assertEqual([len(row["观察"]) for row in candidates], [2, 1])
        rubric["候选裁决"] = {"R1": {"处理": "计分", "对应细则": ["S1"], "理由": "有原文"}}
        with self.assertRaisesRegex(ValueError, "每个参考候选"):
            check_reference_decisions(rubric, candidates)
        rubric["候选裁决"]["R2"] = {"处理": "排除", "对应细则": [], "理由": "与原文相反"}
        check_reference_decisions(rubric, candidates)
