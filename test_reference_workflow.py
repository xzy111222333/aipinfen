import unittest
from scoring.engine import count_exam_grids, load_question_bundle, parse_hard_constraints
from scoring.reference_workflow import (
    clean_body_and_votes,
    compute_layout_stats,
    reputation_score,
    select_build_and_holdout,
    target_score_split,
    validate_benchmark_style,
)


class TestReferenceWorkflowOffline(unittest.TestCase):
    def test_reputation_score_prefers_high_volume_over_tiny_sample(self):
        # 500赞/60踩(约89.3%好评) 应高于 20赞/1踩(95%好评但样本极小)
        high_vol = reputation_score(500, 60)
        tiny_vol = reputation_score(20, 1)
        self.assertGreater(high_vol, tiny_vol)

    def test_clean_body_strips_vote_header_and_word_count_tail(self):
        raw = "点赞 2483 · 点踩 126 · 采集时间 2026-11-21T08:37:55+08:00\n\n①强化沟通。收集意见。（249字）"
        info = clean_body_and_votes(raw)
        self.assertEqual(info["点赞"], 2483)
        self.assertEqual(info["点踩"], 126)
        self.assertEqual(info["正文"], "①强化沟通。收集意见。")

    def test_holdout_isolation_blocks_same_author_and_polluted_refs(self):
        q = "试卷_含赞踩_2026-11-21/公务员/国考/2025年国家公考《申论》题（行政执法）/第2题.md"
        bundle = load_question_bundle(q)
        cap = parse_hard_constraints(bundle["题干与要求"])["字数上限"]
        training, holdout, _ = select_build_and_holdout(bundle, cap=cap, exclude_sources=["白鹭", "小马哥"], build_count=5)
        self.assertEqual([h["来源"] for h in holdout], ["白鹭", "小马哥"])
        for t in training:
            self.assertNotIn("白鹭", t["来源"])
            self.assertNotIn("小马哥", t["来源"])
            self.assertGreaterEqual(t["点赞"] / max(1, t["点赞"] + t["点踩"]), 0.78)
        self.assertAlmostEqual(sum(t["归一化权重"] for t in training), 1.0, places=3)

    def test_two_part_question_detects_suggestion_heavier_ratio(self):
        q = "试卷_含赞踩_2026-11-21/公务员/国考/2023年国家公考《申论》题（地市级）/第3题.md"
        bundle = load_question_bundle(q)
        cap = parse_hard_constraints(bundle["题干与要求"])["字数上限"]
        training, _, _ = select_build_and_holdout(bundle, cap=cap, exclude_sources=["小张", "小马哥"], build_count=5)
        stats = compute_layout_stats(training, cap)
        self.assertIn("前后两问篇幅占比", stats)
        self.assertLess(stats["前后两问篇幅占比"]["前问占比"], 0.45)
        self.assertGreater(stats["前后两问篇幅占比"]["后问占比"], 0.55)

    def test_score_split_and_benchmark_style_rules(self):
        self.assertEqual(target_score_split(10.0, 1), (10.0, 0.0))
        self.assertEqual(target_score_split(15.0, 2), (11.0, 4.0))
        self.assertEqual(target_score_split(20.0, 2), (15.0, 5.0))
        validate_benchmark_style("S市总工会坚持问需于民。①优化调查。收集意见。②强化培训。提升技能。", True, False)
        with self.assertRaises(ValueError):
            validate_benchmark_style("S市总工会做法如下：①优化调查。②强化培训。", True, False)
        with self.assertRaises(ValueError):
            validate_benchmark_style("①优化调查：收集意见。②强化培训：提升技能。", True, False)
        with self.assertRaises(ValueError):
            validate_benchmark_style("一是优化调查。二是强化培训。", True, False)



    def test_verify_verbatim_terms_filters_low_consensus_words(self):
        from scoring.reference_workflow import verify_verbatim_terms
        training = [
            {"归一化权重": 0.30, "正文": "采用漏斗架，绿色环保种植，发挥碳库作用"},
            {"归一化权重": 0.30, "正文": "传承漏斗架，绿色种植获绿色认证，形成碳库"},
            {"归一化权重": 0.40, "正文": "发展漏斗架，通过国家绿色食品检验，发挥碳库效应"},
        ]
        mat = ["发展漏斗架栽培方式，获得了绿色食品证书，发挥了碳库的作用"]
        self.assertEqual(verify_verbatim_terms(["漏斗架", "绿色食品", "碳库"], training, mat), ["漏斗架", "碳库"])


    def test_preparation_cache_is_separate_by_model(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import Mock
        from scoring.reference_workflow import stage

        first = Mock(return_value={"结果": "甲"})
        second = Mock(return_value={"结果": "乙"})
        with TemporaryDirectory() as directory:
            args = (Path(directory), "制作", "生成结果", {}, lambda result: self.assertIn("结果", result))
            self.assertEqual(stage(*args, model="甲模型", call=first), {"结果": "甲"})
            self.assertEqual(stage(*args, model="乙模型", call=second), {"结果": "乙"})
            self.assertEqual(stage(*args, model="甲模型", call=first), {"结果": "甲"})
        first.assert_called_once()
        second.assert_called_once()

    def test_frozen_rubric_round_trip_keeps_fingerprint(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from scoring.reference_workflow import digest, grade

        rubric = {"材料原文": {1: "第一段", 2: "第二段", 10: "第十段"}}
        loaded = json.loads(json.dumps(rubric))
        with TemporaryDirectory() as directory:
            grade_result = grade(loaded, "", Path(directory))
        self.assertEqual(grade_result["细则指纹"], digest(rubric))


    def test_benchmark_layout_counts_breaks_indent_and_blank_rows(self):
        from scoring.grid_sheet import measure_benchmark_layout

        self.assertEqual(measure_benchmark_layout("标题\n\n　　正文甲乙\n落款", 5), 22)

    def test_benchmark_uses_distinct_budgets_for_small_answers_and_documents(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import Mock
        from scoring.reference_workflow import build_benchmark

        pack = {"题干": "试题", "材料原文": {1: "提供公众服务"}, "字限": 20, "满分": 10,
                "细则": {"内容簇": [{"簇号": "C1"}]}}
        plan = {"段落与分点蓝图": [{"纳入簇号": ["C1"], "前置用法": "原文统领",
                 "前置统领词": "提供公众服务", "材料短引": [{"段号": 1, "短引": "提供公众服务"}]}]}
        align = {"必答任务块": [{"块名": "服务"}], "文种格式硬要求": {}}
        short = {"候选A": "①措施。②补充。", "候选B": "①措施。②补充。", "优选版本": "A", "优选理由": "结构清楚"}
        document = {**short, "候选A": "标题\n　　提供公众服务。\n落款", "候选B": "标题\n　　提供公众服务。\n落款"}
        with TemporaryDirectory() as directory:
            work = Path(directory)
            with self.assertRaisesRegex(ValueError, "至少18格"):
                build_benchmark(pack, align, {}, [], work, call=Mock(side_effect=[plan, short, short, short]))
            two_questions = {**align, "必答任务块": [{"块名": "问题"}, {"块名": "建议"}]}
            wrapped = "问题：①有问题。\n建议：①补齐事项。②加强服务。"
            pack = {**pack, "字限": 25}
            pair = {**short, "候选A": wrapped, "候选B": wrapped}
            answer = build_benchmark(pack, two_questions, {}, [], work, call=Mock(side_effect=[plan, pair, {"正文": wrapped, "补写": [], "复核说明": "已覆盖"}]))
            self.assertEqual(answer["正文"], wrapped.replace("\n", ""))
            self.assertNotIn("排版估算", answer)
            align["文种格式硬要求"] = {"需要标题": True, "禁止分条列点": True}
            with self.assertRaisesRegex(ValueError, "答题区域仅25格"):
                build_benchmark(pack, align, {}, [], work, call=Mock(side_effect=[plan, document, document, document]))
            pack = {**pack, "字限": 75}
            answer = build_benchmark(pack, align, {}, [], work, call=Mock(side_effect=[plan, document, {"正文": document["候选A"], "补写": [], "复核说明": "已覆盖"}]))
            self.assertEqual(answer["格数"], 11)
            self.assertEqual(answer["排版估算"]["排版占行"], 3)
            with self.assertRaisesRegex(ValueError, "答题区域仅20格"):
                build_benchmark(pack, align, {}, [], work, sheet=(10, 2), call=Mock(side_effect=[plan, document, document, document]))


    def test_reference_body_extraction_is_after_holdout_isolation(self):
        held = {"原编号": -1, "来源": "甲老师", "正文": "甲独有的留出正文，它不能提供给制作端。" * 4}
        unrelated = ["①提供就业服务。", "②受理劳动争议。", "③开展技能培训。"]
        bundle = {"题干与要求": "概括主要做法", "参考答案": [
            {"来源": "机构甲老师", "正文": held["正文"]},
            {"来源": "转载者", "正文": "说明文字" * 30 + held["正文"]},
            *[{"来源": str(i), "正文": "答题讲解：" + "分析" * 90 + "\n" + body}
              for i, body in enumerate(unrelated)],
        ]}

        def prepare(_name, _prompt, payload, validate):
            self.assertNotIn(held["正文"], str(payload))
            picked = {"提取": [{"原编号": k, "原文片段": [v.splitlines()[-1]], "说明": "剔除讲解"}
                                for k, v in payload["投稿"].items()]}
            validate(picked)
            picked["提取"][0]["原文片段"] = ["自行改写"]
            with self.assertRaisesRegex(ValueError, "无法按原顺序定位"):
                validate(picked)
            picked["提取"][0]["原文片段"] = [unrelated[0]]
            return picked

        training, holdout, excluded = select_build_and_holdout(bundle, 25, external_holdout=[held], prepare=prepare)
        self.assertEqual([row["正文"] for row in training], unrelated)
        self.assertTrue(all(row["投稿格数"] > 25 >= row["格数"] for row in training))
        self.assertEqual(holdout, [held])
        self.assertEqual([row["原编号"] for row in excluded], [0, 1])


if __name__ == "__main__":
    unittest.main()
