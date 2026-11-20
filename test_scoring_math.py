import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

from scoring.engine import (
    clean_reference_body,
    compile_quantitative_rubric,
    compute_score_from_decisions,
    count_exam_grids,
    index_answer_evidence,
    is_verbatim_quote,
    load_question_bundle,
    parse_hard_constraints,
    resolve_answer_evidence,
    truncate_by_exam_grids,
)
from scoring.universal_engine import (
    _separate_fixed_title,
    compile_universal_rubric,
    compute_universal_ledger,
)
from scoring.paper_pipeline import prefab_paper_question
from scoring.rubric_cache import load_or_build_rubric
from scoring.essay_engine import _resolve_reasoning_chains, grade_essay_answer


class ScoringMathTest(unittest.TestCase):
    def test_essay_adequate_argument_does_not_award_full_marks(self):
        rubric = {
            "硬约束": {"满分": 35, "最少字数": None, "字数上限": 1000},
            "立意要求": [{}, {}],
            "材料原文": [],
        }
        thesis = [{"编号": key, "状态": "完整"} for key in ("T1", "T2")]
        keys = ("A1", "A2", "A3", "A4", "B1", "B2")
        adequate_rows = thesis + [{"编号": key, "状态": "完整"} for key in keys]
        strong_rows = thesis + [{"编号": key, "状态": "充分"} for key in keys]
        first_chain = {"论据原句": "证据甲", "推理原句": "推理甲", "展开原句": "中间步骤甲"}
        second_chain = {"论据原句": "证据乙", "推理原句": "推理乙", "展开原句": "中间步骤乙"}
        with patch("scoring.essay_engine._check_answer", return_value=(adequate_rows, [first_chain])):
            adequate = grade_essay_answer(rubric, "有条理地说明观点并列举依据。", recalled=False)
        with patch("scoring.essay_engine._check_answer", return_value=(strong_rows, [first_chain])):
            one_chain = grade_essay_answer(rubric, "只写出一条充分论证。", recalled=False)
        with (patch("scoring.essay_engine._check_answer", return_value=(strong_rows, [first_chain, second_chain])),
              patch("scoring.essay_engine._review_high_band", return_value=([first_chain, second_chain], []))):
            strong = grade_essay_answer(rubric, "用关键机制充分论证判断。", recalled=False)
        rejected = {**second_chain, "展开原句": ""}
        with (patch("scoring.essay_engine._check_answer", return_value=(strong_rows, [first_chain, second_chain])),
              patch("scoring.essay_engine._review_high_band", return_value=([first_chain, rejected], []))):
            rejected_top = grade_essay_answer(rubric, "第二条被独立复核认定只有观点。", recalled=False)
        self.assertEqual(adequate["立意档位"], "二类")
        self.assertEqual(one_chain["立意档位"], "二类")
        self.assertLess(adequate["最终得分"], 28)
        self.assertLess(one_chain["最终得分"], 28)
        self.assertEqual(strong["立意档位"], "一类")
        self.assertEqual(strong["最终得分"], 35)
        self.assertEqual(rejected_top["立意档位"], "二类")

    def test_essay_does_not_count_copied_material_as_original_reasoning(self):
        material = "食品分储可以减少生熟食品交叉污染，从而降低就餐人员感染食源性疾病的风险。"
        rubric = {"硬约束": {"满分": 35, "最少字数": None, "字数上限": 1000}, "立意要求": [{}, {}], "材料原文": [material]}
        rows = [{"编号": key, "状态": "完整"} for key in ("T1", "T2", "A1", "A2", "A3", "A4", "B1", "B2")]
        chain = {"论据原句": material, "推理原句": material, "展开原句": material}
        with patch("scoring.essay_engine._check_answer", return_value=(rows, [chain])):
            grade = grade_essay_answer(rubric, material, recalled=False)
        self.assertEqual(grade["立意档位"], "四类")
        self.assertEqual(grade["论证链"], [])

    def test_reasoning_chain_cannot_reuse_the_same_reason(self):
        text = "原工艺耐用。保留工艺可以减少开胶。另一组样本也耐用。"
        spans = index_answer_evidence(text)
        rows = [
            {"论据编号": [1], "推理编号": [2], "说明": "解释质量关系"},
            {"论据编号": [3], "推理编号": [2], "说明": "重复用同一个推理"},
        ]
        self.assertEqual(len(_resolve_reasoning_chains(rows, text, spans)), 1)
        rows[1]["推理编号"] = [99]
        self.assertIsNone(_resolve_reasoning_chains(rows, text, spans))

    def test_rubric_cache_builds_once_for_concurrent_readers(self):
        with TemporaryDirectory() as temp, ThreadPoolExecutor(max_workers=2) as executor:
            entered, release = Event(), Event()

            def build():
                entered.set()
                self.assertTrue(release.wait(3))
                return {"满分": 10}

            def unexpected_build():
                self.fail("相同题目已在预制，同版本不可重复预制")

            first = executor.submit(load_or_build_rubric, temp, ["同一道题"], build)
            self.assertTrue(entered.wait(3))
            second = executor.submit(load_or_build_rubric, temp, ["同一道题"], unexpected_build)
            release.set()
            self.assertEqual(first.result(), {"满分": 10})
            self.assertEqual(second.result(), first.result())

    def test_question_cache_rebuilds_after_material_change(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            question = root / "第1题.md"
            material = root / "材料1.md"
            question.write_text("[材料 1](材料1.md)\n\n## 题目\n说明相关做法。（10分）\n", encoding="utf-8")
            material.write_text("# 材料1\n\n第一版材料。", encoding="utf-8")
            first_rubric = {
                "题型": 1, "分类理由": "概括做法", "阅卷规则": "按要点计分", "档位": [],
                "评分项": [{"名称": "具体做法", "类别": "内容", "分值": 10,
                         "细则": "按准确性给分", "材料依据": "资料1第1段"}],
            }
            updated_rubric = {**first_rubric, "阅卷规则": "按更新的材料计分"}
            with patch("scoring.paper_pipeline.call_llm_json", side_effect=[first_rubric, updated_rubric]) as build:
                params = {"ref_limit": 0, "cache_dir": root / "cache"}
                first = prefab_paper_question(question, **params)
                self.assertEqual(prefab_paper_question(question, **params), first)
                material.write_text("# 材料1\n\n更新的材料。", encoding="utf-8")
                self.assertEqual(prefab_paper_question(question, **params)["细则"], updated_rubric)
                self.assertEqual(build.call_count, 2)

    def test_fixed_title_is_checked_without_changing_body(self):
        rubric = {
            "字面指令槽位": {"字面指定标题": "保护与发展"},
            "外壳客观门禁列表": [{"门禁编号": "F_FIXED_TITLE"}, {"门禁编号": "F_THESIS"}],
        }
        body = "保护生态有助于培育长期产业。"
        for title, expected in (("《保护与发展》", True), ("地方经济的新路", False)):
            with self.subTest(title=title):
                filtered, visible, decision = _separate_fixed_title(rubric, f"{title}\n{body}")
                self.assertEqual(visible, body)
                self.assertEqual(filtered["外壳客观门禁列表"], [{"门禁编号": "F_THESIS"}])
                self.assertEqual(decision["F_FIXED_TITLE"], {"通过": expected, "答卷引文": title})
        self.assertEqual(_separate_fixed_title(rubric, body)[1], body)

    def test_exam_grid_counter_and_truncation(self):
        # 2026 占 2 格，20% 占 1+1=2 格（20占1格，%占1格）
        text = "2026年发布8个项目，占20%。"
        self.assertEqual(count_exam_grids("2026"), 2)
        self.assertEqual(count_exam_grids("8"), 1)
        self.assertGreater(count_exam_grids(text), 10)

        long_text = "甲" * 318 + "乙" * 80
        vis, over = truncate_by_exam_grids(long_text, 318)
        self.assertEqual(vis, "甲" * 318)
        self.assertEqual(over, "乙" * 80)

    def test_clean_reference_body_strips_notes(self):
        raw = "一、助力返岗。\n参考答案说明：各条要点并列。"
        self.assertEqual(clean_reference_body(raw), "一、助力返岗。")
        raw2 = "你我携手走近科学(正文部分共计354字)\n各位市民朋友："
        self.assertEqual(clean_reference_body(raw2), "你我携手走近科学\n各位市民朋友：")

    def test_verbatim_quote_handles_nbsp_and_rejects_hallucination(self):
        src = "\u00a0\u00a0为营造“公众参与科学”氛围，与上级部门 协同推进。"
        self.assertTrue(is_verbatim_quote('营造"公众参与科学"氛围,与上级部门协同推进', src))
        self.assertFalse(is_verbatim_quote("营造公众参与科学...协同推进", src))
        self.assertFalse(is_verbatim_quote("营造崇尚科学氛围", src))

    def test_numbered_evidence_returns_original_text(self):
        answer = "排查设施隐患。\n发现问题及时整改。"
        spans = index_answer_evidence(answer)
        self.assertEqual(resolve_answer_evidence({"证据编号": [2]}, answer, spans), "发现问题及时整改。")
        self.assertEqual(resolve_answer_evidence({"证据编号": [1, 2]}, answer, spans), answer)
        self.assertEqual(resolve_answer_evidence({"证据编号": [3]}, answer, spans), "")

    def test_core_requirement_changes_score_with_same_content(self):
        rubric = {
            "硬约束": {"满分": 20, "字数上限": 400},
            "分池配置": {"各任务块封顶配额": {"正文": 18}, "内容总分池": 18, "实证外壳门禁分池": 2},
            "采分卡片表": [{
                "卡号": "M1", "所属簇": "G1", "对应任务块": "正文", "全局段号": 1,
                "卡片性质": "做法", "计分角色": "内容", "核心语义": "有参与渠道", "原文": "平台报名",
                "满分值": 18,
            }],
            "功能簇封顶表": [{"簇编号": "G1", "簇名称": "报名", "所属任务块": "正文", "簇封顶分": 18}],
            "外壳客观门禁列表": [{
                "门禁编号": "R1", "名称": "有号召力", "分值": 2, "核查标准": "能让公众理解如何参与",
                "核心任务": True,
            }],
        }
        facts = {"M1": {"状态": "完整", "答卷引文": "平台报名"}}
        for level, expected in (("达成", 20), ("部分达成", 19), ("未达成", 16)):
            with self.subTest(level=level):
                result = compute_universal_ledger(rubric, 150, "", {
                    "卡片判定": facts,
                    "门禁判定": {"R1": {"通过": level == "达成", "程度": level,
                                           "答卷引文": "平台报名", "说明": "按正文核对参与路径"}},
                })
                self.assertEqual(result["分项得分"]["内容总得分"], 18)
                self.assertEqual(result["最终得分"], expected)
                if level != "达成":
                    self.assertTrue(any("按正文核对参与路径" in line for line in result["精准扣分点评"]))

    def test_load_bundle_filters_specified_materials(self):
        pbc = Path(r"D:\pythonproject\shenlun\试卷\事业单位\中国人民银行\2020年11月7日中国人民银行招聘考试《申论》\第1题.md")
        bundle = load_question_bundle(pbc)
        mats = {p["材料名"] for p in bundle["材料段落"]}
        self.assertEqual(mats, {"资料2", "资料3", "资料4"})
        self.assertEqual(len(bundle["参考答案"]), 1)
        self.assertNotIn("参考答案说明", bundle["参考答案"][0]["正文"])

    def test_single_answer_overcompression_and_gongwen_scoring(self):
        constraints = parse_hard_constraints("拟写发布词，呼吁公众积极报名参加。（20分）要求：不超过450字。")
        self.assertEqual(constraints["满分"], 20.0)
        self.assertEqual(constraints["字数上限"], 450)
        self.assertEqual(constraints["物理截断格数"], 477)

        spec = {
            "题型类别": "公文应用文",
            "文种子类": "讲话发言发布词类",
            "发文身份": "S市科委工作人员",
            "目标受众": "社会公众",
            "核心答题目标": "发布项目并呼吁公众报名",
            "是否要求号召动员": True,
            "强制归类维度": [],
            "受众错位陷阱提示": "勿将科学家申报立项混为市民报名",
        }
        card_pack = {
            "功能簇列表": [
                {"簇编号": "G1", "簇名称": "核心信息", "簇说明": "项目发布"},
                {"簇编号": "G2", "簇名称": "案例激励", "簇说明": "多个例子"},
            ],
            "卡片": [
                {"卡号": "M01", "所属簇": "G1", "全局段号": 1, "主体": "科委", "面向或惠及对象": "公众", "信息类型": "目的用意", "题干相关性": "直接相关", "核心语义": "营造科学氛围", "原文": "营造科学氛围"},
                {"卡号": "M02", "所属簇": "G2", "全局段号": 2, "主体": "豆豆", "面向或惠及对象": "公众", "信息类型": "个体案例与体验", "题干相关性": "直接相关", "核心语义": "离科学近了", "原文": "离科学近了"},
                {"卡号": "M03", "所属簇": "G2", "全局段号": 2, "主体": "许师傅", "面向或惠及对象": "公众", "信息类型": "个体案例与体验", "题干相关性": "直接相关", "核心语义": "吸收知识养分", "原文": "吸收知识养分"},
                {"卡号": "M04", "所属簇": "G2", "全局段号": 2, "主体": "小郭", "面向或惠及对象": "公众", "信息类型": "个体案例与体验", "题干相关性": "直接相关", "核心语义": "公众科学家身份", "原文": "公众科学家身份"},
            ],
        }
        aligned = [{"答案编号": 0, "命中列表": [{"卡号": "M01", "程度": "完整", "答案引文": "营造氛围"}], "无依据包装词": []}]
        rubric = compile_quantitative_rubric(
            {"题目路径": "x", "题干与要求": "y"}, spec, constraints, card_pack, aligned, "单答案测试"
        )
        # 案例簇封顶应小于簇内三张案例卡之和，防止狂抄故事刷分
        g2_rule = [r for r in rubric["功能簇封顶表"] if r["簇编号"] == "G2"][0]
        self.assertLess(g2_rule["簇封顶分"], g2_rule["簇内卡片总分"])

        # 测试严重受众错位且无公众呼吁的鸡汤卷不会骗到号召分
        bad_pass = {
            "卡片判定": {k: {"状态": "未写", "答卷引文": ""} for k in ["M01", "M02", "M03", "M04"]},
            "公文客观核对": {
                "有标题或首句明确发布宣告": True,
                "有独立标题": True,
                "有受众称呼或首段直接面向受众说话": True,
                "有落款单位与日期": False,
                "落款单位或年份是否写错": False,
                "明确呼吁目标受众报名或行动": False,
                "呼吁引文": "",
                "受众错位情况": "严重错位",
                "错位引文": "诚邀科学家申报立项",
            },
            "归类核对": None,
        }
        res = compute_score_from_decisions(rubric, 350, "", bad_pass)
        self.assertEqual(res["分项得分"]["场景号召与受众得分"], 0.0)
        self.assertEqual(res["最终得分"], 2.0)

    def test_universal_empirical_shell_and_block_quota_cap(self):
        # 测试：报刊短评（0%称呼、0%落款、100%指定标题）自动不考称呼落款；双块题目（原因+建议）只写原因无法突破原因池上限
        constraints = parse_hard_constraints("阐述原因并提出建议。（20分）不超过450字。")
        spec = {
            "字面任务块": ["推进缓慢的原因", "下一步工作建议"],
            "是否含问题反推对策": True,
            "是否要求明确观点表态": False,
            "字面指定标题": None,
            "题干是否明示写标题或文种名": False,
            "题干字面要求免格式": False,
            "题干明示目标受众": None,
            "题干字面要求呼吁号召": False,
        }
        card_pack = {
            "功能簇列表": [
                {"簇编号": "G1", "簇名称": "原因簇", "所属任务块": "推进缓慢的原因", "簇说明": "病因"},
                {"簇编号": "GS1", "簇名称": "建议簇", "所属任务块": "下一步工作建议", "簇说明": "对策"},
            ],
            "全部卡片": [
                {"卡号": "M01", "所属簇": "G1", "对应任务块": "推进缓慢的原因", "全局段号": 1, "主体": "村民", "卡片性质": "问题与原因", "是否具体案例细节": False, "核心语义": "担心收不回房", "原文": "收不回来"},
                {"卡号": "M02", "所属簇": "G1", "对应任务块": "推进缓慢的原因", "全局段号": 2, "主体": "村内", "卡片性质": "问题与原因", "是否具体案例细节": False, "核心语义": "道路狭窄坑洼", "原文": "坑洼不平"},
                {"卡号": "S01", "所属簇": "GS1", "对应任务块": "下一步工作建议", "全局段号": 1, "主体": "建议", "卡片性质": "反推对策措施", "是否具体案例细节": False, "核心语义": "明确收房与续租细则", "原文": "收不回来"},
                {"卡号": "S02", "所属簇": "GS1", "对应任务块": "下一步工作建议", "全局段号": 2, "主体": "建议", "卡片性质": "反推对策措施", "是否具体案例细节": False, "核心语义": "修缮道路补齐配套", "原文": "坑洼不平"},
            ],
        }
        aligned = [
            {
                "答案编号": i,
                "命中列表": [
                    {"卡号": "M01", "程度": "完整", "答案引文": "a"},
                    {"卡号": "M02", "程度": "完整", "答案引文": "b"},
                    {"卡号": "S01", "程度": "完整", "答案引文": "c"},
                    {"卡号": "S02", "程度": "完整", "答案引文": "d"},
                ],
                "外观特征": {
                    "首行有独立标题": False,
                    "有顶格受众称呼": False,
                    "文末有落款单位或日期": False,
                    "首段有总括破题或明确表态句": False,
                    "文末有行动号召祈使句": False,
                    "有分条或分块标识": True,
                    "各任务块估算字数": {"推进缓慢的原因": 180, "下一步工作建议": 240},
                },
            }
            for i in range(6)
        ]
        # 机构集体漏写有材料依据的对策时，采分点仍应保留。
        for row in aligned:
            row["命中列表"] = [hit for hit in row["命中列表"] if hit["卡号"] != "S02"]
        rubric = compile_universal_rubric(
            {"题目路径": "x", "题干与要求": "阐述原因并提出建议"}, spec, constraints, card_pack, aligned
        )
        # 6份答案均无标题称呼落款，仅触发分块条理门禁(1.0分)
        self.assertEqual(len(rubric["外壳客观门禁列表"]), 1)
        self.assertEqual(rubric["外壳客观门禁列表"][0]["门禁编号"], "F_STRUCTURE")
        missed_by_references = next(card for card in rubric["采分卡片表"] if card["卡号"] == "S02")
        self.assertGreater(missed_by_references["满分值"], 0)

        # 模拟偏科考生：把原因全写满，但建议1字未写 -> 绝不能突破原因块配额上限(8.0~9.0分)
        only_cause_pass = {
            "卡片判定": {
                "M01": {"状态": "完整", "答卷引文": "担心收不回房"},
                "M02": {"状态": "完整", "答卷引文": "道路坑洼"},
                "S01": {"状态": "未写", "答卷引文": ""},
                "S02": {"状态": "未写", "答卷引文": ""},
            },
            "门禁判定": {"F_STRUCTURE": {"通过": False, "答卷引文": ""}},
            "零分冗余原句": ["这是一段白占格子的废话寒暄"],
        }
        ledger = compute_universal_ledger(rubric, 220, "", only_cause_pass)
        cause_quota = rubric["分池配置"]["各任务块封顶配额"]["推进缓慢的原因"]
        self.assertLessEqual(ledger["最终得分"], cause_quota)
        self.assertTrue(any("下一步工作建议" in line for line in ledger["精准扣分点评"]))
        self.assertTrue(any("无效占格" in line for line in ledger["精准扣分点评"]))


if __name__ == "__main__":
    unittest.main()
