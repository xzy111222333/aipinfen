from copy import deepcopy
import unittest
from unittest.mock import patch

from rubric_workshop.blind_batch import aspect_rubric, hierarchy_rubric, layered_rubric, relation_rubric, catalog, evaluate, minimal_conditions, freeze_boundaries, task_rubric, specified_materials, validate

class BlindBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries=list(catalog())

    def test_paper_coverage(self):
        self.assertEqual(len(self.entries),61)
        self.assertEqual(len({p.parent for _,p,_ in self.entries}),15)
        self.assertEqual(sum(s["type"]==3 for _,_,s in self.entries),2)
        self.assertTrue(any(key=="2022执法-5" for key,_,_ in self.entries))

    def test_source_boundary(self):
        for _,_,src in self.entries:
            self.assertEqual(set(src),{"question","materials","type","budget"})
            for text in [src["question"],*src["materials"].values()]:
                self.assertNotIn("## 答案",text)
                self.assertNotIn("### 白鹭",text)

    def test_explicit_scope_overrides_archive_links(self):
        sources={key:s for key,_,s in self.entries}
        self.assertEqual({p.split("-")[0] for p in sources["2024执法-1"]["materials"]},{"材料1"})
        self.assertEqual({p.split("-")[0] for p in sources["2023副省-4"]["materials"]},{"材料4","材料5"})
        self.assertEqual(specified_materials("请根据给定资料2—4及给定材料6"),{"2","3","4","6"})

    def test_reject_invalid_budgets_and_quotes(self):
        src=dict(materials={"材料1-1":"居民投票决定改造范围"},budget={"内容":1})
        r=dict(paragraphs={"材料1-1":"计入决策"},tasks=["问做法"],limit="100字",
               items=[dict(zone="内容",name="居民决定范围",points=1,levels="全1缺0",
                           accept="居民表决改哪儿",reject="居民听通知",evidence={"材料1-1":"居民投票决定"})])
        validate(src,r)
        for field,value in (("points",2),("evidence",{"材料1-1":"居民获奖"})):
            bad=deepcopy(r)
            bad["items"][0][field]=value
            with self.assertRaises(ValueError):
                validate(src,bad)

    def test_frozen_units_score_only_evidenced_conditions(self):
        src = {"question": "谈谈治理方式。"}
        rubric = {"items": [{"name": "治理", "points": 1}]}
        units = {"units": {"治理": [[0.25, "语义", "提出方案"], [0.75, "语义", "表决方案"]]}}
        with patch("rubric_workshop.blind_batch.call_llm_json", return_value=units):
            frozen = freeze_boundaries(src, rubric)
        with patch("rubric_workshop.blind_batch.call_llm_json", return_value={"hits": {"1.2": "居民投票"}}):
            marks = evaluate(src, frozen, "居民投票")
        self.assertEqual(marks[0]["score"], 0.75)
        with patch("rubric_workshop.blind_batch.call_llm_json", return_value={"hits": {"1.2": "政府表决"}}):
            with self.assertRaises(ValueError):
                evaluate(src, frozen, "居民投票")

    def test_frozen_units_reject_wrong_sum(self):
        with patch("rubric_workshop.blind_batch.call_llm_json", return_value={"units": {"措施": [[0.5, "语义", "公开表决"]]}}):
            with self.assertRaises(ValueError):
                freeze_boundaries({}, {"items": [{"name": "措施", "points": 1}]})

    def test_task_routes_keep_source_and_budgets(self):
        src = {"type": 1, "question": "请概括做法。", "budget": {"内容": 1},
               "materials": {"材料1-1": "居民投票决定"}}
        response = {"limit": "100字", "items": [{"name": "决策", "zone": "内容",
                    "evidence": {"材料1-1": "居民投票"}, "units": [[0.5, "语义", "居民表决"], [0.5, "语义", "居民表决决定事项"]]}]}
        with patch("rubric_workshop.blind_batch.call_llm_json", return_value=response) as model:
            self.assertEqual(task_rubric(src)["items"][0]["points"], 1)
            first_prompt = model.call_args.args[0]
            task_rubric(dict(src, type=2))
            self.assertNotEqual(first_prompt, model.call_args.args[0])
            self.assertEqual(model.call_args.args[1], dict(src, type=2, materials={k: {"1": v} for k, v in src["materials"].items()}))
        with patch("rubric_workshop.blind_batch.call_llm_json", return_value=response):
            with self.assertRaises(ValueError):
                task_rubric(dict(src, budget={"内容": 2}))

    def test_minimum_condition_preserves_source_rubric_and_weights(self):
        src = {"question": "概括做法", "materials": {"材料1-1": "共同讨论"}}
        old = {"items": [{"name": "协商", "units": [[0.5, "语义", "居民共同讨论或参加议事会"]]}]}
        response = {"minimum": {"1.1": "居民协商"}}
        with patch("rubric_workshop.blind_batch.call_llm_json", return_value=response):
            revised = minimal_conditions(src, old)
        self.assertEqual(revised["items"][0]["units"], [[0.5, "语义", "居民协商"]])
        self.assertEqual(old["items"][0]["units"][0][2], "居民共同讨论或参加议事会")

    def test_aspect_budget_frozen_before_boundaries(self):
        src = {"question":"概括做法，不超过10字","materials":{"材料1":"村民集体协商"},"budget":{"内容":1}}
        aspect = dict(name="协商",zone="内容",points=1,claim="村民协商",evidence={"材料1":"集体协商"})
        with patch("rubric_workshop.blind_batch.call_llm_json",side_effect=[
                {"aspects":[aspect]}, {"bounds":{"协商":["征求意见","村民共议","村民接通知"]}},
                {"levels":{"1":[1,"征求意见","尚未形成共同决策"]}}]):
            r = aspect_rubric(src)
            self.assertEqual(r["content_chars"],4)
            self.assertEqual(evaluate(src,r,"征求意见")[0]["score"],0.5)
        aspect["claim"]="这一句超过了题目允许的十个字"
        with patch("rubric_workshop.blind_batch.call_llm_json",return_value={"aspects":[aspect]}):
            with self.assertRaises(ValueError):
                aspect_rubric(src)

    def test_only_abstract_meanings_enter_relation_budgeting(self):
        src={"question":"概括做法", "materials":{"材料1-1":"甲乙两村召开圆桌会议"}, "budget":{"内容":1}}
        extracted={"relations":[{"meaning":"跨村协商","sources":["1-1"]}], "excluded":{}}
        plan={"items":[{"evidence":{"关系1":"跨村协商"}}]}
        with patch("rubric_workshop.blind_batch.call_llm_json",return_value=extracted), patch(
                "rubric_workshop.blind_batch.aspect_rubric",return_value=plan) as budgeting:
            result=relation_rubric(src)
        self.assertEqual(budgeting.call_args.args[0]["materials"],{"关系1":"跨村协商"})
        self.assertEqual(result["items"][0]["evidence"],src["materials"])

    def test_layered_rubric_counts_entire_answer_before_freezing(self):
        src={"question":"不超过10字","materials":{"材料1-1":"两村共同商议"},"budget":{"内容":1}}
        row=["协商","内容",1,"跨村协商","两村协商",["1-1"],"两村代表商量","只征询意见","单方决定"]
        too_long={"answer":"这是超过十字的标题。两村协商","rows":[row]}
        valid={"answer":"共议。两村协商","rows":[row]}
        with patch("rubric_workshop.blind_batch.audit_boundaries",return_value=[]), patch("rubric_workshop.blind_batch.call_llm_json", side_effect=[too_long,valid]) as model:
            rubric=layered_rubric(src)
        self.assertEqual(model.call_count,2)
        self.assertEqual(rubric["answer_chars"],7)
        self.assertEqual(rubric["items"][0]["witness"],"两村协商")

    def test_sample_must_fit_with_paragraph_breaks(self):
        src={"type":2,"question":"不超过50字","materials":{"材料1-1":"两村共同商议"},"budget":{"内容":1}}
        row=["协商","内容",1,"跨村协商","两村协商",["1-1"],"两村代表商量","只征询意见","单方决定"]
        drafts=[{"rows":[row],"answer":s} for s in ("村民朋友们：\n请提意见。\n两村协商。","村民朋友们：\n请提意见，两村协商。")]
        with patch("rubric_workshop.blind_batch.audit_boundaries",return_value=[]), patch("rubric_workshop.blind_batch.call_llm_json",side_effect=drafts) as model:
            self.assertEqual(layered_rubric(src)["answer"],drafts[1]["answer"])
        self.assertEqual(model.call_count,2)

    def test_contradictory_anchor_is_found_without_revealing_expected_score(self):
        from rubric_workshop.blind_batch import audit_boundaries
        item=dict(name="协商",claim="两村共同商议",alternative="两村共同商量",partial="两村共同商量",reject="一方决定")
        def judge(prompt,data):
            return {"conflicts":[],"levels":{key:[0 if phrase=="一方决定" else 2,"检查协商主体"] for key,(_,phrase) in data["短答"].items()}}
        with patch("rubric_workshop.blind_batch.call_llm_json",side_effect=judge):
            errors=audit_boundaries({"question":"概括做法","materials":{"材料1":"两村共同商议"}},[item])
        self.assertEqual(len(errors),1)
        self.assertIn("预设1，试评2",errors[0])

    def test_content_budget_cannot_drop_a_prepared_core(self):
        src={"question":"概括做法，不超过25字","materials":{"关系1":"开展协商","关系2":"跟踪落实"},"budget":{"内容":1}}
        draft={"answer":"两村协商。","rows":[["协商","内容",1,"共同商议","两村协商",["关系1"],"两村商量","只征询意见","单方决定"]]}
        with patch("rubric_workshop.blind_batch.call_llm_json",return_value=draft):
            with self.assertRaisesRegex(ValueError,"关系2"):
                layered_rubric(src,audit_source=src)

    def test_examples_never_reach_budgeting_input(self):
        src={"question":"概括协商做法","materials":{"材料1-1":"甲乙两村召开圆桌会议"},"budget":{"内容":1}}
        hierarchy={"groups":[{"main":"村际合作","cores":[{"meaning":"两村共同商议","refs":["1-1"],
                    "illustrations":["圆桌会议"],"required_terms":[]}]}],"excluded":{}}
        frozen={"items":[{"evidence":{"关系1":"两村共同商议"}}]}
        with patch("rubric_workshop.blind_batch.call_llm_json",return_value=hierarchy), patch(
                "rubric_workshop.blind_batch.layered_rubric",return_value=frozen) as builder:
            result=hierarchy_rubric(src)
        self.assertEqual(builder.call_args.args[0]["materials"],{"关系1":"村际合作：两村共同商议"})
        self.assertEqual(result["unallocated_core_ids"],[])
        self.assertEqual(result["items"][0]["evidence"],src["materials"])
