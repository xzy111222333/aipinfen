"""冻结细则后回放既有真题答卷。用 --output 指定每轮独立目录。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rubric_workshop.blind_batch import OUT, catalog, aspect_rubric, layered_rubric, hierarchy_rubric, evaluate, load, minimal_conditions, relation_rubric, save, task_rubric


def cases_for(key):
    baseline = OUT / "边界计分复测"
    answers = {p.name: load(p)["answer"] for p in sorted(baseline.glob(key + "-答卷*.json"))}
    if key == "2025地市-1":
        for p in (OUT / "分类逐项复测").glob(key + "-删*.json"):
            answers[p.name] = load(p)["answer"]
    if key == "2026地市-3":
        original = answers[key + "-答卷2.json"]
        start = original.index("有意参与者")
        end = original.index("无论", start)
        answers[key + "-删报名指引.json"] = original[:start] + original[end:]
    return answers


def replay(key, output, minimum_from=None, aspects=False, relations=False, layered=False, hierarchy=False):
    source = next(s for name, _, s in catalog() if name == key)
    frozen = output / (key + ".json")
    if not frozen.exists():
        if hierarchy:
            rubric = hierarchy_rubric(source,checkpoint=output / (key+"-材料分层.json"))
        elif layered:
            rubric = layered_rubric(source)
        elif relations:
            rubric = relation_rubric(source)
        elif aspects:
            rubric = aspect_rubric(source)
        elif minimum_from is None:
            rubric = task_rubric(source)
        else:
            base = load(minimum_from / (key + ".json"))
            if base["source"] != source:
                raise ValueError("基础细则题源不同")
            rubric = minimal_conditions(source, base["rubric"])
        save(frozen, dict(source=source, rubric=rubric))
    record = load(frozen)
    if record["source"] != source:
        raise ValueError("复测题源与冻结题源不同")
    rubric = record["rubric"]
    if bool(rubric.get("aspect_levels")) != (aspects or relations or layered or hierarchy) or bool(rubric.get("cores")) != hierarchy or bool(rubric.get("layered")) != (layered or hierarchy) or bool(rubric.get("relations")) != relations:
        raise ValueError("已冻结细则的实验方法与本次运行不同，请指定对应目录")
    answers = cases_for(key)
    if key == "2024执法-4":
        answers[key+"-扩题.json"] = load(OUT / "首轮试评" / (key+".json"))["answers"][0]["answer"]
    if layered or hierarchy:
        answers[key+"-可写性验证.json"] = rubric["answer"]
        if key == "2025地市-1":
            answers.update({key+"-概括句.json": "种植中就地选用天然材料，采用环保生产方式。",
                            key+"-例证句.json": "用马蔺叶绑枝、木杆搭架、农家肥施肥。",
                            key+"-口号句.json": "加强建设，促进发展。"})
    for name, answer in answers.items():
        target = output / name
        if target.exists():
            if load(target)["answer"] != answer:
                raise ValueError("已存答卷与复测输入不同")
            continue
        marks = evaluate(source, rubric, answer)
        content = sum(m["score"] for m, i in zip(marks, rubric["items"]) if i["zone"] == "内容")
        save(target, dict(answer=answer, marks=marks, score=sum(m["score"] for m in marks), content=content))


def assess(output):
    comparisons = [
        ("葡萄园储碳转述", "2025地市-1-答卷2", "2025地市-1-答卷3", "content", "equal"),
        ("葡萄园保土与储碳", "2025地市-1-答卷2", "2025地市-1-答卷4", "carbon", "greater"),
        ("葡萄园删储碳", "2025地市-1-答卷1", "2025地市-1-删储碳", "score", "greater"),
        ("发布词流程转述", "2026地市-3-答卷3", "2026地市-3-答卷4", "content", "equal"),
        ("发布词删流程", "2026地市-3-答卷3", "2026地市-3-答卷5", "content", "greater"),
        ("发布词删报名指引", "2026地市-3-答卷2", "2026地市-3-删报名指引", "score", "greater"),
    ]
    if (output / "2025地市-1-概括句.json").exists():
        comparisons.extend([
            ("绿色种植概括与例证逐项等价", "2025地市-1-概括句", "2025地市-1-例证句", "content", "equal"),
            ("绿色种植与空口号", "2025地市-1-概括句", "2025地市-1-口号句", "content", "greater"),
            ("葡萄园删文旅", "2025地市-1-答卷1", "2025地市-1-删文旅", "score", "greater"),
        ])
    results = []
    for name, first, second, field, relation in comparisons:
        left = load(output / (first + ".json"))
        right = load(output / (second + ".json"))
        if field == "carbon":
            def carbon_score(record):
                rows = [m for m in record["marks"] if "碳" in m["name"]]
                if not rows:
                    raise ValueError("找不到可独立核查的储碳项")
                return sum(m["score"] for m in rows)
            left_score, right_score = carbon_score(left), carbon_score(right)
        else:
            left_score, right_score = left[field], right[field]
        key = first.rsplit("-", 1)[0]
        rubric = load(output / (key + ".json"))["rubric"]
        if [m["name"] for m in left["marks"]] != [m["name"] for m in right["marks"]]:
            raise ValueError("对照使用的评分项不同")
        changes = {}
        for a, b, item in zip(left["marks"], right["marks"], rubric["items"]):
            if field == "carbon" and "碳" not in item["name"]:
                continue
            if field == "content" and item["zone"] != "内容":
                continue
            if a["score"] != b["score"]:
                changes[a["name"]] = a["score"] - b["score"]
        passed = not changes if relation == "equal" else left_score > right_score
        retained = []
        if "-删" in second:
            for a, b in zip(left["marks"], right["marks"]):
                for u, v in zip(a["decisions"], b["decisions"]):
                    if u["score"] > v["score"] and u["quote"] and u["quote"] in right["answer"]:
                        retained.append(dict(item=a["name"], condition=u["condition"], quote=u["quote"]))
        results.append(dict(name=name, left=left_score, right=right_score, numeric_pass=passed,
                            changes=changes, retained_evidence_to_review=retained))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-from", type=Path, help="沿用指定目录已冻结的项目和预算，重新提炼最低条件")
    parser.add_argument("--aspects", action="store_true", help="先冻结任务方面与预算，再界定满分和部分分")
    parser.add_argument("--relations", action="store_true", help="归纳指定材料的任务关系后再制标")
    parser.add_argument("--layered", action="store_true", help="分离核心与例证，校验整份作答字数")
    parser.add_argument("--hierarchy", action="store_true", help="预算只接收抽取的核心关系，整篇计字")
    args = parser.parse_args()
    if sum(map(bool,(args.aspects,args.relations,args.layered,args.hierarchy,args.minimum_from))) > 1:
        parser.error("两种实验方法须分开运行")
    if args.minimum_from and args.minimum_from.resolve() == args.output.resolve():
        parser.error("基础细则与回测输出须为不同目录")
    keys = ["2025地市-1", "2026副省-1", "2026地市-3", "2023执法-3"]
    if args.layered or args.hierarchy:
        keys.append("2024执法-4")
    with ThreadPoolExecutor(4) as pool:
        futures = [pool.submit(replay, key, args.output, args.minimum_from, args.aspects, args.relations, args.layered, args.hierarchy) for key in keys]
        for future in futures:
            future.result()
    checks = assess(args.output)
    save(args.output / "对照检查.json", checks)
    failed = [row["name"] for row in checks
              if not row["numeric_pass"] or row["retained_evidence_to_review"]]
    if failed:
        parser.exit(1, "联合复测未通过，需逐项复核：" + "、".join(failed) + "\n")
