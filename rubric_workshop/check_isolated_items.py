"""回放三题与跨题对照，保留每笔得分证据。"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rubric_workshop.blind_batch import (
    OUT,
    load,
    save,
    catalog,
    evaluate,
    count_pure_exam_chars,
)
from rubric_workshop.joint_replay import cases_for
from rubric_workshop.isolated_items import build, compile_boundaries

FOLDER = OUT / "逐项独立核验"
KEYS = ("2023地市-1", "2025地市-1", "2026地市-3", "2022地市-1", "2025执法-2")


def run(key, calibrate=False, select_evidence=False, positive=False):
    folder = (OUT / "字限与独立得分" if calibrate else FOLDER) / key
    if select_evidence:
        folder = OUT / "冻结表取证复核" / key
    if positive:
        folder = OUT / "正向条件回测" / key
    src = next(s for k, _, s in catalog() if k == key)
    path = folder / "细则.json"
    if not path.exists():
        if select_evidence or positive:
            frozen = load(OUT / "字限与独立得分" / key / "细则.json")
            if frozen["source"] != src:
                raise ValueError("冻结题源不同")
            rubric = (
                compile_boundaries(src, frozen["rubric"])
                if positive
                else dict(frozen["rubric"], select_evidence=True)
            )
        else:
            rubric = build(src, calibrate=calibrate)
        save(path, dict(source=src, rubric=rubric))
    record = load(path)
    if record["source"] != src:
        raise ValueError("题源不一致")
    answers = (
        {
            k.removeprefix(key + "-").removesuffix(".json"): v
            for k, v in cases_for(key).items()
        }
        if key in KEYS[:3]
        else {}
    )
    if key == "2026地市-3":
        original = answers["答卷2"]
        phrase = "在线教程、直播或线下讲座"
        if original.count(phrase) != 1:
            raise ValueError("培训形式对照的原句不匹配")
        answers["删培训形式"] = original.replace(phrase, "")
    if key in KEYS[3:]:
        for f in (OUT / "删除检验跨题回测" / key).glob("*.json"):
            r = load(f)
            if "marks" in r:
                answers[f.stem] = r["answer"]
    if key == "2023地市-1":
        for n in ("独立概括稿", "独立行动稿", "套话稿", "概念误写稿"):
            answers[n] = load(OUT / "独立概括与行动表达" / key / (n + ".json"))[
                "answer"
            ]

    if key == "2024副省-2":
        answers = {
            n: load(OUT / "冻结表取证复核" / key / (n + ".json"))["answer"]
            for n in ("盲写原稿", "原段倒序")
        }

    def one(pair):
        name, answer = pair
        path = folder / (name + ".json")
        if path.exists():
            if load(path)["answer"] != answer:
                raise ValueError("答卷变更")
            return
        marks = evaluate(src, record["rubric"], answer)
        score = sum(m["score"] for m in marks)
        save(
            path,
            dict(
                answer=answer,
                chars=count_pure_exam_chars(answer),
                marks=marks,
                score=score,
            ),
        )
        print(key, name, score, flush=True)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(one, answers.items()))


def assess(folder):
    """只在回测端标记冻结表中的目标，保留每处额外浮动。"""
    rows = []

    def compare(key, first, second, expected, targets=(), content_only=False):
        path = folder / key
        items = load(path / "细则.json")["rubric"]["items"]
        left, right = [
            load(path / (name + ".json"))["marks"] for name in (first, second)
        ]
        if [i["name"] for i in items] != [m["name"] for m in left] or [
            m["name"] for m in left
        ] != [m["name"] for m in right]:
            raise ValueError("对照评分项不匹配")
        changes = {
            i["name"]: a["score"] - b["score"]
            for i, a, b in zip(items, left, right)
            if (not content_only or i["zone"] == "内容") and a["score"] != b["score"]
        }
        relevant = {i["name"] for i in items if any(t in i["name"] for t in targets)}
        if targets and not relevant:
            raise ValueError("冻结表缺少预期目标")
        if expected == "equal":
            passed = not changes
        elif expected == "remove":
            passed = (
                bool(changes)
                and changes.keys() <= relevant
                and all(d > 0 for d in changes.values())
            )
        elif expected == "zero":
            passed = all(
                m["score"] == 0 for m in right if m["name"] in relevant
            ) and any(m["score"] > 0 for m in left if m["name"] in relevant)
        rows.append(
            dict(
                key=key,
                first=first,
                second=second,
                expected=expected,
                targets=sorted(relevant),
                changes=changes,
                passed=passed,
            )
        )

    gep, garden, speech, travel, _ = KEYS
    compare(gep, "原稿", "等义转述", "equal")
    compare(gep, "原稿", "删融资", "remove", ("融资",))
    compare(garden, "答卷2", "答卷3", "equal")
    compare(garden, "答卷2", "答卷4", "zero", ("碳",))
    compare(garden, "答卷1", "删文旅", "equal")
    compare(garden, "答卷1", "删储碳", "remove", ("碳", "还田"))
    compare(speech, "答卷3", "答卷4", "equal", content_only=True)
    compare(speech, "答卷3", "答卷5", "zero", ("参与流程", "公众参与任务"))
    compare(speech, "答卷2", "删培训形式", "equal")
    compare(travel, "白鹭1", "删第三案例", "remove", ("动车所",))
    if (folder / "2024副省-2" / "原段倒序.json").exists():
        compare("2024副省-2", "盲写原稿", "原段倒序", "equal", content_only=True)
    save(folder / "逐项验收.json", rows)
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--select-evidence", action="store_true")
    parser.add_argument("--assess", type=Path, help="只核验指定目录已有的成对结果")
    parser.add_argument("--positive", action="store_true")
    parser.add_argument("keys", nargs="*")
    args = parser.parse_args()
    if args.assess:
        rows = assess(args.assess)
        print(sum(r["passed"] for r in rows), "/", len(rows))
        raise SystemExit(int(any(not r["passed"] for r in rows)))
    with ThreadPoolExecutor(5) as pool:
        list(
            pool.map(
                lambda key: run(
                    key, args.calibrate, args.select_evidence, args.positive
                ),
                args.keys or KEYS,
            )
        )
