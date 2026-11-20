import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from scoring.engine import MODEL_DEFAULT, _resolve_api_key, clean_reference_body, load_question_bundle, parse_hard_constraints
from scoring.rubric_cache import load_or_build_rubric, save_json
from scoring.run_batch_calibration import load_recalled_paper
from scoring.v0_frozen.essay_engine import grade_essay_answer, prefab_essay_question
from scoring.v0_frozen.paper_pipeline import is_essay_question
from scoring.v0_frozen.universal_engine import grade_universal_answer, prefab_universal_bundle, select_universal_references

from .baselines.direct_prompt import VERSION as DIRECT_VERSION, grade_direct
from .generate_variants import applicable_codes
from .human_pairs import DIR as PAIR_DIR, read_pairs
from .metrics import compare_real, mean_or_none, summarize_stability, summarize_variants, variant_result
from .prepare_dataset import EVAL, PAPERS, ROOT, load_bundle, read_json, write_once
from .telemetry import capture_network_usage, track_usage


def _band(score, maximum):
    ratio = score / maximum
    return 1 if ratio >= 0.8 else 2 if ratio >= 0.6 else 3 if ratio >= 0.35 else 4


def _v0_evidence(grade):
    evidence = []
    for row in grade.get("逐卡账本", []):
        evidence.append({"quote": row.get("答卷引文", ""), "reason": row["核心语义"],
                         "state": row["判定"], "source": row.get("材料原文", "")})
    for key in ("外壳门禁明细", "证据账本"):
        for row in grade.get(key, []):
            evidence.append({"quote": row.get("答卷引文", ""), "reason": row.get("说明", ""),
                             "state": row.get("状态", row.get("程度", row.get("通过")))})
    return evidence


class Baseline:
    def __init__(self, name, excluded):
        self.name = name
        self.excluded = excluded
        self.key = _resolve_api_key() if name != "B0" else None

    def prepare(self, question, fresh=False):
        if self.name == "B0":
            return None
        bundle = load_bundle(question)
        banned = self.excluded.get(question["path"], set())
        refs = select_universal_references([r for r in bundle["参考答案"]
                                            if _plain(r["正文"]) not in banned], 8)
        bundle["参考答案"] = refs
        if self.name == "B1":
            return bundle
        essay = question["genre"] == "大作文"

        def build():
            rubric = prefab_essay_question(bundle, self.key) if essay else prefab_universal_bundle(bundle, refs, self.key)
            return {"题型": "作文" if essay else "通用", "细则": rubric}

        fingerprint = [MODEL_DEFAULT, question["path"], bundle["题干与要求"], bundle["材料段落"], [] if essay else refs,
                       (EVAL / "v0_frozen.sha256").read_text(encoding="utf-8")]
        return build() if fresh else load_or_build_rubric(EVAL / "cache" / "baselines", fingerprint, build, engine="v0")

    def grade(self, question, prepared, text):
        if self.name == "B0":
            score = question["maximum"] * 0.65
            return {"score": score, "band": _band(score, question["maximum"]), "comments": []}
        if self.name == "B1":
            return grade_direct(question, prepared, text, api_key=self.key)
        rubric = prepared["细则"]
        if prepared["题型"] == "作文":
            grade = grade_essay_answer(rubric, text, api_key=self.key, recalled=False)
        else:
            grade = grade_universal_answer(rubric, text, api_key=self.key)
        score = grade["最终得分"]
        band_name = grade.get("立意档位")
        band = {"一类": 1, "二类": 2, "三类": 3, "四类": 4}.get(band_name) or _band(score, question["maximum"])
        return {"score": score, "band": band, "comments": _v0_evidence(grade),
                "overflow_excluded": bool(grade.get("被截断溢出文字")), "audit": grade}


def _checkpoint(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return read_json(path) if path.is_file() else {"grades": {}, "prep": {}}


def _verify_frozen():
    for line in (EVAL / "v0_frozen.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"冻结文件与原始校验和不一致：{name}")


def _score_question(engine, question, bases, variants, pair_answers, path):
    data = _checkpoint(path)
    grades = data["grades"]
    answers = {b["id"]: clean_reference_body(b["text"]) for b in bases}
    answers.update({v["variant_id"]: v["text"] for v in variants})
    answers.update(pair_answers)
    types = {v["type"] for v in variants}
    comparison = next(v["variant_id"] for v in variants if v["type"] == sorted(types)[0])
    stability_ids = [b["id"] for b in bases] + [comparison]
    rounds = {0: [(key, round_no) for key in answers for round_no in range(5 if key in stability_ids else 1)]}
    if engine.name == "B2":
        rounds.update({1: [(key, 0) for key in stability_ids], 2: [(key, 0) for key in stability_ids]})
    for rubric_round, jobs in rounds.items():
        pending = [(answer, repeat) for answer, repeat in jobs if f"{rubric_round}:{answer}:{repeat}" not in grades]
        if not pending:
            continue
        prep_key = str(rubric_round)
        if prep_key in data["prep"]:
            prepared = data["prep"][prep_key]["rubric"]
        else:
            with track_usage() as usage:
                prepared = engine.prepare(question, fresh=rubric_round != 0)
            data["prep"][prep_key] = {"rubric": prepared, "usage": usage}
            save_json(path, data)
        for answer, repeat in pending:
            with track_usage() as usage:
                score = engine.grade(question, prepared, answers[answer])
            grades[f"{rubric_round}:{answer}:{repeat}"] = {
                "question_id": question["id"], "answer_id": answer, "rubric_round": rubric_round,
                "maximum": question["maximum"], "essay": question["genre"] == "大作文",
                "repeat": repeat, "usage": usage, **score,
            }
            save_json(path, data)
    return data


def _score_real(engine, record, dev_mean, path):
    if path.is_file():
        return read_json(path)
    if engine.name == "B0":
        result = {**record, "prediction": dev_mean, "details": []}
    else:
        _, complete, _, answers = load_recalled_paper(ROOT / record["path"])
        if not complete:
            raise ValueError(f"整卷基线要求有完整作答记录：{record['path']}")
        details = []
        for number, text in sorted(answers.items()):
            qpath = PAPERS / record["paper"] / f"第{number}题.md"
            stem = load_question_bundle(qpath, exclude_human_scored=True)["题干与要求"]
            limits = parse_hard_constraints(stem)
            q = {"path": qpath.relative_to(ROOT).as_posix(), "maximum": limits["满分"],
                 "genre": "大作文" if is_essay_question(stem) else "小题"}
            with track_usage() as prep_usage:
                prepared = engine.prepare(q)
            with track_usage() as grade_usage:
                scored = engine.grade(q, prepared, clean_reference_body(text))
            details.append({"number": number, "prep_usage": prep_usage, "usage": grade_usage, **scored})
        result = {**record, "prediction": sum(row["score"] for row in details), "details": details}
    save_json(path, result)
    return result


def _plain(text):
    return "".join(clean_reference_body(text).split())


def _prepare_pairs(questions):
    labels_path = PAIR_DIR / "pairs_labeled.json"
    if not labels_path.is_file():
        return [], {}
    labels = {row["id"]: row["verdict"] for row in read_json(labels_path) if row["confidence"] == "高"}
    allowed = {q["id"] for q in questions}
    pairs, by_question = [], {}
    for pair in read_pairs():
        if pair["id"] not in labels or pair["question_id"] not in allowed:
            continue
        pairs.append({**pair, "verdict": labels[pair["id"]]})
        by_question.setdefault(pair["question_id"], {}).update({f"{pair['id']}-{side}": pair[side] for side in ("a", "b")})
    return pairs, by_question


def _excluded_references(questions, bases, pairs, real):
    index = {q["path"]: {_plain(b["text"]) for b in bases[q["id"]]} | {_plain(v) for v in pairs.get(q["id"], {}).values()}
             for q in questions}
    for row in real:
        _, _, _, answers = load_recalled_paper(ROOT / row["path"])
        for number, text in answers.items():
            path = (PAPERS / row["paper"] / f"第{number}题.md").relative_to(ROOT).as_posix()
            index.setdefault(path, set()).add(_plain(text))
    return index


def _load_ready(split):
    questions = [q for q in read_json(EVAL / "questions.json") if q["split"] == split]
    bases, variants = {}, {}
    all_bases = read_json(EVAL / "bases.json")
    for q in questions:
        qid = q["id"]
        bases[qid] = [b for b in all_bases if b["question_id"] == qid]
        bundle = load_bundle(q)
        variants[qid] = []
        for b in bases[qid]:
            for code in applicable_codes(q, bundle["题干与要求"]):
                path = EVAL / "variants" / qid / f"{b['id']}-{code}.json"
                if not path.is_file():
                    raise ValueError(f"评测集尚未生成完毕：{path.name}")
                variant = read_json(path)
                if variant["verified"]:
                    variants[qid].append(variant)
                elif variant["attempts"] < 3:
                    raise ValueError(f"评测集尚有待继续核验的变体：{path.name}")
        if len(bases[qid]) != 2 or not variants[qid]:
            raise ValueError(f"需要两份底稿及核验通过的变体：{qid}")
    return questions, bases, variants


def _run_jobs(tasks, workers):
    results, failures = {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(call, *args): key for key, call, args in tasks}
        for job in as_completed(jobs):
            key = jobs[job]
            try:
                results[key] = job.result()
                print(f"完成 {key}", flush=True)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                failures.append((key, str(exc)))
                print(f"失败 {key} {exc}", flush=True)
    if failures:
        raise ValueError(f"评测有未完成项，已记录的分数可接续：{failures}")
    return results


def _usage_summary(values):
    values = list(values)
    return {"n": len(values), **{key: sum(row.get(key, 0) for row in values)
                                 for key in ("requests", "input_tokens", "output_tokens", "missing_usage", "seconds")}}


def _summarize(questions, bases, variants, scored, real, pairs):
    compared, stability, evidence = [], [], []
    for q in questions:
        qid = q["id"]
        grades = scored[qid]["grades"]
        answers = {b["id"]: clean_reference_body(b["text"]) for b in bases[qid]}
        answers.update({v["variant_id"]: v["text"] for v in variants[qid]})
        for variant in variants[qid]:
            compared.append(variant_result(q, variant, grades[f"0:{variant['base_id']}:0"], grades[f"0:{variant['variant_id']}:0"]))
        repeats = {g["answer_id"] for g in grades.values() if g["repeat"] > 0}
        stability.extend(g for g in grades.values() if g["answer_id"] in repeats)
        for answer, text in answers.items():
            for item in grades[f"0:{answer}:0"]["comments"]:
                if item.get("reason") or item.get("quote"):
                    quote = item.get("quote", "")
                    exists = not quote or "".join(quote.split()) in "".join(text.split())
                    evidence.append({"answer_id": answer, "question_id": qid, "exact": exists, **item})
    cited = [row for row in evidence if row.get("quote")]
    usable_review = [row for row in evidence if row.get("reason") and row.get("state") not in {"完整", True}]
    sampled = sorted(usable_review, key=lambda row: hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).digest())[:20]
    human = []
    for pair in pairs:
        grades = scored[pair["question_id"]]["grades"]
        left, right = [grades[f"0:{pair['id']}-{side}:0"]["score"] for side in ("a", "b")]
        predicted = "A更好" if left > right else "B更好" if left < right else "相当"
        human.append({"id": pair["id"], "source": "human", "predicted": predicted,
                      "expected": pair["verdict"], "passed": predicted == pair["verdict"]})
    return {"M1": summarize_stability(stability), **summarize_variants(compared), **compare_real(real, human),
            "M6": {"n_quotes": len(cited), "verbatim_rate": mean_or_none(r["exact"] for r in cited), "human_review": sampled,
                   "human_pass_rate": None},
            "cost": {"question_prep": _usage_summary(p["usage"] for row in scored.values() for p in row["prep"].values()),
                     "question_grading": _usage_summary(g["usage"] for row in scored.values() for g in row["grades"].values()),
                     "real_grading": _usage_summary(g["usage"] for row in real for g in row["details"])},
            "comparisons": compared}


def main():
    parser = argparse.ArgumentParser(description="评分评测及逐题断点保存")
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--engine", choices=["B0", "B1", "B2"], nargs="+", default=["B0", "B1", "B2"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--run-name", required=True, help="例如 2026-11-18_baselines_dev；同名继续断点")
    parser.add_argument("--milestone", help="仅在任务书规定的留出测试节点使用")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("并发数至少为 1")
    if args.split == "test" and not args.milestone:
        parser.error("测试集只能使用里程碑运行")
    if not args.run_name.replace("-", "").replace("_", "").isalnum():
        parser.error("运行名称只能由字母、数字、下划线或连字符组成")
    _verify_frozen()
    questions, bases, variants = _load_ready(args.split)
    real = read_json(EVAL / "real" / "index.json")
    dev_mean = mean(r["recorded"] for r in real if r["split"] == "dev")
    real = [r for r in real if r["split"] == args.split]
    pairs, pair_answers = _prepare_pairs(questions)
    excluded = _excluded_references(questions, bases, pair_answers, real)
    reports = EVAL / "reports"
    reports.mkdir(exist_ok=True)
    run = EVAL / "runs" / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    dataset = [questions, bases, variants, real, pairs, DIRECT_VERSION, (EVAL / "v0_frozen.sha256").read_text(encoding="utf-8")]
    fingerprint = hashlib.sha256(json.dumps(dataset, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    write_once(run / "inputs.json", {"split": args.split, "milestone": args.milestone, "fingerprint": fingerprint})
    if args.split == "test":
        markers = EVAL / "milestones"
        markers.mkdir(exist_ok=True)
        if not args.milestone.replace("-", "").replace("_", "").isalnum():
            parser.error("无效里程碑名称")
        write_once(markers / f"{args.milestone}.json", {"run": args.run_name})
    summary_path = reports / f"{args.run_name}.json"
    all_scores = read_json(summary_path) if summary_path.is_file() else {"split": args.split, "engines": {}}
    for name in dict.fromkeys(args.engine):
        if name in all_scores["engines"]:
            continue
        engine = Baseline(name, excluded)
        folder = run / name
        questions_jobs = [(q["id"], _score_question, (engine, q, bases[q["id"]], variants[q["id"]],
                          pair_answers.get(q["id"], {}), folder / f"{q['id']}.json"))
                          for q in questions]
        real_jobs = [(record["path"], _score_real, (engine, record, dev_mean, folder / "real" / f"R{i:03}.json"))
                     for i, record in enumerate(real, start=1)]
        (folder / "real").mkdir(parents=True, exist_ok=True)
        with capture_network_usage():
            scored = _run_jobs(questions_jobs, args.workers)
            real_scores = list(_run_jobs(real_jobs, args.workers).values())
        all_scores["engines"][name] = _summarize(questions, bases, variants, scored, real_scores, pairs)
        save_json(summary_path, all_scores)
        write_markdown(reports / f"{args.run_name}.md", all_scores, questions, variants)


def _rate(metric):
    value = metric["rate"]
    return "不适用" if value is None else f"{value:.1%} ({metric['passed']}/{metric['total']})"


def write_markdown(path, summary, questions, variants):
    lines = ["# 申论评测基线成绩单", "", f"集合 {summary['split']}，题目 {len(questions)} 道，核验通过的变体 {sum(map(len, variants.values()))} 份。", "",
             "| 方法 | 同细则稳定性 | 重建细则稳定性 | 不变性 | 敏感性 | 同卷排序 | 整卷平均绝对分差 | 引文真实率 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, values in summary["engines"].items():
        mae = values["M5"]["mae"]
        verbatim = values["M6"]["verbatim_rate"]
        cells = [name, _rate(values["M1"]["fixed"]), _rate(values["M1"]["rebuilt"]),
                 *(_rate(values[m]) for m in ("M2", "M3", "M4")),
                 f"{mae:.2f}" if mae is not None else "无样本", f"{verbatim:.1%}" if verbatim is not None else "无引文"]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "B0 对整卷恒猜开发集记录均值，对单题恒给满分的 65%。B1 是温度为 0 的直接提示词。B2 是冻结版评分引擎。", ""])
    for name, values in summary["engines"].items():
        m1 = values["M1"]
        m5 = values["M5"]
        mae = f"{m5['mae']:.2f}" if m5["mae"] is not None else "无样本"
        bias = f"{m5['bias']:+.2f}" if m5["bias"] is not None else "无样本"
        lines.extend([f"## {name}", "", f"- M1 同细则 {_rate(m1['fixed'])}，重建细则 {_rate(m1['rebuilt'])}。",
                      f"- M2 {_rate(values['M2'])}，M3 {_rate(values['M3'])}，M4 {_rate(values['M4'])}。",
                      f"- M5 样本 {m5['n']}，平均绝对分差 {mae}，有符号分差 {bias}。",
                      f"- M6 原句样本 {values['M6']['n_quotes']}，原句真实率 {values['M6']['verbatim_rate']}，人工理由核验待标注。",
                      "", "| 成本 | 样本 | 请求 | 输入 token | 输出 token | 无 token 统计的请求 | 累计秒 |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for metric, label in (("question_prep", "预制"), ("question_grading", "单题批改"), ("real_grading", "真实卷题目批改")):
            cost = values["cost"][metric]
            lines.append(f"| {label} | {cost['n']} | {cost['requests']} | {cost['input_tokens']} | {cost['output_tokens']} | {cost['missing_usage']} | {cost['seconds']:.1f} |")
        lines.extend(["", "累计耗时包含并发调用，因此可以超过实际跑分时长。每次调用的记录见同名运行目录。", "", "### 变体细项", ""])
        lines.extend(f"- {code} {_rate(metric)}" for code, metric in values["by_type"].items())
        lines.extend(["", "### 最差案例候选", ""])
        failures = sorted((r for r in values["comparisons"] if not r["passed"]), key=lambda r: abs(r["diff_ratio"]), reverse=True)[:5]
        for row in failures:
            lines.append(f"- {row['id']} 属于 {row['type']}，底稿 {row['base_score']:g} 分，变体 {row['score']:g} 分；预期关系未达到。")
        lines.extend(["", "案例的归因须对照相应答卷和证据账本人工核查。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
