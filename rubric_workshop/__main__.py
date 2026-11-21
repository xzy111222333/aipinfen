import argparse
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scoring.engine import (
    call_llm_json, count_exam_grids, load_question_bundle, llm_cache_model,
    parse_hard_constraints, truncate_by_exam_grids,
)
from scoring.rubric_cache import load_or_build_rubric, save_json

from .prompts import ANALYZE, COMPILE, GRADE, REFERENCES

PAPER = Path("试卷/公务员/国考/2026年国家公考《申论》题（地市级）")
OUTPUT = Path(__file__).parent / "2026国考地市"
BANDS = [kind + level for kind in ("一类", "二类", "三类", "四类") for level in ("上", "下")]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cite(quote, source):
    marks = str.maketrans({"“": '"', "”": '"', "‘": '"', "’": '"', "'": '"'})
    return bool(quote and "".join(quote.translate(marks).split()) in
                "".join(source.translate(marks).split()))


def stage(prompt, payload, check, attempts=3, max_tokens=20000):
    data = dict(payload)
    for attempt in range(attempts):
        try:
            result = call_llm_json(prompt, data, max_tokens=max_tokens)
        except RuntimeError as error:
            if attempt == attempts - 1 or not str(error).startswith(("API HTTP 520", "API HTTP 522")):
                raise
            continue
        except KeyError as error:
            if attempt == attempts - 1 or error.args != ("content",):
                raise
            continue
        try:
            check(result)
            return result
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
            if attempt == attempts - 1:
                raise ValueError(f"模型输出校验失败：{error}，返回字段：{list(result) if isinstance(result, dict) else type(result).__name__}") from error
            data["上一输出的问题"] = f"{error}。请完整重做，并按系统约定使用原字段名。"


def check_analysis(result, source):
    if result["题型"] in ("1", "2", "3"):
        result["题型"] = int(result["题型"])
    require(type(result["题型"]) is int and result["题型"] in (1, 2, 3), "题型只能是1/2/3")
    tasks = result["任务"]
    ids = [row["编号"] for row in tasks]
    require(ids and len(ids) == len(set(ids)), "任务编号重复或为空")
    for task in tasks:
        require(cite(task["题干引文"], source["题干"]), f"任务引文不在题干：{task}")
        require(task["交付"] and task["判定"] and task["所属"] in
                ("内容", "逻辑格式", "篇章约束", "档位"), f"任务字段错误：{task}")
    paragraphs = {row["全局段号"]: row["原文"] for row in source["材料"]}
    notes = result["材料提取"]
    require([row["段号"] for row in notes] == list(paragraphs), "按原顺序覆盖所有材料段落")
    for row in notes:
        require(cite(row["原句"], paragraphs[row["段号"]]), f"材料引文错误：{row}")
        require(set(row["对应任务"]) <= set(ids), f"未知任务：{row}")


def check_references(result, refs, analysis):
    if "参考核对" not in result and isinstance(result.get("参考答案"), list):
        result["参考核对"] = result.pop("参考答案")
    require([row["编号"] for row in result["参考核对"]] == [row["编号"] for row in refs],
            "参考答案未逐份核对")
    tasks = {row["编号"]: row["所属"] for row in analysis["任务"]}
    paragraphs = {row["段号"] for row in analysis["材料提取"]}
    for record, reference in zip(result["参考核对"], refs):
        require(record["候选"], "参考候选为空，若无适用内容也需摘录并注明排除")
        for candidate in record["候选"]:
            require(cite(candidate["摘录"], reference["正文"]), f"参考引文错误：{candidate}")
            require(set(candidate["依据"]) <= paragraphs and set(candidate["任务"]) <= set(tasks),
                    f"参考依据或任务不存在：{candidate}")
            if candidate["处理"] != "排除":
                format_only = any(tasks[key] in ("逻辑格式", "篇章约束") for key in candidate["任务"])
                require(candidate["处理"] in ("采纳", "合并", "可接受写法")
                        and candidate["任务"] and (candidate["依据"] or format_only), f"缺少采用依据：{candidate}")


def collect_reference_candidates(checks):
    groups = {}
    for record in checks:
        for row in record["候选"]:
            point = "".join(unicodedata.normalize("NFKC", row["归纳"]).split())
            key = (point, tuple(sorted(set(row["任务"]))), tuple(sorted(set(row["依据"]))))
            group = groups.setdefault(key, {"编号": f"R{len(groups) + 1}", "归纳": row["归纳"],
                                           "任务": list(key[1]), "依据": list(key[2]), "观察": []})
            observation = {"参考编号": record["编号"], "摘录": row["摘录"],
                           "初判": row["处理"], "理由": row["理由"]}
            if observation not in group["观察"]:
                group["观察"].append(observation)
    return list(groups.values())


def check_reference_decisions(rubric, candidates):
    decisions = rubric.get("候选裁决", {})
    require(isinstance(decisions, dict) and set(decisions) == {row["编号"] for row in candidates},
            "每个参考候选都须有最终裁决，包括初筛排除的候选")
    targets = ({row["编号"] for row in rubric["评分项"]} |
               {row["名称"] for row in rubric["档位"]})
    for decision in decisions.values():
        outcome, links = decision["处理"], decision["对应细则"]
        require(decision["理由"] and isinstance(links, list) and set(links) <= targets and
                ((outcome in ("计分", "同义", "替代") and links) or
                 (outcome in ("例证", "排除") and not links)),
                f"参考候选裁决无效：{decision}")


def check_rubric(rubric, source, analysis):
    tasks = {row["编号"]: row["所属"] for row in analysis["任务"]}
    paragraphs = {row["段号"] for row in analysis["材料提取"]}
    kind, full = analysis["题型"], source["满分"]
    require(set(rubric["任务覆盖"]) == set(tasks) and all(rubric["任务覆盖"].values()), "任务覆盖不完整")
    require(rubric["阅卷规则"] and rubric["计分理由"], "缺少阅卷或计分理由")
    items, bands = rubric["评分项"], rubric["档位"]
    if kind == 3:
        require(not items and [row["名称"] for row in bands] == BANDS, "作文必须使用四类八档")
        ceiling = full
        for row in bands:
            require(row["最高"] == ceiling and type(row["最低"]) is int and
                    0 <= row["最低"] <= ceiling and row["判据"], f"档位断裂：{row}")
            ceiling = row["最低"] - 1
        require(ceiling == -1, "档位必须覆盖零分")
        limits = rubric.get("限档", [])
        require(len({row["编号"] for row in limits}) == len(limits) and
                all(row["条件"] and row["最高档"] in BANDS for row in limits), "限档条件非法")
    else:
        require(items and not bands and len({row["编号"] for row in items}) == len(items), "评分项非法")
        totals = {"内容": 0, "逻辑格式": 0}
        covered = set()
        groups = defaultdict(float)
        categories = {}
        for item in items:
            levels = item["等级"]
            scores = [level["分值"] for level in levels]
            require(len(scores) >= 2 and all(type(score) in (float, int) and math.isfinite(score)
                    for score in scores) and scores[0] == 0 and scores == sorted(set(scores)) and
                    all(level["条件"] for level in levels), f"等级非法：{item}")
            require(item["任务"] and set(item["任务"]) <= set(tasks) and set(item["依据"]) <= paragraphs,
                    f"任务或依据非法：{item}")
            require(item["类别"] != "内容" or item["依据"], f"内容项缺材料依据：{item}")
            require(item["有效写法"] and item["误判边界"] and item["组"], f"缺判分边界：{item}")
            covered.update(item["任务"])
            totals[item["类别"]] += scores[-1]
            group = item["组"]
            groups[group] += scores[-1]
            categories.setdefault(group, set()).add(item["类别"])
        caps = rubric.get("组封顶", {})
        require(isinstance(caps, dict), "组封顶必须是组名到数字的对象；无封顶时用空对象或省略")
        for group, cap in caps.items():
            require(group in groups and len(categories[group]) == 1 and
                    type(cap) in (int, float) and 0 < cap <= groups[group], f"组上限非法：{group}")
            totals[next(iter(categories[group]))] -= groups[group] - cap
        score_roles = {"内容"} if kind == 1 else set(totals)
        require({key for key, role in tasks.items() if role in score_roles} <= covered, "有计分任务缺少评分项")
        content = full if kind == 1 else full * .75
        require(math.isclose(totals["内容"], content) and math.isclose(totals["逻辑格式"], full - content),
                f"分值分配错误：{totals}")
    examples = rubric["样例"]
    require(len(examples) >= 4 and len({row["名称"] for row in examples}) == len(examples), "样例不足或重名")
    for row in examples:
        low, high = row["预期区间"]
        require(0 <= low <= high <= full and row["答卷"] and row["说明"], f"试答无效：{row}")
    require(any(row["预期区间"][0] >= full * .7 and
                source["字数下限"] <= count_exam_grids(row["答卷"]) <= (source["字数上限"] or math.inf)
                for row in examples), "缺符合字数要求的高水平样卷")


def grade(source, analysis, rubric, answers):
    kind = analysis["题型"]
    limit = source["字数上限"]
    visible = [truncate_by_exam_grids(a, limit)[0] if limit and kind != 3 else a for a in answers]
    instructions = {key: value for key, value in rubric.items() if key not in ("样例", "候选裁决")}

    def check(result):
        require(len(result["判卷"]) == len(visible), "判卷数量错误")
        for text, row in zip(visible, result["判卷"]):
            if kind == 3:
                band = next((band for band in rubric["档位"] if band["名称"] == row["档位"]), None)
                require(band and type(row["分值"]) is int and band["最低"] <= row["分值"] <= band["最高"],
                        "作文分值不属于所选档位")
                limits = rubric.get("限档", [])
                checks = row.get("限档判断", []) if limits else []
                require([entry["编号"] for entry in checks] == [rule["编号"] for rule in limits],
                        "作文限档未逐条核对")
                for entry, rule in zip(checks, limits):
                    require(type(entry["命中"]) is bool and entry["理由"], "缺少限档判断")
                    require(not entry["命中"] or BANDS.index(row["档位"]) >= BANDS.index(rule["最高档"]),
                            "作文超过细则允许的最高档")
                marks = [row]
            else:
                marks = row["明细"]
                require([mark["编号"] for mark in marks] == [item["编号"] for item in rubric["评分项"]],
                        "判卷评分项未逐项覆盖")
                for mark, item in zip(marks, rubric["评分项"]):
                    require(type(mark["分值"]) in (int, float) and
                            mark["分值"] in [level["分值"] for level in item["等级"]], "模型选了未授权分值")
            for mark in marks:
                require(mark["理由"] and (not mark["引文"] or cite(mark["引文"], text))
                        and (mark["分值"] == 0 or mark["引文"]), f"判卷引文或理由无效：{mark}")
            if kind == 3:
                row["得分"] = row["分值"]
            else:
                groups = defaultdict(float)
                caps = rubric.get("组封顶", {})
                for item, mark in zip(rubric["评分项"], marks):
                    groups[item["组"]] += mark["分值"]
                row["得分"] = round(sum(min(score, caps.get(group, score))
                                      for group, score in groups.items()), 8)

    return stage(GRADE, {"题目材料": source, "题型": kind, "细则": instructions, "答卷": visible},
                 check, max_tokens=6000)["判卷"]


def render(path, saved):
    source, analysis, rubric = (saved[key] for key in ("题目材料", "分析", "细则"))
    lines = [f"# {path.stem}评分标准与细则", "",
             f"第 {analysis['题型']} 类 · 满分 {source['满分']:g} · {saved['状态']}", "",
             "## 题目", "", source["题干"], "", "## 阅卷规则", "",
             rubric["阅卷规则"], "", rubric["计分理由"], "", "## 逐项细则", ""]
    for item in rubric["评分项"]:
        lines.extend([f"### {item['名称']}（{item['类别']}，{item['组']}）", ""])
        lines.extend(f"- {row['分值']:g} 分：{row['条件']}" for row in item["等级"])
        lines.extend([f"- 有效写法：{item['有效写法']}", f"- 误判边界：{item['误判边界']}",
                      f"- 材料全局段号：{'、'.join(map(str, item['依据'])) or '按题干要求'}", ""])
    for band in rubric["档位"]:
        lines.extend([f"- {band['名称']}（{band['最低']}–{band['最高']} 分）：{band['判据']}"])
    for row in rubric.get("限档", []):
        lines.append(f"- {row['条件']}，最高{row['最高档']}。")
    for group, cap in rubric.get("组封顶", {}).items():
        lines.append(f"- 「{group}」组最高 {cap:g} 分。")
    lines.extend(["", "## 材料依据", ""])
    for note in analysis["材料提取"]:
        lines.extend([f"- 第 {note['段号']} 段「{note['原句']}」{note['作用']}。{note['限定']}"])
    lines.extend(["", "## 试评", ""])
    for row in saved["试评"]:
        low, high = row["预期"]
        lines.append(f"- {row['样例']}：预期 {low:g}–{high:g} 分，实得 {row['判卷']['得分']:g} 分。")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(question, output):
    draft = load_question_bundle(question, exclude_human_scored=True)
    if re.search(r"(?:写|撰写|写作)\s*(?:一篇)?\s*(?:议论文|文章)", draft["题干与要求"]):
        draft = load_question_bundle(question, include_all_materials=True, exclude_human_scored=True)
    stem = draft["题干与要求"]
    bounds = parse_hard_constraints(stem)
    source = {"题干": stem, "材料": draft["材料段落"], "满分": bounds["满分"],
              "字数上限": bounds["字数上限"], "字数下限": bounds["最少字数"] or 0}
    require(source["材料"] and source["满分"], "题干或材料缺失")
    def prepare(prompt, payload, check):
        return load_or_build_rubric(
            output / ".制作过程", [llm_cache_model(), prompt, payload],
            lambda: stage(prompt, payload, check), engine="rubric-workshop",
        )

    analysis = prepare(ANALYZE, source, lambda result: check_analysis(result, source))
    print(f"{question.stem}：完成判型和材料提取", flush=True)
    refs = [{"编号": index, **row} for index, row in enumerate(draft["参考答案"], 1)]
    checks = []
    for position in range(0, len(refs), 5):
        batch = refs[position:position + 5]
        anonymous = [{key: value for key, value in row.items() if key != "来源"} for row in batch]
        reviewed = prepare(REFERENCES, {"题目材料": source, "分析": analysis, "参考答案": anonymous},
                           lambda result: check_references(result, batch, analysis))
        checks.extend(reviewed["参考核对"])
    print(f"{question.stem}：已核对 {len(checks)} 份参考，开始制作和试评", flush=True)
    kind = analysis["题型"]
    full = source["满分"]
    if kind == 3:
        allocation = {"整篇档位": full}
    else:
        content = full if kind == 1 else full * .75
        allocation = {"内容": content, "逻辑格式": full - content}
    candidates = collect_reference_candidates(checks)
    payload = {"题目材料": source, "分析": analysis, "参考候选": candidates, "各类别固定满分": allocation}

    def check_compiled(result):
        check_rubric(result, source, analysis)
        check_reference_decisions(result, candidates)

    samples = None
    path = output / f"{question.stem}.json"
    for attempt in range(3):
        rubric = prepare(COMPILE, payload, check_compiled)
        if samples is None:
            samples = rubric["样例"]
        rubric["样例"] = samples
        saved = {"题目材料": source, "分析": analysis, "参考来源": [row["来源"] for row in refs],
                 "参考核对": checks, "参考候选": candidates, "细则": rubric, "试评": [], "模型": llm_cache_model(),
                 "状态": "待试评"}
        save_json(path, saved)
        render(path, saved)
        try:
            marks = grade(source, analysis, rubric, [row["答卷"] for row in samples])
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            saved["试评错误"] = str(error)
            save_json(path, saved)
            raise
        audit = [{"样例": example["名称"], "预期": example["预期区间"], "判卷": mark,
                  "通过": example["预期区间"][0] <= mark["得分"] <= example["预期区间"][1]}
                 for example, mark in zip(samples, marks)]
        saved["试评"] = audit
        saved["状态"] = "已锁定" if all(row["通过"] for row in audit) else "待修订"
        save_json(path, saved)
        render(path, saved)
        if saved["状态"] == "已锁定":
            return path
        payload["上一版细则"] = rubric
        payload["试评失败项"] = [row for row in audit if not row["通过"]]
        payload["修订要求"] = "查清给分条件为什么产生不符。保留本轮样例及预期，修订细则，勿通过调整预期放行。"
        print(f"{question.stem}：第 {attempt + 1} 轮试评发现分歧，正在修订细则", flush=True)

    raise ValueError(f"{question.stem}试评失败，已保存原因：{path}")


def main():
    parser = argparse.ArgumentParser(description="申论逐题细则制作")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make")
    make.add_argument("--paper", type=Path, default=PAPER)
    make.add_argument("--output", type=Path, default=OUTPUT)
    make.add_argument("--question", type=int, nargs="+")
    mark = commands.add_parser("grade")
    mark.add_argument("rubric", type=Path)
    mark.add_argument("answer", type=Path)
    blind = commands.add_parser('make-blind', help='只用题目和指定材料制作小题细则')
    blind.add_argument('key', help='五年题库中的题号')
    methods = blind.add_mutually_exclusive_group()
    methods.add_argument('--isolated-items', action='store_true', help='按题干确定独立判据并逐项复评')
    methods.add_argument('--independent-tasks', action='store_true', help='冻结前检验细节是否独立回答题意')
    methods.add_argument('--task-evidence', action='store_true', help='分配分值前核查每项独立意义和材料依据')
    blind.add_argument('--quantize', action='store_true', help='在任务证据实验中将独立语义转换为固定分值')
    blind.add_argument('--output', type=Path, required=True)
    trial = commands.add_parser('grade-blind', help='按无机构答案细则逐项试评')
    trial.add_argument('rubric', type=Path)
    trial.add_argument('answer', type=Path)
    args = parser.parse_args()
    if args.command == 'make-blind':
        from rubric_workshop.blind_batch import catalog, load, save
        from rubric_workshop.semantic_rubric import build as build_blind
        entry = next((src for key,_,src in catalog() if key == args.key), None)
        require(entry is not None and (entry['type'] in (1,2) or args.task_evidence), '题号无效或不是小题')
        require(not args.quantize or args.task_evidence, '--quantize 需与 --task-evidence 一起使用')
        target=args.output/args.key/'细则.json'
        if target.exists():
            existing=load(target)
            require(bool(existing['rubric'].get('binary_units')) == (args.quantize and entry['type'] != 3) and existing['source']==entry and (bool(existing['rubric'].get('isolated_items')) == args.isolated_items) and bool(existing['rubric'].get('task_evidence')) == args.task_evidence and (args.task_evidence or args.isolated_items or existing['rubric'].get('compact_minimum')) and bool(existing['rubric'].get('independent_tasks')) == args.independent_tasks, '输出位置已有另一版本细则')
        else:
            if args.task_evidence:
                from rubric_workshop.task_evidence import build
                rubric = build(entry, target.parent)
                if args.quantize:
                    from rubric_workshop.task_evidence import compile_units
                    save(target.parent/'给分草案.json', rubric)
                    rubric = compile_units(entry, rubric)
            elif args.isolated_items:
                from rubric_workshop.isolated_items import build as build_isolated
                rubric = build_isolated(entry)
            else:
                rubric = build_blind(entry,target.parent)
            if args.independent_tasks:
                from rubric_workshop.semantic_rubric import review_independent_tasks
                rubric = review_independent_tasks(entry,rubric)
            save(target,dict(source=entry,rubric=rubric))
        print(target)
        return 0
    if args.command == 'grade-blind':
        from rubric_workshop.blind_batch import evaluate, load, count_pure_exam_chars
        record=load(args.rubric)
        answer=args.answer.read_text(encoding='utf-8-sig')
        marks=evaluate(record['source'],record['rubric'],answer)
        print(json.dumps(dict(字数=count_pure_exam_chars(answer),得分=sum(m['score'] for m in marks),明细=marks),ensure_ascii=False,indent=2))
        return 0
    if args.command == "grade":
        saved = json.loads(args.rubric.read_text(encoding="utf-8"))
        require(saved["状态"] in ("已锁定", "人工核对"), "仅可使用已锁定或人工核对的细则判卷")
        result = grade(saved["题目材料"], saved["分析"], saved["细则"],
                       [args.answer.read_text(encoding="utf-8-sig")])
        print(json.dumps(result[0], ensure_ascii=False, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    questions = ([args.paper / f"第{number}题.md" for number in args.question]
                 if args.question else sorted(args.paper.glob("第[0-9]*题.md")))
    require(questions, "未找到题目文件")
    require(len(questions) == len(set(questions)), "题号重复")
    failed = 0
    with ThreadPoolExecutor(max_workers=5) as pool:
        pending = {pool.submit(build, question, args.output): question for question in questions}
        for future in as_completed(pending):
            try:
                print(f"已锁定：{future.result()}", flush=True)
            except Exception as error:
                print(f"{pending[future].stem}：{error}", file=sys.stderr, flush=True)
                failed += 1
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
