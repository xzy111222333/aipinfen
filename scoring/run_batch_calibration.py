import argparse
import json
import re
from pathlib import Path

from .paper_pipeline import prefab_full_paper, process_full_paper
from .rubric_cache import DEFAULT_CACHE_DIR, save_json as save_report


def load_recalled_paper(path):
    raw = path.read_text(encoding="utf-8-sig")
    sections = re.split(r"(?m)^## 第(\d+)题\s*$", raw)
    header = sections[0]
    score = re.search(r"(?m)^#\s*(\d+(?:\.\d+)?)分", header)
    completeness = re.search(r"(?m)^- 完整度[：:]\s*(.+)", header)
    if not score or not completeness or len(sections) == 1:
        raise ValueError(f"答卷缺少总分、完整度或题号：{path}")
    answers = {}
    for number, body in zip(sections[1::2], sections[2::2]):
        number = int(number)
        if number in answers:
            raise ValueError(f"重复题号 {number}：{path}")
        lines = body.strip().splitlines()
        while lines and (not lines[0].strip() or lines[0].lstrip().startswith(">")):
            lines.pop(0)
        answers[number] = "\n".join(lines).strip()
    description = completeness[1]
    incomplete_numbers = {int(value) for value in re.findall(r"第\s*(\d+)\s*题", description)}
    return float(score[1]), description.startswith("完整答案"), incomplete_numbers, answers


def evaluate_recalled_paper(prepared, path):
    recorded, complete, missing, answers = load_recalled_paper(path)
    result = process_full_paper(prepared, answers, missing_content=missing, complete=complete)
    predicted = result["总分"]
    details = result["分项详情"]
    first_pass = None
    if predicted is not None:
        first_pass = sum((row.get("双盲审计") or {}).get("正序独立分", row["最终得分"]) for row in details.values())
    delta = round(predicted - recorded, 2) if predicted is not None else None
    return {
        "答卷": path.name,
        "记录总分": recorded,
        "预测总分": predicted,
        "首轮总分": first_pass,
        "分差": delta,
        "相对偏差": round(abs(delta) / recorded, 4) if delta is not None and recorded else None,
        "各题得分": {n: row["最终得分"] for n, row in details.items()},
        "分项详情": details,
    }


def main():
    parser = argparse.ArgumentParser(description="运行申论回忆卷批量校验")
    parser.add_argument("paper", type=Path, help="题干与材料所在的试卷目录")
    parser.add_argument("answers", type=Path, nargs="+", help="同一套试卷的回忆答卷 Markdown")
    parser.add_argument("--output", required=True, type=Path, help="输出报告 JSON 路径")
    parser.add_argument("--ref-limit", type=int, help="建表读取的参考答案上限，默认使用全部")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="题目评分细则缓存目录")
    parser.add_argument("--rubric-from", type=Path, help="从以前的报告读取已锁定的评分细则")
    args = parser.parse_args()
    if args.ref_limit is not None and args.ref_limit < 0:
        parser.error("--ref-limit 必须大于或等于 0")
    if args.rubric_from:
        previous = json.loads(args.rubric_from.read_text(encoding="utf-8"))
        if previous["试卷"] != args.paper.name:
            parser.error("评分细则对应的试卷与输入试卷不一致")
        prepared = {int(number): entry for number, entry in previous["预制细则"].items()}
        ref_limit = previous["建表参考答案上限"]
    else:
        prepared = prefab_full_paper(args.paper, ref_limit=args.ref_limit, cache_dir=args.cache_dir)
        ref_limit = args.ref_limit
    print(f"已锁定 {len(prepared)} 题评分细则", flush=True)
    results = []
    report = {"试卷": args.paper.name, "建表参考答案上限": ref_limit, "预制细则": prepared, "结果": results}
    save_report(args.output, report)
    for path in args.answers:
        result = evaluate_recalled_paper(prepared, path)
        results.append(result)
        save_report(args.output, report)
        print(json.dumps({key: value for key, value in result.items() if key != "分项详情"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
