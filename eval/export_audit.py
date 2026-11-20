import argparse
import hashlib
from pathlib import Path

from .prepare_dataset import EVAL, read_json
from .variant_prompts import VARIANTS


def export_variants():
    output = EVAL / "variant_audit_todo.md"
    questions = {q["id"]: q for q in read_json(EVAL / "questions.json") if q["split"] == "dev"}
    bases = {b["id"]: b for b in read_json(EVAL / "bases.json")}
    variants = [read_json(path) for qid in questions for path in (EVAL / "variants" / qid).glob("*.json")]
    if output.exists():
        print(f"保留已导出的表：{output}")
        return
    lines = ["# 对照答卷抽查", "", "每类预先选三份开发集样本。查对照稿是否只改变声明的内容，保留其余要素。", ""]
    covered = 0
    for code, (instruction, _) in VARIANTS.items():
        group = sorted((v for v in variants if v["verified"] and v["type"] == code),
                       key=lambda v: hashlib.sha256(v["variant_id"].encode()).digest())[:3]
        lines.extend([f"## {code}", "", instruction, ""])
        if len(group) < 3:
            lines.extend([f"可审样本 {len(group)} 份，少于三份。", ""])
        covered += len(group)
        for variant in group:
            q = questions[variant["question_id"]]
            lines.extend([f"### {variant['variant_id']}", "", f"[题干和材料](../{q['path']})", "",
                          "**原稿**", "", bases[variant["base_id"]]["text"], "",
                          "**对照稿**", "", variant["text"], "",
                          "- 复核结论：", "- 观察依据：", ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"抽查表覆盖 {covered} 份对照稿")


def export_reasons(scorecard):
    scores = read_json(scorecard)
    questions = {q["id"]: q for q in read_json(EVAL / "questions.json")}
    bases = {b["id"]: b["text"] for b in read_json(EVAL / "bases.json")}
    output = EVAL / "reports" / f"{scorecard.stem}_reasons_todo.md"
    if output.exists():
        print(f"保留已导出的表：{output}")
        return
    lines = ["# 评分理由人工核查", "", "请按题目和答卷判断每条扣分理由是否成立，填成立或不成立。", ""]
    for engine, values in scores["engines"].items():
        rows = values["M6"]["human_review"]
        lines.extend([f"## {engine}", "", f"可审理由 {len(rows)} 条。", ""])
        for n, row in enumerate(rows, 1):
            qid, aid = row["question_id"], row["answer_id"]
            question_path = questions[qid]["path"]
            if aid in bases:
                answer = bases[aid]
            else:
                answer = read_json(EVAL / "variants" / qid / f"{aid}.json")["text"]
            lines.extend([f"### {engine}-{n:02d}（{aid}）", "", f"[题干和材料](../../{question_path})", "",
                          "**答卷**", "", answer, "", "**所引答卷句子**", "", row.get("quote") or "无引文",
                          "", "**评分理由**", "", row["reason"], "", "- 复核结论：", "- 观察依据：", ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"已导出理由核查表：{output}")


def main():
    parser = argparse.ArgumentParser(description="导出评测人工核查表")
    parser.add_argument("--scorecard", type=Path)
    args = parser.parse_args()
    if args.scorecard:
        export_reasons(args.scorecard)
    else:
        export_variants()


if __name__ == "__main__":
    main()
