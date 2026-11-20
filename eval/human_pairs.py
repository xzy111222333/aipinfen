import argparse
import hashlib
import random
import re
from itertools import combinations
from pathlib import Path

from scoring.engine import clean_reference_body
from scoring.run_batch_calibration import load_recalled_paper
from scoring.universal_engine import select_universal_references

from .prepare_dataset import EVAL, ROOT, load_bundle, read_json, write_once


DIR = EVAL / "human_pairs"


def _pool(question, bases, real):
    number = int(re.search(r"\d+", Path(question["path"]).stem)[0])
    group = [clean_reference_body(b["text"]) for b in bases if b["question_id"] == question["id"]]
    for row in real:
        if row["paper"] == question["paper"]:
            answer = load_recalled_paper(ROOT / row["path"])[3].get(number)
            if answer:
                group.append(clean_reference_body(answer))
    refs = select_universal_references(load_bundle(question)["参考答案"], 5)
    group.extend(ref["正文"] for ref in refs)
    return list(dict.fromkeys(text for text in group if text))


def export_pairs():
    questions = read_json(EVAL / "questions.json")
    bases = read_json(EVAL / "bases.json")
    real = read_json(EVAL / "real" / "index.json")
    rng = random.Random(9051)
    available = []
    for question in questions:
        pairs = list(combinations(_pool(question, bases, real), 2))
        rng.shuffle(pairs)
        available.append((question, pairs))
    result = []
    for index in range(10):
        for question, pairs in available:
            if index >= len(pairs) or len(result) >= 100:
                continue
            left, right = pairs[index]
            if rng.getrandbits(1):
                left, right = right, left
            result.append({"id": f"P{len(result) + 1:03}", "question_id": question["id"], "a": left, "b": right})
    if len(result) < 100:
        raise ValueError("缺少 100 对可比较答卷")
    DIR.mkdir(exist_ok=True)
    write_once(DIR / "pairs.json", result)
    _write_form(DIR / "pairs_todo.md", result, questions)
    print(f"生成 {len(result)} 对匿名比较")


def _write_form(output, result, questions):
    by_id = {q["id"]: q for q in questions}
    lines = ["# 申论答卷人工比较", "", "每对填入 A更好 / B更好 / 相当，并标记把握 高 / 低。可以先填部分。", ""]
    for pair in result:
        question = by_id[pair["question_id"]]
        stem = load_bundle(question)["题干与要求"]
        lines.extend([f"## {pair['id']}", "", f"原题及材料链接 [{question['id']}](../../{question['path']})", "",
                      stem, "", "**答卷 A**", "", pair["a"], "", "**答卷 B**", "", pair["b"], "",
                      "- 比较：", "- 把握：", ""])
    if not output.exists():
        output.write_text("\n".join(lines), encoding="utf-8")


def export_variant_pairs():
    questions = read_json(EVAL / "questions.json")
    bases = read_json(EVAL / "bases.json")
    rng = random.Random(9052)
    result = []
    for split, count in (("dev", 7), ("test", 3)):
        remaining = [q for q in questions if q["split"] == split]
        rng.shuffle(remaining)
        for question in remaining:
            qid = question["id"]
            variants = [read_json(path) for path in sorted((EVAL / "variants" / qid).glob("*-V*.json"))]
            candidates = [row for row in variants if row["verified"] and row["type"] in {"V1", "V4"}]
            if not candidates:
                continue
            variant = rng.choice(candidates)
            other = next(b for b in bases if b["question_id"] == qid and b["id"] != variant["base_id"])
            left, right = variant["text"], clean_reference_body(other["text"])
            if rng.getrandbits(1):
                left, right = right, left
            result.append({"id": f"P{101 + len(result):03}", "question_id": qid, "a": left, "b": right})
            count -= 1
            if count == 0:
                break
        if count:
            raise ValueError(f"{split} 缺少可用于匿名比较的等义变体")
    write_once(DIR / "pairs_variants.json", result)
    _write_form(DIR / "pairs_variants_todo.md", result, questions)
    print(f"生成 {len(result)} 对匿名补充比较")


def read_pairs():
    pairs = read_json(DIR / "pairs.json")
    supplement = DIR / "pairs_variants.json"
    return pairs + (read_json(supplement) if supplement.is_file() else [])


def load_labels(path):
    text = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"(?m)^## (P\d{3})\s*$", text)
    pairs = {row["id"]: row for row in read_pairs()}
    results = []
    for pair_id, block in zip(blocks[1::2], blocks[2::2]):
        opinion = re.search(r"(?m)^- 比较[：:][ \t]*(.*)$", block)
        confidence = re.search(r"(?m)^- 把握[：:][ \t]*(.*)$", block)
        a, b = (opinion[1].strip() if opinion else ""), (confidence[1].strip() if confidence else "")
        if not a and not b:
            continue
        if pair_id not in pairs or a not in {"A更好", "B更好", "相当"} or b not in {"高", "低"}:
            raise ValueError(f"{pair_id} 的标记不完整或格式错误")
        results.append({"id": pair_id, "verdict": a, "confidence": b})
    return results


def answer_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def main():
    parser = argparse.ArgumentParser(description="导出或读取匿名人工比较")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--import-labels", type=Path, nargs="+")
    mode.add_argument("--include-variants", action="store_true")
    args = parser.parse_args()
    if args.import_labels:
        from scoring.rubric_cache import save_json

        label_index = {}
        for path in args.import_labels:
            for label in load_labels(path):
                key = label["id"]
                if key in label_index and label_index[key] != label:
                    raise ValueError(f"{key} 有冲突的人工标记")
                label_index[key] = label
        labels = list(label_index.values())
        save_json(DIR / "pairs_labeled.json", labels)
        print(f"已读取 {len(labels)} 对人工比较")
    elif args.include_variants:
        export_variant_pairs()
    else:
        export_pairs()


if __name__ == "__main__":
    main()
