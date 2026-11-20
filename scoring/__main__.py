import argparse
import json
from pathlib import Path

from .paper_pipeline import prefab_paper_question
from .rubric_cache import save_json
from .rubric_grading import grade_rubric_answer


def main():
    parser = argparse.ArgumentParser(description="制作或使用申论逐题阅卷细则")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare", help="制作并保存题型、细则和本题资料")
    build.add_argument("question", type=Path)
    build.add_argument("output", type=Path)
    grade = commands.add_parser("grade", help="用已保存的细则批改答卷")
    grade.add_argument("rubric", type=Path)
    grade.add_argument("answer", type=Path, help="仅包含考生本题作答的 UTF-8 文本")
    args = parser.parse_args()
    if args.command == "prepare":
        prepared = prefab_paper_question(args.question)
        save_json(args.output, prepared)
        print(f"第{prepared['题型']}类，细则已保存到 {args.output}")
    else:
        prepared = json.loads(args.rubric.read_text(encoding="utf-8-sig"))
        result = grade_rubric_answer(prepared, args.answer.read_text(encoding="utf-8-sig"))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
