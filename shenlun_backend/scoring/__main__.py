import argparse
import json
from pathlib import Path

from .service import grade_submission, prepare_question
from .reference_workflow import save


def main():
    parser = argparse.ArgumentParser(description="申论小题预制与逐卷批改")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("question", type=Path)
    prepare.add_argument("directory", type=Path)
    prepare.add_argument("--holdout", action="append", default=[])
    prepare.add_argument("--writing-count", type=int, choices=(3, 5), default=5)
    prepare.add_argument("--sheet", nargs=2, type=int, metavar=("WIDTH", "ROWS"))
    grade = commands.add_parser("grade")
    grade.add_argument("rubric", type=Path)
    grade.add_argument("answer_file", type=Path)
    grade.add_argument("--cache", type=Path, required=True)
    grade.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_question(args.question, args.directory, holdout_sources=args.holdout,
                         writing_count=args.writing_count, sheet=args.sheet)
        print(args.directory / "全流程结果.json")
    else:
        result = grade_submission(args.rubric, args.answer_file.read_text(encoding="utf-8-sig"), args.cache)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(args.output, result)
        print(json.dumps({"总分": result["总分"], "输出": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
