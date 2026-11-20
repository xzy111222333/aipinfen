import hashlib
import json
import re
from pathlib import Path

from scoring.engine import load_question_bundle, parse_hard_constraints
from scoring.paper_pipeline import is_essay_question
from scoring.rubric_cache import save_json
from scoring.run_batch_calibration import load_recalled_paper
from scoring.run_corpus_calibration import _group_samples, _paper_identity
from scoring.universal_engine import select_universal_references


ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval"
PAPERS = ROOT / "试卷" / "公务员" / "国考"
TEST_PAPERS = {("2020", "地市"), ("2021", "副省"), ("2022", "副省"),
               ("2024", "地市"), ("2024", "行政执法"), ("2026", "副省")}

SELECTION = {
    "归纳概括": [(2020, "副省", 1), (2021, "地市", 1), (2022, "行政执法", 1),
                 (2026, "地市", 2), (2026, "副省", 1)],
    "综合分析": [(2021, "地市", 2), (2022, "行政执法", 2), (2025, "副省", 1),
                 (2026, "行政执法", 2), (2021, "副省", 1), (2026, "副省", 2)],
    "提出对策": [(2021, "地市", 4), (2025, "副省", 4), (2023, "地市", 3), (2026, "副省", 4)],
    "公文": [(2020, "副省", 5, "讲话稿"), (2026, "地市", 3, "发布词"),
             (2025, "行政执法", 3, "指南"), (2022, "行政执法", 5, "公开信"),
             (2020, "地市", 3, "情况介绍提纲"), (2024, "行政执法", 3, "情况报告")],
    "短评": [(2023, "地市", 4), (2021, "副省", 4), (2026, "副省", 3)],
    "大作文": [(2022, "地市", 5), (2025, "副省", 5), (2023, "行政执法", 5),
               (2026, "地市", 5), (2024, "行政执法", 5), (2026, "副省", 5)],
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"已锁定且当前输入不一致：{path}")
        return
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def load_bundle(question):
    return load_question_bundle(ROOT / question["path"], include_all_materials=question["genre"] == "大作文",
                                exclude_human_scored=True)


def main():
    files = sorted(file for pack in ROOT.glob("国考申论*MD整合包") for file in (pack / "完整答案").glob("*.md"))
    groups, failures = _group_samples(PAPERS, files)
    if failures:
        raise ValueError(failures)
    papers = {_paper_identity(path.name): path for path in groups}
    splits = {paper.name: "test" if identity in TEST_PAPERS else "dev" for identity, paper in papers.items()}
    real = []
    for paper, paths in sorted(groups.items()):
        for path in paths:
            recorded, complete, _, answers = load_recalled_paper(path)
            real.append({"path": path.relative_to(ROOT).as_posix(), "paper": paper.name,
                         "split": splits[paper.name], "recorded": recorded, "complete": complete})
    questions, bases = [], []
    for genre, selections in SELECTION.items():
        for year, track, number, *form in selections:
            paper = papers[str(year), track]
            path = paper / f"第{number}题.md"
            question = {"id": f"Q{len(questions) + 1:02}", "path": path.relative_to(ROOT).as_posix(),
                        "paper": paper.name, "split": splits[paper.name], "genre": genre,
                        "form": form[0] if form else None}
            bundle = load_bundle(question)
            stem = bundle["题干与要求"]
            if is_essay_question(stem) != (genre == "大作文"):
                raise ValueError(f"题型标记与原题不符：{path}")
            limits = parse_hard_constraints(stem)
            refs = select_universal_references(bundle["参考答案"], None)
            question.update(maximum=limits["满分"], limit=limits["字数上限"], references=len(refs))
            questions.append(question)
            if not refs:
                raise ValueError(f"缺少机构或教师参考：{path}")
            offered = next(((p, a[number]) for p in groups[paper]
                            if (a := load_recalled_paper(p)[3]).get(number, "").strip()), None)
            second = {"source_path": offered[0].relative_to(ROOT).as_posix(), "text": offered[1],
                      "kind": "回忆答卷"} if offered else {"text": refs[1]["正文"], "kind": "机构参考", "source": refs[1]["来源"]}
            first = {"text": refs[0]["正文"], "kind": "机构参考", "source": refs[0]["来源"]}
            for n, base in enumerate((first, second), start=1):
                bases.append({"id": f"{question['id']}-B{n}", "question_id": question["id"], **base})

    seen = {}
    for question in questions:
        normalized = re.sub(r"[\W_]+", "", load_bundle(question)["题干与要求"])
        fingerprint = hashlib.sha256(normalized.encode()).hexdigest()
        if fingerprint in seen and seen[fingerprint] != question["split"]:
            raise ValueError(f"同题跨集合：{question['path']}")
        seen[fingerprint] = question["split"]
    if not any(sum(r["paper"] == paper for r in real) >= 2 for paper, split in splits.items() if split == "test"):
        raise ValueError("测试集没有同卷多份记录")
    write_once(EVAL / "split.json", {
        "papers": dict(sorted(splits.items())),
        "reason": "19 套试卷按整卷划分 13 套开发、6 套测试。测试覆盖三个卷别和多个年份，包含 2026 副省三份回忆卷。2024 地市和执法共享部分题目，放在同一侧；2023 地市和执法同样放在一侧。",
    })
    write_once(EVAL / "questions.json", questions)
    write_once(EVAL / "bases.json", bases)
    real_dir = EVAL / "real"
    real_dir.mkdir(exist_ok=True)
    write_once(real_dir / "index.json", real)
    print(f"锁定 {len(splits)} 套试卷、{len(questions)} 道题、{len(bases)} 份底稿、{len(real)} 份记录卷")


if __name__ == "__main__":
    main()
