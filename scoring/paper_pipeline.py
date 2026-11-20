import re
from pathlib import Path

from .engine import call_llm_json, llm_cache_model, load_question_bundle, parse_hard_constraints
from .essay_engine import grade_essay_answer
from .rubric_cache import DEFAULT_CACHE_DIR, load_or_build_rubric
from .rubric_grading import RUBRIC_PROMPT, grade_rubric_answer, validate_rubric
from .universal_engine import grade_universal_answer



def is_essay_question(stem):
    return bool(re.search(r"(?:写|撰写|写作)\s*(?:一篇)?\s*(?:议论文|文章)|以议论文形式", stem))


def prefab_full_paper(paper_dir, api_key=None, *, ref_limit=None, cache_dir=DEFAULT_CACHE_DIR):
    files = {}
    for path in Path(paper_dir).glob("第*题.md"):
        match = re.fullmatch(r"第(\d+)题", path.stem)
        if match:
            files[int(match[1])] = path
    if not files:
        raise ValueError(f"未找到题目文件：{paper_dir}")

    prepared = {}
    for number, path in sorted(files.items()):
        prepared[number] = prefab_paper_question(path, api_key, ref_limit=ref_limit, cache_dir=cache_dir)
    return prepared


def prefab_paper_question(path, api_key=None, *, ref_limit=None, cache_dir=DEFAULT_CACHE_DIR):
    bundle = load_question_bundle(path, exclude_human_scored=True)
    stem = bundle["题干与要求"]
    if is_essay_question(stem) or not re.search(r"(?:资料|材料)\s*\d+", stem):
        bundle = load_question_bundle(path, include_all_materials=True, exclude_human_scored=True)
    if not bundle["材料段落"]:
        raise ValueError(f"未找到题目所需材料：{path}")
    constraints = parse_hard_constraints(stem)
    prepared = {
        "题干与要求": stem, "材料段落": bundle["材料段落"], "满分": constraints["满分"],
        "字数上限": constraints["字数上限"],
    }
    refs = bundle["参考答案"][:ref_limit]
    payload = {**prepared, "参考答案": refs}

    def build():
        return validate_rubric(
            call_llm_json(RUBRIC_PROMPT, payload, api_key=api_key), prepared["满分"],
        )

    cache_input = [llm_cache_model(api_key), RUBRIC_PROMPT, payload]
    rubric = build() if cache_dir is None else load_or_build_rubric(
        cache_dir, cache_input, build, engine="three-types",
    )
    validate_rubric(rubric, prepared["满分"])
    return {**prepared, "题型": rubric["题型"], "细则": rubric}


def process_full_paper(prepared, answers, api_key=None, *, missing_content=(), complete=False):
    expected = set(prepared)
    unknown = (answers.keys() | set(missing_content)) - expected
    if unknown:
        raise ValueError(f"答卷存在题库未收录的题号：{sorted(unknown)}")
    if complete and (expected - answers.keys() or missing_content):
        raise ValueError("标记为完整的答卷存在缺题或残缺题")

    results = {}
    for number, entry in sorted(prepared.items()):
        rubric = entry["细则"]
        if number not in answers:
            results[number] = {"最终得分": None, "答卷形态": "未记录"}
        elif entry["题型"] in (1, 2, 3):
            answer = answers[number]
            if number in missing_content:
                diagnosis = grade_rubric_answer(entry, answer, api_key=api_key)
                results[number] = {"最终得分": None, "答卷形态": "回忆残卷", "记录内容诊断": diagnosis}
            elif not answer.strip() and not complete:
                results[number] = {"最终得分": None, "答卷形态": "空白"}
            else:
                results[number] = grade_rubric_answer(entry, answer, api_key=api_key)
        elif entry["题型"] == "作文":
            results[number] = grade_essay_answer(
                rubric, answers[number], api_key=api_key,
                recalled=True if number in missing_content else False if complete else None,
            )
        elif number in missing_content:
            diagnosis = grade_universal_answer(rubric, answers[number], api_key=api_key)
            results[number] = {"最终得分": None, "答卷形态": "回忆残卷", "记录内容诊断": diagnosis}
        elif not answers[number].strip():
            results[number] = {"最终得分": 0.0 if complete else None, "答卷形态": "空白"}
        else:
            results[number] = grade_universal_answer(rubric, answers[number], api_key=api_key)

    scores = [result["最终得分"] for result in results.values()]
    total = sum(scores) if complete and all(score is not None for score in scores) else None
    return {"总分": total, "分项详情": results}
