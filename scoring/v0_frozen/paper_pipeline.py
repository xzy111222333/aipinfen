import re
from pathlib import Path

from .engine import MODEL_DEFAULT, load_question_bundle
from .essay_engine import PREFAB_PROMPT, grade_essay_answer, prefab_essay_question
from .rubric_cache import DEFAULT_CACHE_DIR, load_or_build_rubric
from .universal_engine import (
    UNIVERSAL_ALIGN_PROMPT, UNIVERSAL_CARD_PROMPT, UNIVERSAL_SPEC_PROMPT,
    grade_universal_answer, prefab_universal_bundle, select_universal_references,
)


def is_essay_question(stem):
    return bool(re.search(r"(?:写|撰写|写作)\s*(?:一篇)?\s*(?:议论文|文章)|以议论文形式", stem))


def prefab_full_paper(paper_dir, api_key=None, *, ref_limit=15, cache_dir=DEFAULT_CACHE_DIR):
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


def prefab_paper_question(path, api_key=None, *, ref_limit=15, cache_dir=DEFAULT_CACHE_DIR):
    bundle = load_question_bundle(path, exclude_human_scored=True)
    essay = is_essay_question(bundle["题干与要求"])
    if essay:
        bundle = load_question_bundle(path, include_all_materials=True, exclude_human_scored=True)
        refs = []
        prompts = [PREFAB_PROMPT]

        def build():
            return prefab_essay_question(bundle, api_key=api_key)
    else:
        refs = select_universal_references(bundle["参考答案"], ref_limit)
        prompts = [UNIVERSAL_SPEC_PROMPT, UNIVERSAL_CARD_PROMPT]
        if refs:
            prompts.append(UNIVERSAL_ALIGN_PROMPT)

        def build():
            return prefab_universal_bundle(bundle, refs, api_key=api_key)

    kind = "作文" if essay else "通用"
    cache_input = [kind, MODEL_DEFAULT, str(Path(path).resolve()), bundle["题干与要求"],
                   bundle["材料段落"], refs, prompts]
    rubric = build() if cache_dir is None else load_or_build_rubric(cache_dir, cache_input, build)
    return {"题型": kind, "细则": rubric}


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
