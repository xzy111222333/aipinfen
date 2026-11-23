import json
from functools import partial
from pathlib import Path

from .class_a_pipeline import call_preparation_responses
from .reference_workflow import build, grade, save


def prepare_question(question, directory, *, holdout_sources=(), writing_count=5, sheet=None):
    """题库后台使用独立版本目录；全流程结果是成功完成标记。"""
    directory = Path(directory)
    model = "gpt-6.2-sol"
    if writing_count not in (3, 5):
        raise ValueError("写作参考份数须为3或5")
    result = build(question, None, None if holdout_sources else [], directory,
                   exclude_sources=list(holdout_sources), preparation_model=model,
                   preparation_call=partial(call_preparation_responses, model),
                   answer_sheet=sheet, writing_count=writing_count)
    save(directory / "全流程结果.json", result)
    return result


def grade_submission(rubric_file, answer, cache_directory):
    """由业务后端选择已发布的细则路径，返回逐项判定和细则指纹。"""
    if not isinstance(answer, str):
        raise ValueError("答卷必须为字符串")
    pack = json.loads(Path(rubric_file).read_text(encoding="utf-8-sig"))
    return grade(pack, answer, Path(cache_directory))
