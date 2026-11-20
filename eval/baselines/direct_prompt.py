import math

from scoring.engine import call_llm_json
from scoring.universal_engine import select_universal_references


VERSION = "direct-v1"
PROMPT = """你批改申论。综合题目、给定材料、参考答案和考生答卷，直接给出不超过题目满分的分数。参考的不同写法同样可以成立。按题目任务评估事实准确、内容覆盖、逻辑结构和表达质量，必要时评价受众与文种适配。作文核查立意、论证与表达。遵守题干明确的字数、身份和格式要求，引用答卷原句说明具体理由。直接输出 JSON：{\"score\":0,\"band\":1,\"comments\":[{\"quote\":\"答卷中的逐字引文，没有原句则留空\",\"reason\":\"具体得分或扣分依据\"}],\"overflow_excluded\":false}。band 用 1 到 4 表示从高到低，overflow_excluded 表示实际核查时排除了超可见字数部分。"""


def grade_direct(question, bundle, body, api_key=None):
    refs = select_universal_references(bundle["参考答案"], 8)
    response = call_llm_json(PROMPT, {
        "满分": question["maximum"], "题干与要求": bundle["题干与要求"], "材料": bundle["材料段落"],
        "机构答案": [ref["正文"] for ref in refs], "答卷": body,
    }, api_key=api_key, temperature=0.0, max_tokens=3000)
    score = response.get("score")
    if type(score) not in {int, float} or not math.isfinite(score) or not 0 <= score <= question["maximum"]:
        raise ValueError("直接提示词基线返回越界得分")
    if type(response.get("band")) is not int or not 1 <= response["band"] <= 4:
        raise ValueError("直接提示词基线返回无效档位")
    if not isinstance(response.get("comments"), list) or not response["comments"]:
        raise ValueError("直接提示词基线未返回评分理由")
    if any(not isinstance(row, dict) or not isinstance(row.get("quote"), str)
           or not isinstance(row.get("reason"), str) or not row["reason"].strip() for row in response["comments"]):
        raise ValueError("直接提示词基线返回了格式无效的证据或理由")
    return response
