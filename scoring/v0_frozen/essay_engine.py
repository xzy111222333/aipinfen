import math
import re

from .engine import (
    call_llm_json, clean_reference_body, index_answer_evidence, is_verbatim_quote,
    parse_hard_constraints, resolve_answer_evidence,
)
from .grid_sheet import detect_answer_completeness, measure_grid_sheet


PREFAB_PROMPT = """从题干与完整材料提取大作文评分细则。严禁评分或给出范文。
提取 2 至 5 项直接来自题意的立意要求，包括必须解释的概念和题意明确要求的逻辑关系。
每项只表达一个可检验的要求。尊重自选角度，不能把材料中的某个案例或唯一表达规定为必答。
依据必须是题干或材料中的连续原句，不拼接。
返回 JSON：{"立意要求":[{"要求":"...","来源引文":"..."}]}。"""

EVIDENCE_PROMPT = """逐项核查大作文答卷。严禁输出分数或档位。
立意要求返回「完整」「部分」「未见」。论证与全文检核返回「充分」「完整」「部分」「未见」。指出实际做到了什么，尚缺什么。
非「未见」的判断必须选择能支撑判断的答卷原句编号；允许选择多个编号，不需要重新抄写原句。
仅出现主题词、只把两个概念并列、重复结论，不算充分解释关系。
对论证四项从各主要论证段整体判断。多数段基本做到为完整，少数做到或只有很薄的展开为部分。
论证达到「充分」需要有能从原句核查的更高质量，不能因为要素齐全而填写充分：
- 分论点从不同角度推进中心判断，说明各自作用，而非同一判断换词重复。
- 据以立论的案例或说理提供足够的条件、行动或结果，使读者能核查它为何相关，没有用概括性标签代替依据。
- 给出论据通向判断的关键中间环节；在题干需要时，能说清条件如何改变结果或为何需要取舍。不因为缺少某一种论证形式降级。
- 各层理由共同说明中心结论的不同部分；逐段重复题干或只在段尾喊口号并不充分。
对 A3 另列支撑论证的原句编号。每条论证链指出考生写出的论据，以及本人明确写出的推理。单独写“可见某某很重要”、复述结果或由你代为补出的推理都不计。不同论证链应靠不同的原句推进不同判断。确实没有写出的关系返回空数组。
对每条链，只有原文明确交代导致结果的中间机制、会改变结论的条件、取舍原因或反证时，才填写"展开编号"。填写时选择包含该机制或条件的原句。多步因果写在一句中时，可以与推理编号重合。只有事例和结论之间简单连线时，展开编号填空数组。程序根据这些证据重新定 A3 等级。
全文结构按观点与理由的真实关系判断，不规定分论点数量。语言关注能否准确、连贯理解意思，不根据排比、金句或卷面猜测质量。联系实际可以用材料中的真实情境展开，只要说明在何种具体条件下成立、做法或结果如何；点名案例但没有展开不算完整。
引用或举例可以用于论证，关键是有没有解释它怎样支持观点。
如果输入是回忆残卷，只判断已记录的文字，未记载的内容不能当作考场漏写。
返回 JSON：{"核查":[{"编号":"...","状态":"完整","证据编号":["1","2"],"说明":"..."}],
"论证链":[{"论据编号":["3"],"推理编号":["4"],"展开编号":["4"],"说明":"这些原句如何支撑该判断"}]}。"""

ARGUMENT_CHECKS = {
    "A1": "分论点具体、清楚，能够支撑中心立意",
    "A2": "依据具体，可来自材料、实际案例或有内容的说理论证",
    "A3": "写出论据与观点之间的因果或推理，不止于复述案例",
    "A4": "在论证中回扣题意，说明分论点与中心立意的联系",
}
WHOLE_ESSAY_CHECKS = {
    "B1": "结构完整，分论点相互区分，并有清楚的推进关系",
    "B2": "表达准确、连贯，关键条件和结论能被读者明确理解",
}
LEVELS = {"未见": 0.0, "部分": 0.5, "完整": 1.0}
ARGUMENT_LEVELS = {"未见": 0.0, "部分": 0.5, "完整": 0.8, "充分": 1.0}
BANDS = (
    ("一类", 0.80, 1.0),
    ("二类", 0.63, 0.80),
    ("三类", 0.43, 0.63),
    ("四类", 0.0, 0.43),
)
COPY_WINDOW = 18

HIGH_BAND_REVIEW_PROMPT = """复核申论大作文是否有充分展开的推理。看候选引文，只判断文中明确写出的内容，不替考生补分析。
判真需要同时能从这些原句读出：论据与题干判断有关；作者写出了从具体做法或条件通往结论的必要中间环节，或说明为何要在具体选项间取舍。这个环节必须提供新解释，不能只是给事实贴标签或在段尾复述观点。
例如“保护带来发展、分类施策改善民生”只复述判断。写清“现有床位已满，一律关停会使老人无法转移；按风险分流可以先控制危险并保留可改造床位”才解释了为什么。
“A地采取了数字技术，所以取得成效”仍要找文内说明技术如何改变了流程或决策。允许在一句内写全，也允许有实际内容的道理论证。不要以术语、引用名人或形式作为条件。
对每条候选单独输出结果，论证清楚即通过，无需假设考场分数。返回 JSON：{"核实":[{"编号":"1","通过":false,"说明":"原因"}]}。"""


def prefab_essay_question(bundle, api_key=None):
    question = bundle["题干与要求"]
    materials = [paragraph["原文"] for paragraph in bundle["材料段落"]]
    constraints = parse_hard_constraints(question)
    result = call_llm_json(
        PREFAB_PROMPT, {"题干": question, "材料": materials}, api_key=api_key
    )
    slots = result["立意要求"]
    if not 2 <= len(slots) <= 5:
        raise ValueError("作文立意要求须包含 2 至 5 项")
    sources = [question, *materials]
    for slot in slots:
        if not slot["要求"].strip() or not any(
            is_verbatim_quote(slot["来源引文"], source) for source in sources
        ):
            raise ValueError(f"作文立意要求缺少可核对的来源引文：{slot}")
    if len({slot["要求"].strip() for slot in slots}) != len(slots):
        raise ValueError("作文立意要求重复")
    return {"题干与要求": question, "硬约束": constraints, "立意要求": slots, "材料原文": materials}


def _copy_ratio(text, materials):
    def normalize(value):
        return re.sub(r"[\W_]", "", value).lower()

    windows = set()
    for material in materials:
        normalized = normalize(material)
        windows.update(normalized[i:i + COPY_WINDOW] for i in range(len(normalized) - COPY_WINDOW + 1))
    answer = normalize(text)
    covered = set()
    for index in range(len(answer) - COPY_WINDOW + 1):
        if answer[index:index + COPY_WINDOW] in windows:
            covered.update(range(index, index + COPY_WINDOW))
    return len(covered) / len(answer) if answer else 0.0


def _check_answer(rubric, text, incomplete, api_key):
    slots = {f"T{i}": slot["要求"] for i, slot in enumerate(rubric["立意要求"], start=1)}
    whole = dict(WHOLE_ESSAY_CHECKS)
    if re.search(r"(?:联系|结合)\s*(?:(?:社会|生活|工作)\s*)?(?:实际|现实)", rubric["题干与要求"]):
        whole["B3"] = "联系具体实际情境，说明判断在什么条件下有作用或如何实施"
    checks = {**slots, **ARGUMENT_CHECKS, **whole}
    spans = index_answer_evidence(text)
    payload = {
        "题干": rubric["题干与要求"],
        "立意要求": slots,
        "论证检核": ARGUMENT_CHECKS,
        "全文检核": whole,
        "回忆残卷": incomplete,
        "答卷原句": {key: span[2] for key, span in spans.items()},
    }
    prompt = EVIDENCE_PROMPT
    for _ in range(3):
        response = call_llm_json(prompt, payload, api_key=api_key)
        rows = response["核查"]
        for row in rows:
            row["答卷引文"] = resolve_answer_evidence(row, text, spans)
        indexed = {row["编号"]: row for row in rows}
        valid = (
            len(rows) == len(checks)
            and indexed.keys() == checks.keys()
            and all(
                row["状态"] in (LEVELS if row["编号"] in slots else ARGUMENT_LEVELS) and (
                    row["状态"] == "未见" or row["答卷引文"]
                )
                for row in rows
            )
        )
        chains = _resolve_reasoning_chains(response.get("论证链"), text, spans)
        if valid and chains is not None:
            return [
                {**indexed[key], "检核项": requirement}
                for key, requirement in checks.items()
            ], chains
        errors = []
        if len(rows) != len(checks) or indexed.keys() != checks.keys():
            errors.append(f"核查编号应逐项且仅出现一次：{list(checks)}；实际收到 {list(indexed)}")
        for key in indexed.keys() & checks.keys():
            row = indexed[key]
            allowed = LEVELS if key in slots else ARGUMENT_LEVELS
            if row["状态"] not in allowed or (row["状态"] != "未见" and not row["答卷引文"]):
                errors.append(f"{key} 状态或证据无效，允许状态为 {list(allowed)}")
        if chains is None:
            errors.append("论证链须为数组，每条须有可用的论据编号、推理编号和说明；无展开可填空数组")
        prompt = EVIDENCE_PROMPT + "\n上一轮校验失败。" + "；".join(errors)
    raise ValueError("作文寻证连续三次未通过编号及引文校验：" + "；".join(errors))


def _resolve_reasoning_chains(rows, text, spans):
    if not isinstance(rows, list):
        return None
    chains = []
    used_reasoning = set()
    used_reasoning_text = set()
    for row in rows:
        if not isinstance(row, dict):
            return None
        support_ids, reason_ids, depth_ids = row.get("论据编号"), row.get("推理编号"), row.get("展开编号", [])
        if any(not isinstance(ids, list) for ids in (support_ids, reason_ids, depth_ids)):
            return None
        support = resolve_answer_evidence({"证据编号": support_ids}, text, spans)
        reasoning = resolve_answer_evidence({"证据编号": reason_ids}, text, spans)
        depth = resolve_answer_evidence({"证据编号": depth_ids}, text, spans)
        explanation = row.get("说明")
        if not support or not reasoning or not isinstance(explanation, str) or not explanation.strip():
            return None
        if depth_ids and not depth:
            return None
        ids = {str(value) for value in reason_ids + depth_ids}
        normalized = re.sub(r"[\W_]", "", reasoning)
        if ids & used_reasoning or normalized in used_reasoning_text:
            continue
        used_reasoning.update(ids)
        used_reasoning_text.add(normalized)
        chains.append({
            "论据编号": support_ids, "推理编号": reason_ids, "展开编号": depth_ids,
            "论据原句": support, "推理原句": reasoning, "展开原句": depth, "说明": explanation.strip(),
        })
    return chains


def _rate_reasoning(ledger, chains, materials):
    own_chains = [chain for chain in chains if _copy_ratio(chain["推理原句"], materials) < 0.8]
    depth_count = sum(
        bool(chain.get("展开原句")) and _copy_ratio(chain["展开原句"], materials) < 0.8
        for chain in own_chains
    )
    if depth_count >= 2:
        level = "充分"
    elif depth_count or len(own_chains) >= 2:
        level = "完整"
    else:
        level = "部分" if own_chains else "未见"
    reasoning = next(row for row in ledger if row["编号"] == "A3")
    reasoning.update({
        "状态": level,
        "证据编号": own_chains[0].get("推理编号", []) if own_chains else [],
        "答卷引文": own_chains[0]["推理原句"] if own_chains else "",
        "说明": f"可核查的自写推理 {len(own_chains)} 条，其中写出机制、条件或取舍的有 {depth_count} 条。",
    })
    return own_chains, depth_count


def _review_high_band(rubric, chains, materials, api_key):
    candidates = {
        str(number): chain
        for number, chain in enumerate(chains, start=1)
        if chain["展开原句"] and _copy_ratio(chain["展开原句"], materials) < 0.8
    }
    payload = {
        "题干与要求": rubric["题干与要求"],
        "候选论证": {
            key: {field: chain[field] for field in ("论据原句", "推理原句", "展开原句")}
            for key, chain in candidates.items()
        },
    }
    prompt = HIGH_BAND_REVIEW_PROMPT
    for _ in range(3):
        result = call_llm_json(prompt, payload, api_key=api_key).get("核实")
        if isinstance(result, list) and len(result) == len(candidates) and all(
            isinstance(row, dict) and isinstance(row.get("编号"), (str, int))
            and isinstance(row.get("通过"), bool) and isinstance(row.get("说明"), str) and row["说明"].strip()
            for row in result
        ):
            indexed = {str(row["编号"]): row for row in result}
            if indexed.keys() == candidates.keys():
                checked = [
                    chain if indexed.get(str(i), {}).get("通过") else {**chain, "展开原句": "", "展开编号": []}
                    for i, chain in enumerate(chains, start=1)
                ]
                audit = [{**row, "展开原句": candidates[key]["展开原句"]} for key, row in indexed.items()]
                return checked, audit
        prompt = HIGH_BAND_REVIEW_PROMPT + f"\n按编号逐项返回核实：{list(candidates)}。"
    raise ValueError("一类文复核返回的编号、判定或说明无效")


def _essay_ratings(states):
    thesis_values = [LEVELS[level] for key, level in states.items() if key.startswith("T")]
    whole_values = [ARGUMENT_LEVELS[level] for key, level in states.items() if key.startswith("B")]
    return {
        "立意覆盖率": sum(thesis_values) / len(thesis_values),
        "论证达成率": sum(ARGUMENT_LEVELS[states[key]] for key in ARGUMENT_CHECKS) / len(ARGUMENT_CHECKS),
        "全文达成率": sum(whole_values) / len(whole_values),
    }


def _score_essay(states, ratings, chains, depth_count, maximum, length_ratio, copied):
    thesis = ratings["立意覆盖率"]
    argument = ratings["论证达成率"]
    whole = ratings["全文达成率"]
    barriers = []
    if thesis < 1:
        barriers.append("核心题意尚未完整回应")
    if depth_count < 2:
        barriers.append("缺少至少两条可核查、充分展开的独立推理")
    if any(ARGUMENT_LEVELS[states[key]] < 0.8 for key in ("A1", "A2", "A4")):
        barriers.append("分论点、论据或回扣中心尚未达到完整")
    if any(ARGUMENT_LEVELS[level] < 0.8 for key, level in states.items() if key.startswith("B")):
        barriers.append("结构、语言或题干要求的联系实际尚未达到完整")

    ceiling = 0 if not barriers else 1
    if thesis < 0.67 or states["A3"] == "未见":
        ceiling = 2 if thesis > 0 else 3
    if copied >= 0.50 and not chains:
        ceiling = 3
        barriers.append("材料原文重合达到一半，未核实独立推理")
    copy_deduction = max(0.0, copied - 0.30) * 0.30
    quality = max(0.0, (0.45 * thesis + 0.40 * argument + 0.15 * whole) * length_ratio - copy_deduction)
    natural_band = next(index for index, (_, lower, _) in enumerate(BANDS) if quality >= lower)
    if natural_band > 0:
        barriers.append("综合质量经篇幅和材料原文重合修正后未达到一类分线")
    band, lower, upper = BANDS[max(ceiling, natural_band)]
    # 在实际档内按质量连续分布，避免只要未进高档就挤到下一档的上限。
    progress = (quality - lower) / (1.0 - lower)
    ordered_score = (lower + (upper - lower) * progress) * maximum
    score = round(ordered_score * 2) / 2
    if upper < 1.0:
        score = min(score, (math.ceil(upper * maximum * 2) - 1) / 2)
    return {
        "立意档位": band, "最终得分": score, "量化质量": round(quality, 4), "一类文未满足条件": barriers,
        "高比例复用扣减": round(copy_deduction * maximum, 2),
    }


def grade_essay_answer(rubric, answer_text, materials_text="", api_key=None, *, recalled=None):
    cleaned = clean_reference_body(answer_text)
    constraints = rubric["硬约束"]
    maximum = constraints["满分"]
    grid = measure_grid_sheet(cleaned, constraints["字数上限"])
    visible = grid["可见正文"]
    completeness = detect_answer_completeness(cleaned)
    incomplete = recalled if recalled is not None else completeness["是否残卷或提纲"]
    report = {
        "满分": maximum,
        "答卷形态": completeness["形态"] if recalled is None else ("回忆提纲或残卷" if recalled else "完整作答"),
        "方格纸": grid,
    }
    if not visible:
        return {**report, "立意档位": "未见作答", "最终得分": None if incomplete else 0.0, "证据账本": []}

    ledger, chains = _check_answer(rubric, visible, incomplete, api_key)
    materials = [materials_text] if materials_text else rubric["材料原文"]
    chains, depth_count = _rate_reasoning(ledger, chains, materials)
    states = {row["编号"]: row["状态"] for row in ledger}
    ratings = _essay_ratings(states)
    copied = _copy_ratio(visible, materials)
    report.update({**ratings, "证据账本": ledger, "论证链": chains, "展开论证条数": depth_count, "材料复用率": round(copied, 4)})
    if incomplete:
        return {**report, "立意档位": None, "最终得分": None, "诊断": "按已记录内容列出立意与论证证据，全文表现待核。"}

    minimum = constraints["最少字数"]
    length_ratio = min(1.0, grid["纯字数"] / minimum) if minimum else 1.0
    scored = _score_essay(states, ratings, chains, depth_count, maximum, length_ratio, copied)
    if scored["立意档位"] == "一类":
        checked, audit = _review_high_band(rubric, chains, materials, api_key)
        chains, depth_count = _rate_reasoning(ledger, checked, materials)
        states = {row["编号"]: row["状态"] for row in ledger}
        ratings = _essay_ratings(states)
        scored = _score_essay(states, ratings, chains, depth_count, maximum, length_ratio, copied)
        report["一类文复核"] = audit
    return {**report, **ratings, **scored, "论证链": chains, "展开论证条数": depth_count, "篇幅达成率": length_ratio}
