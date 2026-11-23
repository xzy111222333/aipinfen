"""A 类热门题双轨制标工具。

流程固定为四步：材料账本、细则、标杆答案、留出答案盲测。
题目内容只从运行时输入进入提示词，提示词本身不包含任何真题专属例子。
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from .engine import MDKJ_KEY, RELAY_MODEL, _read_api_setting, count_exam_grids, load_question_bundle, parse_hard_constraints


LEDGER_PROMPT = r"""你是申论命题组的材料分析员。只依据题干、要求和材料建立“材料量化账本”。

必须遵守：
1. 不代写答案，不凭常识补充材料没有的事实，不把材料中的例子、数字、地点、人物自动当作独立得分点。
2. 先判断题干要交付哪些任务，再给材料段落贴角色标签。角色只能从背景、现状、问题、原因、做法、机制、结果、意义、要求、案例佐证、格式信息中选择，可多选但要说明主角色。
3. 把同一上位事实下的细节归入“内容簇”。内容簇是考生在有限字数内应表达的最小完整意思，通常包含主体、动作/状态、对象和结果/关系中的必要部分。
4. 对每个细节说明它是“核心关系”“重要限定”“可替代例证”还是“背景噪声”。核心关系缺失应影响主要得分，可替代例证被概括覆盖时不得重复扣分。
5. 标出材料原词、可接受的同义转述、不可自行扩展的边界。原词只在改变政策对象、主体、动作、数量级或因果关系时才严格要求。
6. 记录跨段合并关系、重复关系、题干任务关联度和字数颗粒度建议。字数限制优先，能用上位概括覆盖多个例证时必须明确写出。

只输出合法 JSON，字段必须为：
{
  "题型判断": {"主任务槽位": "", "子任务": [], "格式要求": [], "字数上限": 0, "满分": 0},
  "段落账本": [{"段落坐标":"", "主角色":"", "事实摘要":"", "材料原词":[], "核心关系":[], "重要限定":[], "可替代例证":[], "可概括为":"", "可推出边界":[], "题干关联":"直接|辅助|无关"}],
  "内容簇": [{"簇号":"C1", "上位概括":"", "覆盖段落":[], "必须保留":[], "可省略":[], "允许合并":[], "同义表达边界": {"可接受":[], "不可接受":[]}, "缺失后果":"", "建议颗粒度":"上位|中位|下位"}],
  "字数与组织规则": [""],
  "材料不能支持的说法": [""]
}
"""

RUBRIC_PROMPT = r"""你是申论阅卷组制订评分标准的负责人。依据题干、要求、材料量化账本和多份机构答案，生成一份可复用的评分标准与细则。

机构答案只能作为交叉证据，用来发现常见归并、表达和争议边界，不能直接照抄，也不能把最高赞答案当标准。材料是事实边界，题干决定任务。不得针对某份答案的专属句式加分。

制标要求：
1. 先分配任务模块和内容簇，再分配分值。分值允许 0.25、0.5、0.75 等小数。总分必须严格等于题目满分。
2. 上位概括覆盖下位例证时，按覆盖程度给主要分，不要求逐个复现例子、数字、地点或人物。相同内容换一种分点组织不得少分。
3. 每个内容簇要写完整、部分、笼统、错误、未写五档边界。部分分必须对应缺失的主体、动作、对象、关系或重要限定，不能因为没写可替代例子而大扣分。
4. 材料原词严格程度分为：规范对象/政策名称/专门机制，关系和动作，普通概括词。只有前两类改变实质时才明显扣分，普通概括允许等义表达。
5. 对公文、短评、建议等有格式或逻辑要求的题，格式与内容不得混为一谈。格式只评价题干明确要求和文种必要结构。字数是硬约束，不能要求答案同时写入相互替代的细节。
6. 评分理由必须能指出题干任务、材料依据和答卷缺失的层次。不得使用“感觉不够”“不够深入”这类不可执行理由。

只输出合法 JSON：
{
 "版本原则": [""],
 "题型与任务": {"槽位":"", "任务模块":[{"模块":"", "满分":0, "说明":""}]},
 "采分簇": [{"簇号":"C1", "名称":"", "满分":0, "材料依据":[], "核心语义":"", "完整条件":[], "部分条件":[], "笼统条件":[], "错误条件":[], "未写条件":[], "原词严格级别":"高|中|低", "可接受表达":[], "禁止过度要求":[]}],
 "组织与格式": {"满分":0, "项目": [{"名称":"", "分值":0, "完整":"", "部分":"", "不扣分情形":[], "扣分边界":[]}]},
 "字数规则": {"上限":0, "超限处理":"", "颗粒度规则":[], "公文空行或标题处理":""},
 "重复与封顶": [""],
 "评分流程": [""],
 "标杆答案验收": [""]
}
"""

BENCHMARK_PROMPT = r"""你是申论高分答案编辑。只根据题干、要求、材料量化账本和锁定评分细则，生成一份可以在真实答题纸上书写的标杆答案。

要求：
1. 先满足题干交付物和文种，再覆盖评分细则中的上位内容簇。优先使用材料原词或对其做最小压缩，不编造材料外政策，不用空泛口号代替具体行动。
2. 体现申论答案的规范表达，使用必要的前置词、分点和逻辑关系。每个分点写“上位概括 + 核心动作/状态 + 必要对象或结果”，可替代例证只选最省字的一处。
3. 严格不超过题干字数上限。程序会再次计格数。不要为了填满字数堆例子。
4. 只输出 JSON：{"标杆答案":"", "纯字数":0, "覆盖簇":[], "舍弃说明":[], "字数检查":""}
"""

GRADE_PROMPT = r"""你是经过统一尺度培训的申论阅卷员。使用给定评分细则评阅答卷，只按材料证据和题干任务判分。

判分原则：上位概括覆盖下位事实时给主要分，同义转述不因未复现原句扣分；核心主体、动作、对象、关系缺失才按细则扣对应部分。不能要求答卷写入细则列出的所有可替代例子。机构答案不是评分标准。

每个簇只能给该簇的一个状态和小数分，证据引文必须是答卷原文连续片段。先逐簇判断，再按细则计算总分。只输出 JSON：
{"总分":0, "簇判定":[{"簇号":"C1", "状态":"完整|部分|笼统|错误|未写", "得分":0, "答卷引文":"", "材料依据":[], "理由":""}], "组织格式": {"得分":0, "理由":[]}, "字数": {"纯字数":0, "是否超限":false, "扣分":0}, "总评":""}
"""


def _compact_refs(refs: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    rows = refs[: limit or len(refs)]
    return [{"来源": r["来源"], "正文": r["正文"], "格数": r.get("格数", count_exam_grids(r["正文"]))} for r in rows]


def _call(prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    import time
    import urllib.error
    from http.client import IncompleteRead

    key = _read_api_setting(MDKJ_KEY)
    if not key:
        raise ValueError('缺少中转密钥配置')
    body = dict(model=RELAY_MODEL, system=prompt, max_tokens=20000, thinking={'type':'disabled'}, temperature=0, messages=[{'role':'user','content':json.dumps(payload, ensure_ascii=False)}])
    raw_bytes = json.dumps(body, ensure_ascii=False).encode('utf-8')
    last_err: Exception | None = None
    for attempt in range(4):
        request = urllib.request.Request('https://mdkj.lol/v1/messages', raw_bytes, {'Content-Type':'application/json','anthropic-version':'2023-06-01','User-Agent':'Mozilla/5.0'})
        request.add_unredirected_header('x-api-key', key)
        try:
            with urllib.request.urlopen(request, timeout=210) as response:
                data = json.load(response)
            if data['model'] != RELAY_MODEL or data['stop_reason'] != 'end_turn':
                raise ValueError(f"模型 {data['model']}，结束状态 {data['stop_reason']}")
            text = ''.join(b['text'] for b in data['content'] if b['type'] == 'text').strip()
            cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.I).strip()
            cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
            return json.loads(cleaned)
        except urllib.error.HTTPError as err:
            if err.code in {429, 500, 502, 503, 504, 524} and attempt < 3:
                last_err = err
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, IncompleteRead) as err:
            if attempt < 3:
                last_err = err
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError(str(last_err))


def call_preparation_responses(model: str, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    from .engine import _read_relay_response

    key = _read_api_setting("SHENLUN_MDKJ_RESPONSES_API_KEY")
    if not key:
        raise RuntimeError("缺少 SHENLUN_MDKJ_RESPONSES_API_KEY")
    body = {
        "model": model,
        "instructions": prompt,
        "input": json.dumps(payload, ensure_ascii=False),
        "max_output_tokens": 16000,
        "store": False,
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "x-openai-actor-authorization": "local-image-extension",
    }
    request = urllib.request.Request(
        "https://mdkj.lol/v1/responses", json.dumps(body, ensure_ascii=False).encode("utf-8"), headers
    )
    request.add_unredirected_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(request, timeout=180) as response:
        data = _read_relay_response(response)
    if data.get("model") != model or data.get("status") != "completed":
        raise ValueError(f"模型 {data.get('model')}，结束状态 {data.get('status')}")
    text = "".join(
        part["text"] for item in data["output"] if item["type"] == "message"
        for part in item["content"] if part["type"] == "output_text"
    ).strip()
    return json.loads(text.removeprefix("```json").removesuffix("```").strip())


def build_class_a(question_path, build_ids, holdout_ids, stage_dir):
    bundle = load_question_bundle(question_path)
    constraints = parse_hard_constraints(bundle["题干与要求"])
    refs = bundle["参考答案"]
    for ref in refs:
        text = re.sub(r'^点赞\s*\d+\s*·\s*点踩\s*\d+\s*·\s*采集时间[^\n]*\n', '', ref['正文']).strip()
        ref['正文'] = re.sub(r'[（(]\s*\d+\s*字\s*[）)]$', '', text).strip()
        ref['格数'] = count_exam_grids(ref['正文'])
    ids = build_ids + holdout_ids
    if len(set(ids)) != len(ids) or any(i < 0 or i >= len(refs) for i in ids):
        raise ValueError('建表与留出编号重复或越界')
    build_refs = [refs[i] for i in build_ids]
    holdout = [refs[i] for i in holdout_ids]
    if {r['正文'] for r in build_refs} & {r['正文'] for r in holdout}:
        raise ValueError('建表与留出正文重复')
    base = {"题干与要求": bundle["题干与要求"], "材料段落": bundle["材料段落"], "硬约束": constraints}
    stage_dir.mkdir(parents=True, exist_ok=True)
    def stage(name, prompt, inputs):
        result = _call(prompt + '\n逐项简写，避免重复，数组只留必要内容，每项一句。', inputs)
        (stage_dir / (name + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(name, flush=True)
        return result
    ledger = stage('材料账本', '逐段量化材料，JSON仅输出rows数组，每段一行：[全局段号,材料角色,材料摘要,必要关系,可省例证,上位概括]。每行不超过75字，按题干任务分析，区分材料事实和可推出的对策。', base)
    ledger['原文索引'] = base['材料段落']
    examples = [r['正文'] for r in build_refs]
    rubric = stage('评分细则', RUBRIC_PROMPT.split('只输出合法 JSON：')[0] + '''只输出简短JSON，结构为 {"采分簇":[{"簇号":"C1","名称":"","满分":1,"材料段号":[],"满分边界":"","部分给分":"写明分值和条件","可省略":""}],"组织与格式":{"满分":0,"判据":""},"零分与重复计分规则":""}。只列必要项，每项不超过80字。''', {**base, "材料量化账本": ledger, "建表机构答案": examples})
    total = sum(c['满分'] for c in rubric['采分簇']) + rubric['组织与格式']['满分']
    if total != constraints['满分']:
        raise ValueError(f'评分预算之和 {total} 与满分不符')
    payload = {**base, "材料量化账本": ledger, "评分标准与细则": rubric, '建表表达参考': examples}
    for attempt in range(3):
        benchmark = stage('标杆稿', BENCHMARK_PROMPT, payload)
        size = count_exam_grids(benchmark['标杆答案'])
        if 0 < size <= constraints['字数上限']:
            break
        payload['改稿要求'] = f'上稿实际{size}格，请压缩并保留各项核心关系。'
        payload['上稿'] = benchmark['标杆答案']
    else:
        raise ValueError('标杆稿超过字限')
    benchmark["程序复核字数"] = count_exam_grids(benchmark.get("标杆答案", ""))
    blind = []
    for ref in holdout:
        result = stage('留出' + str(len(blind) + 1), GRADE_PROMPT, {"题干与要求": bundle["题干与要求"], "材料段落": bundle["材料段落"], "评分标准与细则": rubric, "考生答卷": ref["正文"]})
        result["来源"] = ref["来源"]
        result["参考格数"] = ref.get("格数", count_exam_grids(ref["正文"]))
        blind.append(result)
    return {
        "题目路径": str(question_path),
        "题干与要求": bundle["题干与要求"],
        "硬约束": constraints,
        "建表答案来源": [r["来源"] for r in build_refs],
        "留出答案来源": [r["来源"] for r in holdout],
        "材料量化账本": ledger,
        "评分标准与细则": rubric,
        "完美标杆答案": benchmark,
        "留出盲测": blind,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 A 类题材料账本、评分细则、标杆答案并留出盲测")
    parser.add_argument("question", nargs="+", help="题目 Markdown 路径")
    parser.add_argument("--output", required=True, help="输出 JSON 文件或目录")
    parser.add_argument('--build', type=int, nargs='*', default=None)
    parser.add_argument('--holdout', type=int, nargs='*', default=None)
    parser.add_argument('--exclude-sources', nargs='*', default=None, help='按来源名称自动踢入留出测分组并物理隔离同名来源')
    parser.add_argument('--build-count', type=int, default=5, help='自动选取的高口碑建表稿数量')
    parser.add_argument('--weighted', action='store_true', help='口碑加权共识聚类、弹性细则锁定、逻辑优先标杆与留出盲测')
    parser.add_argument("--preparation-model", help="weighted 制作端使用的 Responses 模型，阅卷仍用 DeepSeek")
    parser.add_argument("--answer-sheet", type=int, nargs=2, metavar=("每行格数", "区域行数"), help="采用实测的答题区规格检查公文标杆排版")
    args = parser.parse_args()
    if args.answer_sheet and (not args.weighted or min(args.answer_sheet) < 1 or len(args.question) != 1):
        parser.error("--answer-sheet 需要 --weighted、单道题目和两个正整数")
    if args.preparation_model and not args.weighted:
        parser.error("--preparation-model 需要 --weighted")
    output = Path(args.output)
    if len(args.question) > 1 or output.suffix.lower() != ".json":
        output.mkdir(parents=True, exist_ok=True)
    for q in args.question:
        target = output if len(args.question) == 1 and output.suffix.lower() == ".json" else output / (Path(q).stem + "_A类双轨.json")
        if args.weighted:
            from .reference_workflow import build
            preparation = {}
            if args.preparation_model:
                from functools import partial

                preparation = {
                    "preparation_model": args.preparation_model,
                    "preparation_call": partial(call_preparation_responses, args.preparation_model),
                }
            result = build(
                q,
                args.build,
                args.holdout,
                target.with_suffix(''),
                exclude_sources=args.exclude_sources,
                build_count=args.build_count,
                answer_sheet=tuple(args.answer_sheet) if args.answer_sheet else None,
                **preparation,
            )
        else:
            if not args.build or not args.holdout:
                raise ValueError('非 weighted 模式必须指定 --build 与 --holdout')
            result = build_class_a(q, args.build, args.holdout, target.with_suffix(''))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()
