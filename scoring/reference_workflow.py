"""多机构口碑加权共识聚类、弹性满分细则锁定、逻辑优先标杆生成与匿名阅卷。"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import re
from difflib import SequenceMatcher
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .class_a_pipeline import _call
from .engine import (
    RELAY_MODEL,
    count_exam_grids,
    is_verbatim_quote,
    load_question_bundle,
    parse_hard_constraints,
    truncate_by_exam_grids,
)

from .grid_sheet import ROW_WIDTH, measure_benchmark_layout


PIPELINE_VERSION = "consensus_v5"


def digest(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    os.replace(stream.name, path)


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def wilson_lower_bound(up: int, down: int, z: float = 1.96) -> float:
    total = up + down
    if total <= 0:
        return 0.0
    p = up / total
    denom = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (center - margin) / denom


def reputation_score(up: int, down: int) -> float:
    if up <= 0:
        return 0.0
    return round(wilson_lower_bound(up, down) * math.log(1.0 + up), 4)


def clean_body_and_votes(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    match = re.match(r"^\s*点赞\s*(\d+)\s*·\s*点踩\s*(\d+)\s*·\s*采集时间[^\n]*\n", text)
    up, down = 0, 0
    if match:
        up, down = map(int, match.groups())
        text = text[match.end() :].strip()
    text = re.sub(r"^\s*[（(]\s*(?:参考字数[:：]?\s*)?\d+\s*字\s*[）)]\s*", "", text).strip()
    text = re.sub(r"\s*[（(]\s*(?:参考字数[:：]?\s*)?\d+\s*字\s*[）)]\s*$", "", text).strip()
    total_votes = up + down
    approval = round(up / total_votes, 4) if total_votes else None
    return {
        "正文": text,
        "点赞": up,
        "点踩": down,
        "好评率": approval,
        "威尔逊下界": round(wilson_lower_bound(up, down), 4) if total_votes else 0.0,
        "口碑综合分": reputation_score(up, down),
        "格数": count_exam_grids(text),
    }


def similar(a: str, b: str, threshold: float = 0.85) -> bool:
    normalize = lambda s: re.sub(r"[^\w\u4e00-\u9fff]+", "", s).casefold()
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    return SequenceMatcher(None, na, nb, autojunk=False).ratio() >= threshold


def is_polluted_or_commentary(entry: dict[str, Any], cap: int | None) -> str | None:
    up, down = entry["点赞"], entry["点踩"]
    total = up + down
    if total >= 20 and (up / total) < 0.78:
        return f"差评率偏高({up}赞/{down}踩)"
    if total > 0 and up < down:
        return f"踩大于赞({up}赞/{down}踩)"
    if cap and entry["格数"] > round(cap * 1.16):
        return f"字数异常({entry['格数']}格>{cap}格，疑似多版本讲义合并)"
    if re.search(r"(?:短平快[:：]|升级归纳[:：]|解析[:：]|批注[:：])", entry["正文"]):
        return "含讲义解析标签"
    return None


def compute_layout_stats(training: list[dict[str, Any]], cap: int) -> dict[str, Any]:
    weights = [max(r["归一化权重"], 1e-4) for r in training]
    w_sum = sum(weights)
    avg_grids = round(sum(r["格数"] * w for r, w in zip(training, weights)) / w_sum, 1)

    point_counts = []
    two_part_ratios = []
    for r, w in zip(training, weights):
        body = r["正文"]
        pts = len(re.findall(r"(?:^|[\n。；])\s*(?:[①②③④⑤⑥⑦⑧⑨⑩]|\d+\s*[、.．]|[一二三四五六七八九十]+、)", body))
        if pts >= 2:
            point_counts.append((pts, w))
        split_m = re.search(
            r"(?:^|[\n。；])\s*(?:[一二]|\d+)?\s*[、.．]?\s*(?:建议|对策|举措|改进措施)\s*[：:1一①]?",
            body,
        )
        if split_m and split_m.start() >= 30:
            g1 = count_exam_grids(body[: split_m.start()])
            g2 = count_exam_grids(body[split_m.start() :])
            if g1 >= 25 and g2 >= 40:
                two_part_ratios.append((g1 / (g1 + g2), w))

    typical_points = (
        round(sum(p * w for p, w in point_counts) / sum(w for _, w in point_counts), 1)
        if point_counts
        else 5.0
    )
    stats: dict[str, Any] = {
        "字数上限": cap,
        "建表稿加权平均格数": avg_grids,
        "建表稿常见分点数": typical_points,
        "单点建议物理预算上限": max(24, min(65, round((cap * 0.88) / max(3.5, typical_points)))),
    }
    if len(two_part_ratios) >= max(2, len(training) // 2):
        r1 = sum(rt * w for rt, w in two_part_ratios) / sum(w for _, w in two_part_ratios)
        stats["前后两问篇幅占比"] = {
            "前问占比": round(r1, 3),
            "后问占比": round(1.0 - r1, 3),
            "说明": "建表高赞稿普遍压缩前问概括、把更多篇幅与权重留给后问具体措施",
        }
    return stats


def select_build_and_holdout(
    bundle: dict[str, Any],
    cap: int,
    build_ids: list[int] | None = None,
    holdout_ids: list[int] | None = None,
    exclude_sources: list[str] | None = None,
    build_count: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_refs = bundle["参考答案"]
    parsed: list[dict[str, Any]] = []
    for idx, ref in enumerate(raw_refs):
        info = clean_body_and_votes(ref["正文"])
        parsed.append({"原编号": idx, "来源": ref["来源"], **info})

    excl_kw = [k.strip().casefold() for k in (exclude_sources or []) if k and k.strip()]

    if holdout_ids is not None:
        require(all(0 <= i < len(parsed) for i in holdout_ids), "留出编号越界")
        require(len(set(holdout_ids)) == len(holdout_ids), "留出编号重复")
        holdout = [parsed[i] for i in holdout_ids]
    else:
        holdout = []
        seen_src = set()
        for item in sorted(parsed, key=lambda x: x["口碑综合分"], reverse=True):
            if excl_kw and any(k in item["来源"].casefold() for k in excl_kw):
                if is_polluted_or_commentary(item, cap) is None and item["来源"] not in seen_src:
                    if not any(similar(item["正文"], h["正文"]) for h in holdout):
                        holdout.append(item)
                        seen_src.add(item["来源"])
        require(len(holdout) > 0, "未匹配到符合条件的留出测分答案")

    held_indices = {h["原编号"] for h in holdout}
    held_authors = {h["来源"] for h in holdout}
    rejected: list[dict[str, Any]] = []

    def blocked_by_holdout(candidate: dict[str, Any]) -> bool:
        if candidate["原编号"] in held_indices or candidate["来源"] in held_authors:
            return True
        if excl_kw and any(k in candidate["来源"].casefold() for k in excl_kw):
            return True
        return any(similar(candidate["正文"], h["正文"]) for h in holdout)

    training: list[dict[str, Any]] = []
    if build_ids is not None:
        require(all(0 <= i < len(parsed) for i in build_ids), "建表编号越界")
        require(len(set(build_ids)) == len(build_ids), "建表编号重复")
        for i in build_ids:
            cand = parsed[i]
            require(not blocked_by_holdout(cand), f"编号{i}({cand['来源']})命中留出隔离规则")
            reason = is_polluted_or_commentary(cand, cap)
            if reason:
                rejected.append({"原编号": i, "来源": cand["来源"], "原因": reason})
                continue
            if not any(similar(cand["正文"], t["正文"]) for t in training):
                training.append(dict(cand))
    else:
        ranked = sorted(parsed, key=lambda x: (x["口碑综合分"], x["点赞"]), reverse=True)
        for cand in ranked:
            if blocked_by_holdout(cand):
                continue
            reason = is_polluted_or_commentary(cand, cap)
            if reason:
                rejected.append({"原编号": cand["原编号"], "来源": cand["来源"], "原因": reason})
                continue
            if any(similar(cand["正文"], t["正文"]) for t in training):
                continue
            training.append(dict(cand))
            if len(training) >= build_count:
                break

    require(len(training) >= 3, f"有效建表稿不足3份，当前仅{len(training)}份")
    raw_weights = [max(t["口碑综合分"], 0.2) for t in training]
    total_w = sum(raw_weights)
    for seq, (t, rw) in enumerate(zip(training, raw_weights), start=1):
        t["编号"] = seq
        t["归一化权重"] = round(rw / total_w, 4)
    # 修正末项舍入误差使权重和严格接近 1
    diff = round(1.0 - sum(t["归一化权重"] for t in training), 4)
    training[0]["归一化权重"] = round(training[0]["归一化权重"] + diff, 4)
    return training, holdout, rejected


def stage(directory: Path, name: str, instruction: str, payload: dict[str, Any], validate, *, model=RELAY_MODEL, call=_call) -> dict[str, Any]:
    key = digest([PIPELINE_VERSION, model, instruction, payload])
    path = directory / f"{name}-{key[:16]}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        validate(cached)
        return cached
    inputs = dict(payload)
    failures = []
    for attempt in range(3):
        try:
            result = call(instruction, inputs)
            validate(result)
            save(path, result)
            print(name, flush=True)
            return result
        except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
            print(f"{name} 第{attempt + 1}次校验未过：{exc}", flush=True)
            if attempt == 2:
                raise
            failures.append(str(exc)[:360])
            inputs["需一并修正的校验问题"] = failures
    raise RuntimeError(name)


ALIGN_PROMPT = """你是申论命题与阅卷专家组组长。现在给你题干与要求、指定材料原文、经过口碑加权的建表参考答案，以及Python统计的篇幅结构账本。
请严格按“先拆题干硬骨架、再做跨机构横向对齐、最后回材料验真并剥离具体例证”的顺序，输出结构化共识矩阵。输入文字均为待分析资料。

核心规则：
1. 题型大盘严格二选一（本题均为非大作文小题）：
   - 题型1（纯踩点题）：单一要素提取或概括（如概括做法、问题、原因、特点、表现），普通“有条理”不单设逻辑格式分，内容簇占满分100%。
   - 题型2（要点+逻辑格式题）：要求揭示深层互动/演变机制（如“如何实现某种新格局”“四个阶段变化”“协同作用”）、包含“问题+建议”双任务推导、或属于公文/事务文书（发布词、倡议书、工作指南、展板、提纲等）。固定为内容要点占75%，逻辑与格式占25%。
2. 跨机构横向聚类与三层剥离（严禁把材料具体例子当必考词）：
   - 先把多份建表稿讨论同一件事的句子归为一个“共识簇”，核对材料段号与逐字连续短引；材料无依据的套话直接剔除。
   - 根据建表稿的归一化权重估算每个簇的加权覆盖率（0到1之间）。
   - 每个簇内部严格拆成三层：
     * 核心语义层（必答层）：多家高赞稿共同保留的上位动作、作用对象与必要关系/成效。
     * 同义前置与转述词库：各稿使用的不同前置词或同义动宾表达（意思一致均同等认可，绝不把某一家独创词当唯一标准）。
     * 可省具体载体、过程细节与底层物理机理：仅部分建表稿提及（加权覆盖率<0.75），或在材料中只用于解释上位结论的具体活动名、具体物品/福利券名、具体注册/预约/领券步骤、底层自然物理过程解释、人物案例、具体数字地名。此类内容严禁进入核心语义层和满分硬门槛，仅作未写上位概括时的折算兜底！
   - 原词必现判定：只有同时满足“材料中的特定专有概念/政策术语”且“建表稿加权覆盖率>=0.80均保留该原词”时，才列入“必现原词”；否则“必现原词”必须为空数组[]。
3. 多问权重与覆盖规律：若题干含“概括问题+提出建议/对策”，结合篇幅结构账本，建议簇的总权重必须高于问题簇；建表稿中加权覆盖率>=0.60的独立建议维度（例如针对专门机制风险的合作开发或转包规范机制）应保留在建议簇内（可独立成簇或作为弹性并列抓手），不得整块遗漏。

只输出合法JSON，结构如下：
{
  "题型": 1,
  "题型理由": "",
  "必答任务块": [
    {"块名": "", "任务性质": "事实概括|机制演变|对策推导|公文载体", "建议分值占比": 0.5, "说明": ""}
  ],
  "文种格式硬要求": {
    "需要标题": false,
    "需要称谓": false,
    "需要落款": false,
    "禁止分条列点": false,
    "说明": "无公文外衣要求则填无"
  },
  "认可的逻辑组织框架": ["框架1简述", "框架2简述"],
  "共识簇": [
    {
      "簇号": "C1",
      "所属任务块": "必须与必答任务块的块名一致",
      "簇名称": "",
      "段号": [1],
      "材料短引": "必须从对应段原文逐字连续复制8~30字",
      "加权覆盖率": 0.95,
      "核心语义层": "不带具体例子的最短完整语义",
      "必现原词": [],
      "同义可认表达": ["表达1", "表达2"],
      "可省例证与细节": ["细节1", "细节2"]
    }
  ]
}"""


RUBRIC_PROMPT = """你是申论阅卷规则制定人。依据题干、材料、跨机构共识矩阵和篇幅预算账本，生成可独立复用的《评分标准与细则》。后续批改时不再读取任何机构答案。

必须严格遵守的量化铁律：
1. 分值大盘与考场弹性选答机制（最小颗粒度0.5分，所有分值必须是0.5的整数倍）：
   - 内容簇分值总和须在输入给定的“内容目标分”与“内容分值池上限”之间。两项输入相等时，内容簇分值加总须严格等于内容目标分。
   - 当内容分值池上限高于内容目标分时，只在低覆盖、可替代的维度上设计选答，最终按内容目标分封顶。逻辑格式总分严格等于输入给定的“逻辑格式目标分”，第1类的逻辑格式项为空数组。
2. 严禁并集死卡，强制执行“字数容量倒推弹性满分”：
   - 查看每个簇可分配到的物理方格数。每个内容簇的“最低满分示范句”格数绝不能超过输入限定的“单簇示范句最大格数”。写作时先以该上限的八成为目标，留出计格余量。
   - 满分条件必须真正弹性：考生只要答出该簇“核心语义层”的核心动作/特征+对象/效果（即2个核心要素组合，或同义上位表达）即给满分！严禁把“底层物理成因+具体载体+过程手段+最终成效”四五层要素同时绑死为满分前提！只有共识簇中经Python核验保留的“必现原词”非空时才可要求该专词，否则严禁自行新增必现原词！
   - 共识簇里被标为“可省例证与细节”的内容，严禁出现在“满分条件”里！必须写进“不扣分声明”，并仅在考生未写上位概括、只抄了具体例子时给一半分。
   - 对“问题+建议/对策”题：问题簇只要点准核心症结与受损关系即满分，不因省略具体欺诈手段/收费名目扣分；建议簇分值高（通常2.5~4分），必须体现题干要求的“具体明确、切实可行”——满分条件要求针对该问题写出至少两层具体可执行抓手（如“支持各类市场主体按自身优势公平参与规模化开发”而非单喊“鼓励公平竞争打破垄断”；“提高准入门槛审核资质+打击欺诈/红黑名单信用监管”而非单喊“加强监管打击违法”；“设专人对接协助屋顶产权协调+明确户用备案方式”而非单喊“专人对接明确备案”）；若考生对某建议簇只写8~12字表态口号而无具体操作抓手，只能按“部分给分”给该簇一半左右分数！
3. 逻辑格式项防僵化与禁止双重扣分（仅题型2）：
   - 格式分只检“文种格式硬要求”里为true的项目；若全为false，不得设任何格式分，25%全部用于评价逻辑条理！
   - 兼容主流拆条粒度：将同一链条的机制拆成3条或4条（例如把“开发模式垄断”与“转包/大企业不熟导致收益低”合并为1条或分列为2条，把“福利平台”与“线下站点”分列或合并为保障服务），均属合理归类，一律不认定为分类交叉！只有毫无同类意识、完全顺着自然段抄成流水账或把不相关要素硬塞进同一条才算交叉混乱。
   - 严禁内容与逻辑双重扣分：某一点具体内容或落脚词遗漏，只能在对应的“内容簇”扣一次分；只要宏观板块（如问题+建议两大块）清晰、分条有前置统领且整体方向对应，逻辑分必须给满，严禁因某一小点未点出结果词而在逻辑项再扣第二次分！严禁因“未单列总括段”“分点序号样式”“字数接近上限”扣分。

只输出合法JSON，结构如下：
{
  "题型": 1,
  "阅卷总则": "简明说明赋分结构、同义采信原则与弹性满分尺度",
  "内容簇": [
    {
      "簇号": "C1",
      "所属任务块": "",
      "名称": "",
      "分值": 2.0,
      "段号": [1],
      "必现原词": [],
      "最低满分示范句": "在格数预算内、不含可省例子的精炼满分句",
      "满分条件": "弹性达标规则：写出上位核心关系或等价表达即满分",
      "部分给分": "写明具体缺哪层核心关系或仅列举零散例子时给几分（0.5的倍数）",
      "不扣分声明": "明确列出省略哪些具体例子、渠道细节或换用哪些前置词绝不扣分"
    }
  ],
  "逻辑格式项": [
    {
      "名称": "",
      "类型": "逻辑|格式",
      "分值": 2.5,
      "满分条件": "包含多种等效组织方式的观察标准",
      "部分与零分条件": "分类严重交叉、顺材料流水账堆砌或漏必要公文要素时的给分（0.5的倍数）"
    }
  ]
}"""


OUTLINE_PROMPT = """你是申论标杆答案总编辑。在动笔写正文前，先依据题干、材料、跨机构共识矩阵和锁定的评分细则，设计一份全网最强的“逻辑骨架与篇幅分配蓝图”。

设计原则：
1. 逻辑立意优先：先看题干真正追问的骨架关系（如“如何以需求推动服务”“问题与建议对应”“阶段演变对比”“公文受众与号召”），参考高赞机构中最巧妙的统领维度，设计既紧扣题干又毫无分类交叉的分点框架。
2. 零死角覆盖：所有锁定的内容簇必须全部纳入对应分点，补齐单一机构容易遗漏的死角。
3. 先找材料已经给出的统领句。原句能够准确涵盖同组内容时，沿用原句或只删冗余修饰；下面只写能补足它的动作、对象、条件或结果。原句已完整作答、下文只是重复意思时，让这一句直接作答。材料确实没有统领句时，再补准确的短前置，或者用动作直接开头。统领可以跨簇，不必每簇自造一个标题。
4. 每个有采分内容的分点在蓝图记录至少一处真实的“材料短引”（原文连续短语，注明段号）。明确是“原文统领”“压缩统领”还是“直接落点”。建议可以由材料问题推出，短引用于证明它解决的问题，避免将建议写成材料已有的举措。
5. 先把字限分给必要的独立任务与关系。普通小题连续书写，靠序号或短标签区分，成稿目标为有效文字达到上限九成至满限。公文只保留文种或题面需要的换行，按输入的排版预算安排篇幅。结构占行时，优先保证内容和阅读功能。先按任务列出仍值得补足的信息，标明材料出处和预计新增格数。优先补题目要求但尚未写实的关系、行动对象、适用条件和执行抓手。

只输出合法JSON：
{
  "组织逻辑说明": "",
  "排版模式": "连续分点|连续两问分点|公文换行",
  "段落与分点蓝图": [
    {"序号": "总起或①", "前置统领词": "", "前置用法": "原文统领|压缩统领|直接落点", "材料短引": [{"段号": 1, "短引": ""}], "纳入簇号": ["C1"], "拟用规范词": [], "剔除的具体例子": [], "目标格数": 40}
  ]
}"""


BENCHMARK_PROMPT = """你是申论标杆答案主笔。请严格按照已审定的“逻辑骨架与篇幅分配蓝图”、题干材料和锁定评分细则，写出两版候选标杆答案（两稿分别尝试有依据的不同组织或压缩方式，均优先沿用材料的有效统领和规范词），并从中选出更优的一版。

硬性格式与语言规范（违者直接打回）：
1. 分点符号与排版：
   - 凡适合列点的小题（非必须连贯成篇的短评/展板/信函），分点符号一律统一使用“①②③④⑤…”，严禁使用“一是、二是”或“1、2、”或“一、二、”！
   - 普通小题连续书写，每个分点紧接上一点。两问直接写成“问题：①……②……。建议：①……②……。”，靠短标签区分，不另起一行。题面明确要求换行时照其要求。
   - 若有单句总起，总起句末尾一律用句号“。”收束，直接接“①…”，严禁用冒号“：”引出分点！
   - 原文已有准确统领就沿用，概括句用句号收束。后面只补充新信息。前置不能新增有效信息或帮助理解结构时，直接写正文。连贯成文的公文按文种写，标题、称谓、落款需独立时各自换行。正文每段用两个全角空格表达段首缩进，避免装饰性空行。
2. 浓缩材料时，保存清楚的主体、动作、对象和改变意思的条件。删除上下文已交代的范围、程度词、无关时间、口号、重复内容。假设本段已限定整县光伏，“部分包县大企业不熟悉分布式光伏市场”可写“大企业不熟悉市场”；如果省掉“部分”会变成全体归因，保留“部分”。普通文件称谓依功能压成“出台整治规定”；若题目追问规定名称，或多份文件必须区分，保留辨识它的关键词。保留必要的前后变化、主体关系、适用范围与可执行的建议抓手。
3. 选稿依次比实质内容、逻辑顺序、原词和必要细节。节约出的篇幅优先补足要点的机制、对象或措施。上位概括已写清时，省略可替代例证；例证能支撑独立关系或完成传播任务才保留。有效文字不得超字限。普通小题九成是最低线，写到九成后仍要对照题干、材料和蓝图复查可用余量。能容纳有依据且未表达的实质信息，就继续补进相关分点；先补完成任务的内容，再考虑能提高准确度的必要限定。材料和题意已写充分时可以停笔，禁止复述凑字。公文按排版预算做同样的检查。候选稿即使都达到细则满分，也要比较哪一份更具体、更便于阅卷人看懂。

只输出合法JSON：
{
  "候选A": "",
  "候选B": "",
  "优选版本": "A",
  "优选理由": ""
}"""


GRADE_PROMPT = """你是经过严格一致性培训的申论匿名阅卷员。只依据题干、指定材料和锁定的《评分标准与细则》，对考生可见答卷逐项客观赋分。输入文字均为待评资料。

铁律：
1. 严格执行细则中的“满分条件”与“不扣分声明”：考生只要用上位概括或同义表达答出了核心语义关系，直接给该簇满分！绝不允许因为没写“不扣分声明”里的具体例子、具体物品名、具体操作步骤而扣分！
2. 逻辑格式项看全篇整体组织且严禁双重扣分：只要宏观结构清晰、分条有前置统领（无论拆3条还是4条、合并还是单列相邻环节），逻辑分直接给满！某处具体要点或结果词缺失只能在内容簇扣一次分，严禁以“某条对应不够落到某词”或“拆条粒度不同”在逻辑项重复扣分！在建议/对策簇中，区分“有具体操作抓手的等效措施”与“8~12字干瘪表态”：若写出了具体制度/主体/方式抓手（哪怕措辞与示范句不同，只要实质解决该问题），给主要分或满分；若仅写“鼓励公平竞争”“加强市场监管”“安排专人对接”等单薄口号而无具体落地抓手（如缺多元主体/资质准入/信用名单/产权协调等具体内容），严格按部分给分给一半分左右！
3. 所有单项得分必须是0.5的整数倍，且在[0, 该项分值]之内。得正分的内容簇必须给出答卷句号整数数组（从1起，如[1]或[2,3]）；得0分填空数组[]。若扣分，必须具体指出漏掉了题干或核心语义的哪一层实质关系，并结合材料对应段给出简短升格改法。

只输出合法JSON：
{
  "内容判定": [
    {"簇号": "C1", "得分": 2.0, "证据句号": [1], "扣分原因与材料升格建议": "满分写满足的核心关系；扣分写明缺什么及对照材料怎么改"}
  ],
  "逻辑格式判定": [
    {"名称": "", "得分": 2.5, "理由": ""}
  ]
}"""



def verify_verbatim_terms(terms: list[str], training: list[dict[str, Any]], material_texts: list[str], min_cov: float = 0.85) -> list[str]:
    """Python 硬核验：只有真正在材料出现且在建表高赞稿中逐字覆盖率>=0.85的专有词才允许保留为必现原词。"""
    full_mat = "\n".join(material_texts)
    verified = []
    for raw_t in (terms or []):
        t = str(raw_t).strip()
        if len(t) < 2 or len(t) > 8 or t not in full_mat:
            continue
        cov = sum(row["归一化权重"] for row in training if t in row["正文"])
        if cov >= min_cov and t not in verified:
            verified.append(t)
    return verified

def split_exam_sentences(visible_text: str) -> list[str]:
    return [m.group().strip() for m in re.finditer(r"[^。；！？\n]+[。；！？]?", visible_text) if m.group().strip()]


def half_step(val: Any) -> bool:
    return type(val) in (int, float) and math.isclose(val * 2.0, round(val * 2.0), abs_tol=1e-6)


def target_score_split(total: float, kind: int) -> tuple[float, float]:
    if kind == 1:
        return total, 0.0
    logic = round((total * 0.25) * 2.0) / 2.0
    return round(total - logic, 2), round(logic, 2)


def validate_benchmark_style(text: str, forbidden_colon_list: bool, is_prose_only: bool) -> None:
    require(bool(text and text.strip()), "标杆答案为空")
    if not is_prose_only:
        require("①" in text and "②" in text, "非连贯成文题必须使用①②分点")
        require(not re.search(r"(?:一是|二是|三是|四是)", text), "禁止使用“一是、二是”分点")
        require(not re.search(r"(?:^|[\n。；])\s*[一二三四五六七八九十]+、", text), "禁止使用“一、二、”分点")
        require(not re.search(r"(?:^|[\n。；])\s*\d+\s*[、.．]", text), "禁止使用“1、2、”分点")
        if forbidden_colon_list:
            require(not re.search(r"[:：]\s*①", text), "总括句后禁止用冒号引出①，只有“问题：”“建议：”“对策：”除外")
        # 检查①后面的前置短句没有用冒号收束
        for seg in re.findall(r"[①②③④⑤⑥⑦⑧⑨⑩]([^。；\n]{1,18}[:：])", text):
            raise ValueError(f"分点前置总括词后请用句号收束，禁用冒号：{seg}")


def grade(pack: dict[str, Any], text: str, directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # JSON 保存会把段号转为字符串，恢复整数键使旧细则的指纹可复核。
    pack = {**pack, "材料原文": {int(n): body for n, body in pack["材料原文"].items()}}
    raw_text = (text or "").strip()
    if not raw_text:
        return {"总分": 0.0, "内容得分": 0.0, "逻辑格式得分": 0.0, "格数": 0, "超出字限": "", "内容判定": [], "逻辑格式判定": [], "细则指纹": digest(pack)}

    visible, overflow = truncate_by_exam_grids(raw_text, pack["字限"])
    sentences = split_exam_sentences(visible)
    c_items = pack["细则"]["内容簇"]
    l_items = pack["细则"]["逻辑格式项"]

    def check_grade(result: dict[str, Any]) -> None:
        c_rows = result["内容判定"]
        l_rows = result["逻辑格式判定"]
        require([r["簇号"] for r in c_rows] == [c["簇号"] for c in c_items], "内容判定簇号顺序不匹配")
        require([r["名称"] for r in l_rows] == [l["名称"] for l in l_items], "逻辑格式项顺序不匹配")
        for row, item in zip(c_rows, c_items):
            sc, raw_nums, reason = row["得分"], row["证据句号"], row["扣分原因与材料升格建议"]
            if type(raw_nums) is int:
                nums = [raw_nums] if raw_nums > 0 else []
                row["证据句号"] = nums
            else:
                require(isinstance(raw_nums, list) and all(type(x) is int for x in raw_nums), f"{item['簇号']}证据句号须为整数数组:{raw_nums}")
                nums = [x for x in raw_nums if x > 0]
                row["证据句号"] = nums
            require(half_step(sc) and 0 <= sc <= item["分值"], f"{item['簇号']}分值非法:{sc}")
            require(all(1 <= x <= len(sentences) for x in nums), f"{item['簇号']}证据句号越界:{nums}")
            require(bool(nums) if sc > 0 else True, f"{item['簇号']}正分必须提供证据句号")
            require(isinstance(reason, str) and bool(reason.strip()), f"{item['簇号']}缺少理由")
        for row, item in zip(l_rows, l_items):
            sc, reason = row["得分"], row["理由"]
            require(half_step(sc) and 0 <= sc <= item["分值"], f"逻辑项{item['名称']}分值非法:{sc}")
            require(isinstance(reason, str) and bool(reason.strip()), f"逻辑项{item['名称']}缺少理由")

    payload = {
        "题干": pack["题干"],
        "材料原文": pack["材料原文"],
        "评分细则": pack["细则"],
        "可见答卷按句编号": dict(enumerate(sentences, 1)),
        "可见答卷全文": visible,
    }
    res = stage(directory, "匿名评分", GRADE_PROMPT, payload, check_grade)
    c_out = []
    for row, item in zip(res["内容判定"], c_items):
        nums = row["证据句号"]
        c_out.append({
            "簇号": item["簇号"],
            "名称": item["名称"],
            "满分": item["分值"],
            "得分": row["得分"],
            "证据原句": " ".join(sentences[n - 1] for n in nums) if nums else "",
            "反馈": row["扣分原因与材料升格建议"],
        })
    l_out = []
    for row, item in zip(res["逻辑格式判定"], l_items):
        l_out.append({
            "名称": item["名称"],
            "类型": item["类型"],
            "满分": item["分值"],
            "得分": row["得分"],
            "理由": row["理由"],
        })
    raw_c_sum = round(sum(r["得分"] for r in c_out), 2)
    c_cap = pack.get("内容封顶分", raw_c_sum)
    c_sum = min(c_cap, raw_c_sum)
    l_sum = round(sum(r["得分"] for r in l_out), 2)
    return {
        "总分": round(c_sum + l_sum, 2),
        "内容得分": c_sum,
        "逻辑格式得分": l_sum,
        "格数": count_exam_grids(raw_text),
        "超出字限": overflow,
        "内容判定": c_out,
        "逻辑格式判定": l_out,
        "细则指纹": digest(pack),
    }


def build_benchmark(pack, align, layout_stats, references, directory, *, model=RELAY_MODEL, call=_call, sheet=None):
    if sheet is not None:
        require(len(sheet) == 2 and all(type(n) is int and n > 0 for n in sheet), "答题区规格须为每行格数、区域行数两个正整数")
    material = {int(n): body for n, body in pack["材料原文"].items()}
    base = {key: pack[key] for key in ("题干", "材料原文", "字限", "满分")}
    cap, rubric = pack["字限"], pack["细则"]
    prepare = partial(stage, directory, model=model, call=call)
    fmt_req = align["文种格式硬要求"]
    is_prose_only = bool(fmt_req.get("禁止分条列点"))
    needs_own_lines = ("需要标题", "需要称谓", "需要落款", "禁止分条列点")
    is_document = any(fmt_req.get(key) for key in needs_own_lines) or any(
        block.get("任务性质") == "公文载体" for block in align["必答任务块"]
    )
    min_grids = None if is_document else (9 * cap + 9) // 10
    row_width = sheet[0] if sheet else ROW_WIDTH
    budget = {"有效字数上限": cap, "普通题优选稿有效字数下限": min_grids}
    if is_document:
        budget.update({
            "每行格数": row_width,
            "答题区域行数": sheet[1] if sheet else None,
            "排版说明": "标题、称谓、落款单列与正文分段均消耗行尾空格。未提供实际行数时，排版只估算。",
        })
    has_qa_two_blocks = (
        len(align["必答任务块"]) == 2
        and any("问题" in b["块名"] for b in align["必答任务块"])
        and any(any(k in b["块名"] for k in ("建议", "对策")) for b in align["必答任务块"])
    )

    def check_outline(result: dict[str, Any]) -> None:
        plan = result["段落与分点蓝图"]
        require(isinstance(plan, list) and bool(plan), "蓝图为空")
        require(all(isinstance(row.get("纳入簇号"), list) for row in plan), "蓝图纳入簇号须为数组")
        covered = {cid for row in plan for cid in row["纳入簇号"]}
        expected = {c["簇号"] for c in rubric["内容簇"]}
        require(covered == expected, f"蓝图与细则内容簇不一致：{sorted(expected ^ covered)}")
        for row in plan:
            if not row.get("纳入簇号"):
                continue
            require(row["前置用法"] in ("原文统领", "压缩统领", "直接落点"), "蓝图须明确前置用法")
            quotes = row["材料短引"]
            require(bool(quotes), "有采分内容的蓝图分点缺材料短引")
            for quote in quotes:
                source = material.get(int(quote["段号"]), "")
                require(is_verbatim_quote(quote["短引"], source), f"蓝图短引无法定位到材料第{quote['段号']}段：{quote['短引']}")
            if row["前置用法"] == "原文统领":
                require(any(is_verbatim_quote(row["前置统领词"], quote["短引"]) for quote in quotes), "原文统领须能在本分点的材料短引中定位")

    outline = prepare(
        "标杆逻辑蓝图",
        OUTLINE_PROMPT,
        {**base, "共识矩阵": align, "评分细则": rubric, "篇幅结构账本": layout_stats, "成稿预算": budget},
        check_outline,
    )

    def check_benchmark(result: dict[str, Any]) -> None:
        require(result["优选版本"] in ("A", "B") and bool(result["优选理由"].strip()), "优选版本须为A或B")
        for key in ("候选A", "候选B"):
            txt = result[key].strip("\r\n")
            if not is_document:
                txt = "".join(line.strip() for line in txt.splitlines())
            result[key] = txt
            grids = count_exam_grids(txt)
            require(0 < grids <= cap, f"{key}有效字数{grids}，须在1至{cap}格内")
            if min_grids is not None and key == "候选" + result["优选版本"]:
                require(grids >= min_grids, f"{key}有效字数{grids}，普通小题优选稿至少{min_grids}格；请补独立信息")
            if is_document and sheet:
                occupied = measure_benchmark_layout(txt, row_width)
                require(occupied <= row_width * sheet[1], f"{key}排版占{occupied}格，答题区域仅{row_width * sheet[1]}格")
            cleaned_for_colon = re.sub(r"(问题|建议|对策)\s*[:：]\s*①", r"\1①", txt)
            validate_benchmark_style(cleaned_for_colon, forbidden_colon_list=True, is_prose_only=is_prose_only)
            if has_qa_two_blocks:
                require(re.match(r"问题：①.+?[。！？；]\s*(?:建议|对策)：①", txt, re.DOTALL), f"{key}须用“问题：①…”与“建议：①…”区分两问")

    bench_res = prepare(
        "标杆双稿选优",
        BENCHMARK_PROMPT,
        {**base, "逻辑蓝图": outline, "评分细则": rubric, "高赞表达参考": references, "成稿预算": budget},
        check_benchmark,
    )
    answer = bench_res["候选" + bench_res["优选版本"]].strip("\r\n")
    result = {"正文": answer, "格数": count_exam_grids(answer), "逻辑蓝图": outline, "双稿选优": bench_res}
    if is_document:
        occupied = measure_benchmark_layout(answer, row_width)
        result["排版估算"] = {
            "每行格数": row_width,
            "采用实测答题区": sheet is not None,
            "答题区域行数": sheet[1] if sheet else None,
            "排版占格": occupied,
            "排版占行": math.ceil(occupied / row_width),
        }
    save(directory / "标杆答案.json", result)
    return result


def build(
    question: str | Path,
    build_ids: list[int] | None,
    holdout_ids: list[int] | None,
    directory: str | Path,
    exclude_sources: list[str] | None = None,
    build_count: int = 5,
    weak_answer: str | None = None,
    *,
    preparation_model: str = RELAY_MODEL,
    preparation_call=_call,
    answer_sheet: tuple[int, int] | None = None,
) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bundle = load_question_bundle(question)
    spec = parse_hard_constraints(bundle["题干与要求"])
    cap, total = spec["字数上限"], spec["满分"]
    require(bool(cap and cap > 0), "小题缺少字数上限")

    training, holdout, rejected = select_build_and_holdout(
        bundle,
        cap=cap,
        build_ids=build_ids,
        holdout_ids=holdout_ids,
        exclude_sources=exclude_sources,
        build_count=build_count,
    )
    prepare = partial(stage, directory, model=preparation_model, call=preparation_call)
    layout_stats = compute_layout_stats(training, cap)
    material = {p["全局段号"]: p["原文"] for p in bundle["材料段落"]}
    base = {"题干": bundle["题干与要求"], "材料原文": material, "字限": cap, "满分": total}

    build_for_llm = [
        {
            "编号": t["编号"],
            "归一化权重": t["归一化权重"],
            "点赞": t["点赞"],
            "点踩": t["点踩"],
            "格数": t["格数"],
            "正文": t["正文"],
        }
        for t in training
    ]

    def check_align(result: dict[str, Any]) -> None:
        require(result["题型"] in (1, 2) and bool(result["题型理由"].strip()), "题型必须为1或2")
        blocks = result["必答任务块"]
        require(isinstance(blocks, list) and len(blocks) >= 1, "必答任务块为空")
        block_names = {b["块名"] for b in blocks if b.get("块名")}
        require(len(block_names) == len(blocks), "必答任务块重名或为空")
        clusters = result["共识簇"]
        require(isinstance(clusters, list) and 3 <= len(clusters) <= 10, "共识簇数量应在3~10之间")
        seen_ids = set()
        for c in clusters:
            cid = c["簇号"]
            require(cid not in seen_ids and c["所属任务块"] in block_names, f"簇{cid}编号重复或任务块未知")
            seen_ids.add(cid)
            segs = c["段号"]
            require(isinstance(segs, list) and segs and set(segs) <= set(material.keys()), f"簇{cid}段号越界")
            quote = c["材料短引"].strip()
            require(len(quote) >= 6 and any(quote in material[s] for s in segs), f"簇{cid}材料短引必须逐字连续取自所标段号")
            require(bool(c["核心语义层"].strip()) and isinstance(c["同义可认表达"], list) and len(c["同义可认表达"]) >= 1, f"簇{cid}核心层或同义表达缺失")
            require(isinstance(c["可省例证与细节"], list), f"簇{cid}可省例证字段格式错误")
            if c["加权覆盖率"] < 0.8:
                require(c["必现原词"] == [], f"簇{cid}覆盖率低于0.8不得设置必现原词")

    align = prepare(
        "跨机构共识对齐",
        ALIGN_PROMPT,
        {**base, "篇幅结构账本": layout_stats, "加权建表稿": build_for_llm},
        check_align,
    )
    mat_list = list(material.values())
    for c in align["共识簇"]:
        c["必现原词"] = verify_verbatim_terms(c.get("必现原词", []), training, mat_list, min_cov=0.85)

    kind = align["题型"]
    c_target, l_target = target_score_split(total, kind)
    cluster_count = len(align["共识簇"])
    max_pool = c_target + (1.0 if cluster_count >= 6 and cap <= 250 else 0.0)
    max_demo_grids = max(28, min(56, round((cap * 0.95) / max(cluster_count, 4))))

    def check_rubric(result: dict[str, Any]) -> None:
        require(result["题型"] == kind and bool(result["阅卷总则"].strip()), "细则题型与共识矩阵不一致")
        c_items = result["内容簇"]
        l_items = result["逻辑格式项"]
        require([c["簇号"] for c in c_items] == [c["簇号"] for c in align["共识簇"]], "内容簇须与共识簇一一对应")
        for c, src in zip(c_items, align["共识簇"]):
            require(half_step(c["分值"]) and c["分值"] >= 0.5, f"{c['簇号']}分值须为>=0.5的半分倍数")
            require(set(c["段号"]) <= set(material.keys()) and bool(c["段号"]), f"{c['簇号']}段号无效")
            demo_grids = count_exam_grids(c["最低满分示范句"])
            require(4 <= demo_grids <= max_demo_grids, f"{c['簇号']}最低满分示范句{demo_grids}格超出单簇上限{max_demo_grids}格，请改写为凝练上位表达")
            require(all(bool(str(c[k]).strip()) for k in ("满分条件", "部分给分", "不扣分声明")), f"{c['簇号']}判据缺失")
            require(set(c.get("必现原词", [])) <= set(src["必现原词"]), f"{c['簇号']}擅自新增未经Python共识验证的必现原词:{c.get('必现原词')}")
            for opt in src["可省例证与细节"]:
                opt_clean = opt.strip()
                if len(opt_clean) >= 2:
                    require(opt_clean not in c["满分条件"] and opt_clean not in c["最低满分示范句"], f"{c['簇号']}禁止将可省例证“{opt_clean}”写进满分条件或最低满分示范句")
        c_sum = round(sum(c["分值"] for c in c_items), 2)
        l_sum = round(sum(l["分值"] for l in l_items), 2)
        require(c_target - 1e-6 <= c_sum <= max_pool + 1e-6, f"内容簇总分{c_sum}须在[{c_target}, {max_pool}]范围内")
        require(math.isclose(l_sum, l_target, abs_tol=1e-6), f"逻辑格式总分{l_sum}不等于目标分{l_target}")
        fmt_req = align["文种格式硬要求"]
        has_fmt = any(bool(fmt_req.get(k)) for k in ("需要标题", "需要称谓", "需要落款", "禁止分条列点"))
        for l in l_items:
            require(half_step(l["分值"]) and l["分值"] >= 0.5, f"逻辑格式项{l['名称']}分值须为半分倍数")
            require(l["类型"] in ("逻辑", "格式"), "逻辑格式项类型只能是逻辑或格式")
            if not has_fmt:
                require(l["类型"] == "逻辑", "本题无公文外衣硬要求，禁止设置格式分")
        if "前后两问篇幅占比" in layout_stats:
            # 校验建议/对策块分值高于问题块
            prob_score = sum(c["分值"] for c in c_items if "问题" in c["所属任务块"])
            sug_score = sum(c["分值"] for c in c_items if any(k in c["所属任务块"] for k in ("建议", "对策", "措施", "解决")))
            if prob_score > 0 and sug_score > 0:
                require(sug_score > prob_score, f"问题+建议题中建议分({sug_score})必须高于问题分({prob_score})")

    rubric = prepare(
        "锁定评分细则",
        RUBRIC_PROMPT,
        {
            **base,
            "共识矩阵": align,
            "篇幅结构账本": layout_stats,
            "内容目标分": c_target,
            "内容分值池上限": max_pool,
            "逻辑格式目标分": l_target,
            "单簇示范句最大格数": max_demo_grids,
        },
        check_rubric,
    )

    pack = {
        **base,
        "题型": kind,
        "内容封顶分": c_target,
        "逻辑格式目标分": l_target,
        "细则": rubric,
        "模型": preparation_model,
        **({"阅卷模型": RELAY_MODEL} if preparation_model != RELAY_MODEL else {}),
        "版本": PIPELINE_VERSION,
    }
    save(directory / "冻结细则.json", pack)

    benchmark = build_benchmark(
        pack, align, layout_stats, [t["正文"] for t in training[:3]], directory,
        model=preparation_model, call=preparation_call, sheet=answer_sheet,
    )
    answer = benchmark["正文"]

    eval_targets: dict[str, str] = {"完美标杆": answer}
    for h in holdout:
        label = h["来源"]
        if label in eval_targets:
            label = f"{label}_{h['原编号']}"
        eval_targets[label] = h["正文"]
    if weak_answer:
        eval_targets["低分弱稿对照"] = weak_answer

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        graded_list = list(pool.map(lambda txt: grade(pack, txt, directory), eval_targets.values()))
    scores = dict(zip(eval_targets.keys(), graded_list))
    save(directory / "盲测汇总.json", scores)

    return {
        "题目路径": str(question),
        "题干与要求": bundle["题干与要求"],
        "建表组": [
            {"原编号": t["原编号"], "来源": t["来源"], "点赞": t["点赞"], "点踩": t["点踩"], "口碑综合分": t["口碑综合分"], "归一化权重": t["归一化权重"], "格数": t["格数"]}
            for t in training
        ],
        "剔除污染稿": rejected,
        "留出测分组": [
            {"原编号": h["原编号"], "来源": h["来源"], "点赞": h["点赞"], "点踩": h["点踩"], "口碑综合分": h["口碑综合分"], "格数": h["格数"]}
            for h in holdout
        ],
        "篇幅结构账本": layout_stats,
        "跨机构共识矩阵": align,
        "冻结细则": pack,
        "标杆逻辑蓝图": benchmark["逻辑蓝图"],
        "标杆答案": answer,
        "标杆格数": benchmark["格数"],
        **({"标杆排版估算": benchmark["排版估算"]} if "排版估算" in benchmark else {}),
        "盲评结果": scores,
    }
