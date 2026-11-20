import concurrent.futures
import json
import math
import re
from pathlib import Path

from .engine import (
    call_llm_json,
    clean_reference_body,
    count_exam_grids,
    index_answer_evidence,
    is_verbatim_quote,
    load_question_bundle,
    parse_hard_constraints,
    resolve_answer_evidence,
    truncate_by_exam_grids,
)


UNIVERSAL_SPEC_PROMPT = """你是申论题干字面指令解析器。严禁凭主观经验预设文种规则，只能提取题干字面明示的客观信息。
输入【题干与要求】，请提取以下结构化字段：
1. "字面任务块"：只列题干分别要求的独立成果，例如 ["推进缓慢的原因", "下一步工作建议"]。为同一个观点表态并提供论证，是一次完整的分析任务，放在同一块；材料案例不能因为支撑观点而重复当作表态任务的独立采分点。写短评、概括做法等单一任务填 ["正文主体"]。
2. "是否含问题反推对策"：布尔值。当题干要求针对材料中的问题/原因提出建议、对策、改进措施时为 true，否则为 false。
3. "是否要求明确观点表态"：布尔值。仅当题干或要求字面出现 "谈谈你的看法"、"观点明确"、"是否应该/该不该"、"对此加以评析" 时为 true。
4. "字面指定标题"：若题干明确指定了固定标题文字（如 以“暗夜星空下的别样繁华”为标题），填该标题纯文本（不含书名号引号）；未指定固定标题填 null。
5. "题干是否明示写标题或文种名"：布尔值。若题干要求写短评、发布词、公开信、工作指南、编者按、发言提纲、建议书、宣传材料等独立文种或指定了标题时为 true，纯问答概括题为 false。
6. "题干字面要求免格式"：布尔值。仅当题干字面出现 "不拘泥于格式"、"不考虑格式"、"只写正文"、"列出要点即可" 时为 true。
7. "题干明示目标受众"：若题干明确写了面向特定对象发言/宣传/呼吁（如 "向参会旅游企业和游客推介"、"呼吁公众积极报名"），填该对象名称；无特定沟通对象（如报刊短评、内部梳理问题、分析看法、编写内部指南内容）填 null。
8. "题干字面要求呼吁号召"：布尔值。仅当题干或要求字面出现 "呼吁"、"号召"、"倡议"、"动员" 时为 true。
9. "可核查要求"：从题目和「要求」段中提取超出材料事实覆盖的独立交付要求，最多四项。每项给连续逐字的"题干引文"、"类别"、"核查标准"、"核心任务"。
   - 类别只能是 "受众沟通"、"组织归类"、"分析关系"、"表达方式"。相近或重复要求合并。「内容全面」「介绍全面」「分析全面」「紧扣资料」及事实准确性用内容采分核查，不建独立要求。混合要求只核查独立的那部分，例如「原因阐述全面、准确、有条理」只提取「有条理」。字数、指定标题或明确表态也不重复提取。
   - 将要求写成这道题的可观察结果。号召力要检查公众能否读懂为什么参加、谁能参加和怎么行动；归类准确要检查事实放进哪一类；要求有针对性要检查建议能否解决它声称对应的问题；工作指南要检查每项工作及对应动作是否方便工作人员定位，不按故事顺序或分条数代替。各条件不得把未写的边缘信息当作结构或号召力失分。
   - 仅当整项缺失会使题目指定的成果无法使用或未完成时，"核心任务" 填 true。检查具体要求，不添加专业术语、固定格式或同一标准用语。

返回 JSON：
{
  "字面任务块": ["..."],
  "是否含问题反推对策": false,
  "是否要求明确观点表态": false,
  "字面指定标题": null,
  "题干是否明示写标题或文种名": false,
  "题干字面要求免格式": false,
  "题干明示目标受众": null,
  "题干字面要求呼吁号召": false,
  "可核查要求": [{"题干引文":"...","类别":"受众沟通","核查标准":"...","核心任务":true}]
}"""

REQUIREMENT_TYPES = {"受众沟通", "组织归类", "分析关系", "表达方式"}
REQUIREMENT_LEVELS = {"达成": 1.0, "部分达成": 0.5, "未达成": 0.0}
CORE_REQUIREMENT_CEILING = 0.80


def _extract_universal_spec(question, api_key):
    prompt = UNIVERSAL_SPEC_PROMPT
    for _ in range(3):
        spec = call_llm_json(prompt, {"题干与要求": question}, api_key=api_key, temperature=0.0)
        requirements = spec.get("可核查要求")
        if (
            isinstance(requirements, list) and len(requirements) <= 4
            and all(
                isinstance(item, dict)
                and item.get("类别") in REQUIREMENT_TYPES
                and isinstance(item.get("题干引文"), str) and item["题干引文"].strip()
                and item["题干引文"] in question
                and isinstance(item.get("核查标准"), str) and item["核查标准"].strip()
                and isinstance(item.get("核心任务"), bool)
                for item in requirements
            )
            and len({item["题干引文"] for item in requirements}) == len(requirements)
        ):
            return spec
        prompt = UNIVERSAL_SPEC_PROMPT + "\n重新输出可核查要求。题干引文须原样连续复制，不加引号、顿号或省略号。"
    raise ValueError("无法预制可核查要求：引文、类别或核查标准未通过校验")


UNIVERSAL_CARD_PROMPT = """你是申论材料原子信息与任务映射器。只做机械拆句、指代还原与字面因果配对，严禁主观编造材料外观点或定分值。
输入包含【题干字面解析】和【给定材料段落】。请完成：
1. 按题干逐段筛出能够回答任务的事实，再拆解为最小客观事实卡片（卡号 M01, M02...）：
   - 保留题目问的对象和关系。例如问原因时，单独陈述做法或成果不能代替原因；若题目问对象如何协同，不能只罗列各自做了什么。
   - 不为凑数量而收录无关细节。相同事实只建一卡，能单独遗漏的并列动作拆为不同卡；起限定作用的对象、条件和结果留在同一卡。
   - 若任务要求结合案例表达看法、解释或评价，优先提取每个案例怎样支撑观点的关键做法、机制和后果。将关键关系写入核心语义；材料里用来举证的数字、年份、操作细节通常留在原文锚点。题目特别要求量、时间或操作细节，或者省去该信息会改变结论时，才将其设为必要条件。不要把人物感叹代替案例结论。
   - 若若干案例围绕同一个判断形成对照，以其论证作用建卡。先问每个主要案例提供了什么独立理由，再各建一张主干关系卡。核心语义用一句话写清“在什么条件下，哪种做法导致怎样的结果”；引文锚定材料中最直接的依据。假如作答用一两句话准确概括案例的意义，应当能命中主干卡。只服务于同一结论的设备数量、投入、时间、销量等不单列采分点，除非题目要求这些数据。主干关系卡的性质填“核心观点与破题定性”，不是微观案例细节。
   - 把代词（他、我们、该厂、当地等）还原为具体名称。
   - "原文" 必须逐字连续复制对应全局段号中的原句片段，一字不差，严禁用省略号！
   - "卡片性质" 只能选：
     * "核心观点与破题定性"（材料中总结升华的核心哲理、辩证关系或总括定性句）
     * "问题与原因"（材料反映的痛点、顾虑、短板、漏洞、失败教训原因）
     * "具体做法与工作内容"（材料中已实施或明确要求的具体举措、巡查动作、处置规范、转型策略）
     * "积极成效与价值"（做法带来的客观效果、收益变化、科学/社会价值）
     * "微观故事与纯数字背景"（纯引子铺垫、某个人物寒暄细节、孤立统计数字账目）
   - "对应任务块"：填【题干字面解析.字面任务块】中最匹配的那一项名称。
   - "是否具体案例细节"：布尔值（若只是单一微观故事细节或纯数字例证填 true，主干机制/病因/普适做法填 false）。
2. 将同一任务块内可归为一个事项的卡片归入 "功能簇列表"（簇编号 G1, G2...）。任务不同，即使讨论同一主题也分开建簇。簇数由材料和任务决定。
3. 【关键客观机制：问题镜像对策卡】仅当【是否含问题反推对策】为 true 时：
   - 对材料里只有问题或原因、缺少现成对策的情况，生成可行的对策核验卡（卡号 S01, S02...）。数量随实际病因而定。
   - 每张 S 卡必须绑定至少一张问题卡号到 "对应病因卡号"，写明建议任务块和能解决该病因的具体动作。程序按所绑定的病因归簇并引用原文，无须返回所属簇和原文。
   - 若【是否含问题反推对策】为 false，"镜像对策卡" 必须返回空数组 []。

返回 JSON：
{
  "功能簇列表": [{"簇编号": "G1", "簇名称": "...", "所属任务块": "...", "簇说明": "..."}],
  "材料卡片": [
    {
      "卡号": "M01",
      "所属簇": "G1",
      "对应任务块": "...",
      "全局段号": 1,
      "主体": "...",
      "卡片性质": "具体做法与工作内容",
      "是否具体案例细节": false,
      "核心语义": "28字以内客观陈述",
      "原文": "逐字连续复制材料原句"
    }
  ],
  "镜像对策卡": [
    {
      "卡号": "S01",
      "对应任务块": "...",
      "对应病因卡号": ["M03"],
      "主体": "对策建议",
      "卡片性质": "反推对策措施",
      "是否具体案例细节": false,
      "核心语义": "针对[村民怕收不回房与租客怕改造亏本]提出明确回收续租与补偿细则的措施"
    }
  ]
}"""


def extract_universal_cards(spec, paragraphs, api_key=None):
    para_map = {p["全局段号"]: p["原文"] for p in paragraphs}
    valid_blocks = set(spec["字面任务块"])
    payload = {"题干字面解析": spec, "给定材料段落": paragraphs}
    prompt = UNIVERSAL_CARD_PROMPT
    for _ in range(3):
        res = call_llm_json(prompt, payload, api_key=api_key, temperature=0.0, max_tokens=12000)
        clusters = res.get("功能簇列表", [])
        m_cards = res.get("材料卡片", [])
        s_cards = res.get("镜像对策卡", []) if spec["是否含问题反推对策"] else []
        errors = []
        problems = {card["卡号"]: card for card in m_cards if card.get("卡片性质") == "问题与原因"}
        derived_clusters = {}
        for card in s_cards:
            causes = card.get("对应病因卡号", [])
            block = card.get("对应任务块")
            if not causes or any(cause not in problems for cause in causes) or block not in valid_blocks:
                errors.append(f"{card.get('卡号')} 缺少有效病因卡或建议任务块")
                continue
            anchor = problems[causes[0]]
            key = (block, tuple(sorted({problems[cause]["所属簇"] for cause in causes})))
            if key not in derived_clusters:
                derived_clusters[key] = {
                    "簇编号": f"S-{len(derived_clusters) + 1}",
                    "簇名称": f"应对：{anchor['核心语义']}",
                    "所属任务块": block,
                }
            card.update({
                "所属簇": derived_clusters[key]["簇编号"],
                "全局段号": anchor["全局段号"],
                "原文": anchor["原文"],
            })
        clusters.extend(derived_clusters.values())
        all_cards = m_cards + s_cards

        cluster_ids = {c.get("簇编号") for c in clusters if c.get("簇编号")}
        if not clusters or len(cluster_ids) != len(clusters):
            errors.append("功能簇为空，或簇编号缺失、重复")
        if not m_cards:
            errors.append("未提取到材料卡片")
        clusters_by_id = {cluster.get("簇编号"): cluster.get("所属任务块") for cluster in clusters}
        if spec["是否含问题反推对策"] and len(s_cards) == 0:
            # 只有当材料本身也没有对策卡归入建议块时才强制要求生成镜像卡
            last_block = spec["字面任务块"][-1]
            has_direct_measure = any(c.get("对应任务块") == last_block for c in m_cards)
            if not has_direct_measure:
                errors.append("本题含反推对策任务块，但既无对应对策材料卡也未生成镜像对策卡 S01...")

        seen = set()
        for c in all_cards:
            cid = c.get("卡号", "")
            pno = c.get("全局段号")
            quote = c.get("原文", "")
            if not cid or cid in seen:
                errors.append(f"卡号缺失或重复: {cid}")
            seen.add(cid)
            if c.get("所属簇") not in cluster_ids:
                errors.append(f"{cid} 所属簇 {c.get('所属簇')} 未在功能簇列表中定义")
            block = c.get("对应任务块")
            if block not in valid_blocks or clusters_by_id.get(c.get("所属簇")) != block:
                errors.append(f"{cid} 的任务块无效，或与所属簇任务块不一致")
            if pno not in para_map or not is_verbatim_quote(quote, para_map[pno]):
                errors.append(f"{cid} 原文引文不在第{pno}段中: {quote[:45]}")

        missing_blocks = valid_blocks - {card.get("对应任务块") for card in all_cards}
        if missing_blocks:
            errors.append(f"以下题干任务块没有可核查的采分卡：{sorted(missing_blocks)}")

        if not errors:
            return {"功能簇列表": clusters, "全部卡片": all_cards}
        prompt = UNIVERSAL_CARD_PROMPT + "\n上轮输出未通过校验，请修正：" + "；".join(errors[:10])
    raise RuntimeError(f"通用材料与镜像卡片提取连续3次未通过校验：{errors[:8]}")


UNIVERSAL_ALIGN_PROMPT = """你是客观文本特征统计员。严禁评价答案优劣，只做机械连线与外观清点。
输入包含【卡片列表】、【字面任务块】和一批【参考答案】。对每份参考答案：
1. 核对它命中了哪些卡片（含 M 卡与 S 对策卡）：
   - "完整"：明确写出了该卡的核心事实、病因或具体可行的解决措施
   - "笼统"：仅用极简泛词一笔带过（缺乏具体对象或执行方式）
   每条命中必须选择该份参考答案里能支持该点的原句编号，填入 "证据编号"，不需要重新抄写。
2. 清点该份参考答案纯客观的【外观特征】（全部填布尔值或整数，不掺杂任何主观偏好）：
   - "首行有独立标题"：第一行是否为独立短标题（非"一、"或正文长句）
   - "有顶格受众称呼"：开头是否有"各位XX："、"广大市民朋友："等称呼行
   - "文末有落款单位或日期"：结尾是否有署名单位或日期行
   - "首段有总括破题或明确表态句"：开头是否先用一句话点明核心态度/总论点（如"不能一概而论，应辩证取舍"、"走出独特转型之路"）
   - "文末有行动号召祈使句"：结尾是否有呼吁受众参与/行动的祈使句
   - "有分条或分块标识"：是否使用序号（1. / 一、/ 事项一）或分段小标题组织内容
   - "各任务块估算字数"：若【字面任务块】有2项及以上，估算该答案在每个任务块上的大致字数（字典格式，键为任务块名，值为整数）；若只有1个任务块填空对象 {}。

返回 JSON：
{
  "对齐结果": [
    {
      "答案编号": 0,
      "命中列表": [{"卡号": "M01", "程度": "完整", "证据编号": ["1"]}],
      "外观特征": {
        "首行有独立标题": false,
        "有顶格受众称呼": false,
        "文末有落款单位或日期": false,
        "首段有总括破题或明确表态句": true,
        "文末有行动号召祈使句": false,
        "有分条或分块标识": true,
        "各任务块估算字数": {}
      }
    }
  ]
}"""


def align_universal_references(cards, blocks, ref_answers, api_key=None, batch_size=3):
    card_ids = {c["卡号"] for c in cards}
    compact_cards = [
        {
            "卡号": c["卡号"],
            "对应任务块": c["对应任务块"],
            "主体": c["主体"],
            "卡片性质": c["卡片性质"],
            "核心语义": c["核心语义"],
        }
        for c in cards
    ]
    batches = [
        [{"答案编号": i, "正文": ref_answers[i]["正文"]} for i in range(s, min(s + batch_size, len(ref_answers)))]
        for s in range(0, len(ref_answers), batch_size)
    ]

    def run_batch(chunk):
        ans_map = {item["答案编号"]: item["正文"] for item in chunk}
        span_maps = {number: index_answer_evidence(body) for number, body in ans_map.items()}
        indexed_refs = [
            {"答案编号": number, "原句": {key: span[2] for key, span in spans.items()}}
            for number, spans in span_maps.items()
        ]
        payload = {"字面任务块": blocks, "卡片列表": compact_cards, "参考答案批次": indexed_refs}
        prompt = UNIVERSAL_ALIGN_PROMPT
        for attempt in range(3):
            res = call_llm_json(prompt, payload, api_key=api_key, temperature=0.2 * attempt, max_tokens=10000)
            rows = res.get("对齐结果", [])
            errors = []
            if len(rows) != len(ans_map) or {r.get("答案编号") for r in rows} != ans_map.keys():
                errors.append("返回的答案编号不完整")
            for r in rows:
                aid = r.get("答案编号")
                if aid not in ans_map:
                    continue
                body = ans_map[aid]
                seen_cards = set()
                for hit in r.get("命中列表", []):
                    cid = hit.get("卡号")
                    deg = hit.get("程度")
                    q = resolve_answer_evidence(hit, body, span_maps[aid])
                    hit["答案引文"] = q
                    if cid not in card_ids or cid in seen_cards:
                        errors.append(f"答案{aid} 卡号不存在或重复: {cid}")
                    seen_cards.add(cid)
                    if deg not in {"完整", "笼统"}:
                        errors.append(f"答案{aid} {cid} 程度非法")
                    if not q:
                        errors.append(f"答案{aid} {cid} 缺少有效证据编号")
                if not isinstance(r.get("外观特征"), dict):
                    errors.append(f"答案{aid} 缺少外观特征对象")
            if not errors:
                return rows
            prompt = UNIVERSAL_ALIGN_PROMPT + "\n上轮证据校验未通过：" + "；".join(errors[:8])
        raise RuntimeError(f"通用参考答案统计连续3次校验未过: {errors[:4]}")

    aligned = []
    if batches:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
            for part in pool.map(run_batch, batches):
                aligned.extend(part)
    aligned.sort(key=lambda x: x["答案编号"])
    return aligned


def compile_universal_rubric(bundle, spec, constraints, card_pack, aligned_refs, mode_label="通用零预设预制"):
    """零人工主观预设的数学编译器：从题干硬指令 + 参考答案实证统计自动生长出外壳契约与分块配额。"""
    full_score = constraints["满分"]
    word_limit = constraints["字数上限"] or 400
    n_refs = max(5, len(aligned_refs))
    has_consensus = len(aligned_refs) >= 5
    blocks = spec["字面任务块"]

    # 1. 自动统计外壳契约项（完全由数据频率 + 题干字面硬指令决定，绝不硬编码“短评怎么样、指南怎么样”）
    def feat_rate(key):
        if not aligned_refs:
            return 0.0
        return sum(1 for r in aligned_refs if (r.get("外观特征") or {}).get(key)) / len(aligned_refs)

    rate_title = feat_rate("首行有独立标题")
    rate_salutation = feat_rate("有顶格受众称呼")
    rate_signoff = feat_rate("文末有落款单位或日期")
    rate_thesis = feat_rate("首段有总括破题或明确表态句")
    rate_cta = feat_rate("文末有行动号召祈使句")
    rate_structured = feat_rate("有分条或分块标识")

    shell_gates = []
    requirements = [
        item for item in spec.get("可核查要求", [])
        if not re.fullmatch(r"(?:观点|立场|态度)明确", item["题干引文"].strip())
    ]
    kinds = {item["类别"] for item in requirements}
    for number, item in enumerate(requirements, start=1):
        shell_gates.append({
            "门禁编号": f"R{number}",
            "名称": item["题干引文"],
            "分值": 2.0 if item["核心任务"] else 1.0,
            "判定依据": "题干明确要求",
            "核查标准": item["核查标准"],
            "核心任务": item["核心任务"],
        })
    if not spec["题干字面要求免格式"]:
        if spec["字面指定标题"]:
            shell_gates.append({
                "门禁编号": "F_FIXED_TITLE",
                "名称": f"首行使用题干指定标题《{spec['字面指定标题']}》",
                "分值": 1.0,
                "判定依据": "题干字面强制指定标题",
            })
        elif spec["题干是否明示写标题或文种名"] and has_consensus and rate_title >= 0.78:
            shell_gates.append({
                "门禁编号": "F_TITLE",
                "名称": f"具备独立文种标题（实证共识率 {rate_title:.0%}）",
                "分值": 1.0,
                "判定依据": f"{len(aligned_refs)}份答案中{rate_title:.0%}包含独立标题",
            })

        if spec["题干明示目标受众"] and has_consensus and rate_salutation >= 0.78:
            shell_gates.append({
                "门禁编号": "F_SALUTATION",
                "名称": f"开篇称呼或直呼目标受众（实证共识率 {rate_salutation:.0%}）",
                "分值": 0.5,
                "判定依据": f"{rate_salutation:.0%}参考答案包含受众称呼",
            })

        if not spec["字面指定标题"] and spec["题干是否明示写标题或文种名"] and has_consensus and rate_signoff >= 0.80:
            shell_gates.append({
                "门禁编号": "F_SIGNOFF",
                "名称": f"文末具备发文落款（实证共识率 {rate_signoff:.0%}）",
                "分值": 0.5,
                "判定依据": f"{rate_signoff:.0%}参考答案包含落款",
            })

    if spec["是否要求明确观点表态"]:
        pts = 2.0 if "看法" in "".join(blocks) or "评析" in bundle["题干与要求"] else 1.5
        shell_gates.append({
            "门禁编号": "F_THESIS",
            "名称": "明确表达核心观点或辩证态度",
            "分值": pts,
            "判定依据": "题干要求观点明确或谈看法",
        })

    if spec["题干字面要求呼吁号召"] and "受众沟通" not in kinds:
        shell_gates.append({
            "门禁编号": "F_CTA",
            "名称": "向目标受众发出明确行动呼吁",
            "分值": 2.0,
            "判定依据": "题干字面要求呼吁号召",
            "核查标准": "正文整体应让目标受众理解为什么行动、行动对象和可以执行的步骤。单有邀请语或打散的信息不算达成。",
            "核心任务": True,
        })

    has_organization = "组织归类" in kinds or any(
        re.search(r"条理|条目|层次|归类", item["题干引文"]) for item in requirements
    )
    if not has_organization and (
        len(blocks) >= 2 or re.search(r"条目清晰|层次分明|条理清晰|有条理|分类", bundle["题干与要求"])
    ):
        shell_gates.append({
            "门禁编号": "F_STRUCTURE",
            "名称": "分块/分条层次清晰（工作事项与内容或原因与建议边界分明）",
            "分值": 1.0,
            "判定依据": "题干要求区分任务或条理清晰",
        })

    # 外壳总分严格封顶在满分的 20% 以内，确保 80%~95% 分值永远锚定在材料事实上
    max_shell = round(full_score * 0.20, 2)
    raw_shell_sum = sum(g["分值"] for g in shell_gates)
    if raw_shell_sum > max_shell and raw_shell_sum > 0:
        budget = round(max_shell * 100)
        exact = [g["分值"] * budget / raw_shell_sum for g in shell_gates]
        cents = [math.floor(value) for value in exact]
        by_remainder = sorted(range(len(cents)), key=lambda index: exact[index] - cents[index], reverse=True)
        for index in by_remainder[:budget - sum(cents)]:
            cents[index] += 1
        for g, amount in zip(shell_gates, cents):
            g["分值"] = amount / 100
    shell_pool = round(sum(g["分值"] for g in shell_gates), 2)
    content_pool = round(full_score - shell_pool, 2)

    # 2. 计算各任务块的分值配额（例如 原因 vs 建议 按参考答案字数比自动切分，杜绝偏科刷分）
    block_quotas = {}
    if len(blocks) == 1:
        block_quotas[blocks[0]] = content_pool
    else:
        char_totals = {b: 0.0 for b in blocks}
        valid_cnt = 0
        for r in aligned_refs:
            bmap = (r.get("外观特征") or {}).get("各任务块估算字数") or {}
            if isinstance(bmap, dict) and sum(float(bmap.get(b, 0) or 0) for b in blocks) > 30:
                for b in blocks:
                    char_totals[b] += float(bmap.get(b, 0) or 0)
                valid_cnt += 1
        if valid_cnt > 0:
            total_chars = sum(char_totals.values())
            shares = {b: max(0.20, min(0.75, char_totals[b] / total_chars)) for b in blocks}
            norm = sum(shares.values())
            allocated = 0.0
            for idx, b in enumerate(blocks):
                if idx < len(blocks) - 1:
                    q = round((content_pool * shares[b] / norm) * 2) / 2
                    block_quotas[b] = q
                    allocated += q
                else:
                    block_quotas[b] = round(content_pool - allocated, 2)
        else:
            even = round((content_pool / len(blocks)) * 2) / 2
            for idx, b in enumerate(blocks):
                block_quotas[b] = even if idx < len(blocks) - 1 else round(content_pool - even * (len(blocks) - 1), 2)

    # 3. 统计每张卡片命中率并在各自任务块内按容量分配权重
    cards = card_pack["全部卡片"]
    clusters = card_pack["功能簇列表"]
    card_stats = {c["卡号"]: {"full": 0, "vague": 0, "quotes": []} for c in cards}
    for r in aligned_refs:
        for hit in r.get("命中列表", []):
            cid = hit["卡号"]
            if cid not in card_stats:
                continue
            if hit["程度"] == "完整":
                card_stats[cid]["full"] += 1
            else:
                card_stats[cid]["vague"] += 1
            q = hit["答案引文"].strip()
            if q and q not in card_stats[cid]["quotes"] and len(card_stats[cid]["quotes"]) < 5:
                card_stats[cid]["quotes"].append(q)

    raw_w = {}
    role_map = {}
    for c in cards:
        cid = c["卡号"]
        st = card_stats[cid]
        eff_rate = (st["full"] + 0.5 * st["vague"]) / n_refs
        nature = c["卡片性质"]
        is_case = bool(c.get("是否具体案例细节"))

        if nature == "微观故事与纯数字背景":
            raw_w[cid] = 0.0
            role_map[cid] = "非采分背景细节"
        elif is_case:
            raw_w[cid] = 0.7 + 0.3 * eff_rate
            role_map[cid] = "案例细节选答点"
        else:
            raw_w[cid] = 1.4 + 0.8 * eff_rate
            role_map[cid] = "材料支撑得分点"

    # 在每个任务块内部独立按 120% 弹性池缩放分值，确保原因池与建议池互不吞噬
    scored_cards = []
    for b in blocks:
        b_quota = block_quotas[b]
        b_cards = [c for c in cards if c["对应任务块"] == b]
        b_raw_sum = sum(raw_w[c["卡号"]] for c in b_cards)
        scale = (b_quota * 1.22 / b_raw_sum) if b_raw_sum > 0 else 0.0
        for c in b_cards:
            cid = c["卡号"]
            if raw_w[cid] <= 0 or b_quota <= 0:
                pts = 0.0
            else:
                pts = round(raw_w[cid] * scale * 4) / 4
                if pts < 0.5:
                    pts = 0.5
            st = card_stats[cid]
            scored_cards.append({
                **c,
                "计分角色": role_map[cid],
                "满分值": pts,
                "笼统系数": 0.5,
                "参考命中统计": f"{st['full']}完整/{st['vague']}笼统 (共{len(aligned_refs)}份)",
                "常见表述锚点": st["quotes"],
            })

    # 计算每个功能簇封顶值
    cluster_rules = []
    for cl in clusters:
        gid = cl["簇编号"]
        g_cards = [sc for sc in scored_cards if sc["所属簇"] == gid and sc["满分值"] > 0]
        raw_sum = round(sum(sc["满分值"] for sc in g_cards), 2)
        if raw_sum == 0:
            cap = 0.0
        else:
            case_ratio = sum(1 for sc in g_cards if sc.get("是否具体案例细节")) / len(g_cards)
            factor = 0.65 if (case_ratio >= 0.6 and len(g_cards) >= 2) else 0.88
            cap = min(raw_sum, round(max(0.75, raw_sum * factor) * 2) / 2)
        cluster_rules.append({
            "簇编号": gid,
            "簇名称": cl["簇名称"],
            "所属任务块": cl["所属任务块"],
            "簇内卡片总分": raw_sum,
            "簇封顶分": cap,
        })

    return {
        "预制模式": mode_label,
        "题目路径": bundle["题目路径"],
        "题干与要求": bundle["题干与要求"],
        "硬约束": constraints,
        "字面指令槽位": spec,
        "实证外观统计率": {
            "独立标题率": round(rate_title, 2),
            "受众称呼率": round(rate_salutation, 2),
            "文末落款率": round(rate_signoff, 2),
            "首段破题表态率": round(rate_thesis, 2),
            "行动号召率": round(rate_cta, 2),
            "分条结构率": round(rate_structured, 2),
        },
        "分池配置": {
            "满分": full_score,
            "内容总分池": content_pool,
            "各任务块封顶配额": block_quotas,
            "实证外壳门禁分池": shell_pool,
        },
        "外壳客观门禁列表": shell_gates,
        "核心要求限分比例": CORE_REQUIREMENT_CEILING,
        "字数与截断死规则": {
            "标准上限格数": word_limit,
            "物理截断格数": constraints["物理截断格数"],
            "规则说明": f"预留5格行尾标点容差；超{word_limit + 5}格起每25格扣1.0分（最多扣{round(full_score * 0.15, 1)}分）；超{constraints['物理截断格数']}格物理截断不计分。",
        },
        "功能簇封顶表": cluster_rules,
        "采分卡片表": scored_cards,
    }


UNIVERSAL_CHECK_PROMPT = """你是严格执行客观寻证的检核员。你看不到分值，严禁给分或主观评价文采。
输入包含【核对卡片清单】、【外壳门禁项】和【考生可见答卷】（已按物理答题卡截断）。
请完成三项纯事实核对：
1. 逐张核对【核对卡片清单】：
   - "完整"：清楚表达了该卡的核心事实、观点、具体工作内容或针对该病因的可行解决措施（允许口语、同义改写、合并表述）。一张卡包含多个并列必要要素时，必须覆盖这些要素才能填完整；例示成分无需全写。反推对策卡只写问题、未写解决动作算“未写”。
   - 判断论证中的案例时，先匹配关键事实或关系，再检查必要细节。大幅缩减手工环节造成质量下滑，与写出精确缩减比例和投诉倍数可以是同一个论证事实；题干未要求数字时，不能只因省略数字就判未写。若对一个复合要点只说对部分关键事实，判笼统并引用已写部分。
   - "笼统"：有实质相关内容，但缺少必要动作、对象、逻辑关系或并列要素。只有可以套用到任何材料的万能措辞，而且无法定位到该要点时，填“未写”。
   - "冲突"：与材料事实或因果方向完全相反（如把该舍弃的落后工艺说成必须死守，或写反核心数据结论）
   - 分类、主体和因果关系是要点的一部分。答卷明确把一个事实归到错误类、或把具体对策用于不相干的问题，不能仅凭文字出现就判完整。关系能从上下文读出时，不要求考生逐项写固定标题。
   - "未写"：未提及（可省略不输出，程序自动记为未写）
   凡状态为 "完整"、"笼统"、"冲突" 的卡片，必须选择【考生可见答卷】中能支持判定的原句编号，填入 "证据编号"。并列必要要素可引用多处，不需要抄写原句。
2. 逐项核对【外壳门禁项】：每个门禁编号均应返回布尔值 "通过" 和相关答卷原句编号，确无对应的原句时返回空数组。
   - 带"核查标准"的项目按全文判断，另填"程度"（达成、部分达成、未达成）和"说明"。关键关系、分组或受众行动表达完整才算达成；有可用的部分但存在具体缺口为部分达成；核心功能缺失或对象、关系严重错位为未达成。说明要点明具体缺口及影响，不能只复述评分项。达成或部分达成须有原句依据。短句、表格感条目、通俗表述可得全分，只要题干允许且读者能理解。
   - 独立要求只核查自己负责的能力。内容漏点由卡片处理，不能再用无关的漏点降低组织、针对性、号召力。判断对应关系时先找答卷中实际成立的关系，再看错配的影响。例如所有建议都被明确连到不相干的病因，应判未达成；已陈述的建议均能解决对应问题时，不能仅因遗漏其他问题判定针对性不足。
   - 注意：若门禁为指定固定标题，只有考生实际写出了该标题（允许微调书名号空格，但不能换别的标题或漏写标题）才算 true！
   - 若门禁要求明确表达观点，全文须能找到清楚的立场，按题干与核对卡片判断是否片面；题干没有规定位置时，无须写在首段。
   - 若门禁为分块/分条层次清晰，按题干要求的任务和可用的组织方式核对。仅有编号或换行不算通过。要求写供日常执行的指南时，正文应将相关动作归入对应事项。把职责不同的事项混在同一条、把同一职责拆散在多个无关条目、或以偶发时间代替事项，都填 false 并引用答卷原句；标题同时写出两项职责并不能证明已完成分类。若题干本身要求按时间、阶段组织，清晰的时间顺序可以通过。规范用语与平实的等价表述同等对待。
3. 抓出【零分冗余或偏题原句】：若考生答卷中存在明显不命中任何卡片的纯空洞排比套话、无材料依据的臆造内容、或大段照抄无关背景故事的句子，逐字复制最多3句最浪费字数的原句片段到数组 "零分冗余原句"（若全文紧凑干货则填空数组 []）。

返回 JSON：
{
  "卡片核对": [{"卡号": "M01", "状态": "完整", "证据编号": ["1"]}],
  "门禁核对": [{"门禁编号": "R1", "通过": false, "程度": "部分达成", "证据编号": ["1"], "说明": "已说明参与意义，但未给公众报名途径。"}],
  "零分冗余原句": []
}"""


UNIVERSAL_ARB_PROMPT = """同一份考生答卷在正序与逆序两次独立检核中对少数项目出现分歧。
请严格对照【考生可见答卷】与项目客观定义做出最终裁定，所有非"未写"卡片须返回答卷原句的编号。
核验结构时按题干要求检查事项归类，不凭编号判定。日常工作指南中以偶发时间代替事项、将同一职责拆散、或把不同职责混在同一条，不满足事项结构。题干要求按时间或阶段组织的除外。
卡片有多个必要要素时，引文须支持其全部才算完整，只覆盖一部分算笼统。万能套话无具体指向算未写。
案例中的非必要数字、年份、省略的例子不影响等价事实的认定；已表达部分关键事实时不判未写。
带"核查标准"的项目按全文裁定，返回程度、原句编号和具体说明。程度为达成、部分达成、未达成之一，只有达成才通过。达成和部分达成须列原句编号。
只核查该项负责的能力，不重复因为卡片漏点降级。核对实际成立的分类与问题对策关系；关系全部错配时判未达成。
返回 JSON：
{
  "卡片裁定": [{"卡号": "M01", "状态": "完整", "证据编号": ["1"]}],
  "门禁裁定": [{"门禁编号": "R1", "通过": false, "程度": "部分达成", "证据编号": ["1"], "说明": "..."}]
}"""


def _resolve_gate(gate, row, text, spans):
    quote = resolve_answer_evidence(row, text, spans)
    if "核查标准" not in gate:
        return {"通过": row["通过"], "答卷引文": quote} if isinstance(row.get("通过"), bool) else None
    level = row.get("程度")
    reason = row.get("说明")
    if level not in REQUIREMENT_LEVELS or not isinstance(reason, str) or not reason.strip():
        return None
    if level != "未达成" and not quote:
        return None
    if row.get("证据编号") and not quote:
        return None
    return {"通过": level == "达成", "程度": level, "答卷引文": quote, "说明": reason.strip()}


def _run_single_universal_pass(rubric, visible_text, reverse_order=False, api_key=None):
    active_cards = [
        {
            "卡号": sc["卡号"],
            "对应任务块": sc["对应任务块"],
            "主体": sc.get("主体", ""),
            "卡片性质": sc["卡片性质"],
            "核心语义": sc["核心语义"],
            "材料原文锚点": sc["原文"],
            "常见表述锚点": sc["常见表述锚点"][:3],
        }
        for sc in rubric["采分卡片表"]
        if sc["满分值"] > 0
    ]
    if reverse_order:
        active_cards = list(reversed(active_cards))

    gates = [
        {key: g[key] for key in ("门禁编号", "名称", "核查标准", "核心任务") if key in g}
        for g in rubric["外壳客观门禁列表"]
    ]
    valid_cids = {c["卡号"] for c in active_cards}
    gate_by_id = {g["门禁编号"]: g for g in gates}
    valid_gids = set(gate_by_id)
    spans = index_answer_evidence(visible_text)

    payload = {
        "题干与要求": rubric["题干与要求"],
        "核对卡片清单": active_cards,
        "外壳门禁项": gates,
        "考生可见答卷": {key: span[2] for key, span in spans.items()},
    }
    prompt = UNIVERSAL_CHECK_PROMPT
    for attempt in range(3):
        res = call_llm_json(prompt, payload, api_key=api_key, temperature=0.15 * attempt, max_tokens=8000)
        errors = []
        seen_cards = set()
        for row in res.get("卡片核对", []):
            cid = row.get("卡号")
            st = row.get("状态")
            q = resolve_answer_evidence(row, visible_text, spans)
            row["答卷引文"] = q
            if cid not in valid_cids or cid in seen_cards:
                errors.append(f"未知或重复卡号: {cid}")
            seen_cards.add(cid)
            if st not in {"完整", "笼统", "冲突", "未写"}:
                errors.append(f"{cid} 状态非法: {st}")
            if st in {"完整", "笼统", "冲突"} and not q:
                errors.append(f"{cid} 缺少有效证据编号")
        gate_rows = res.get("门禁核对", [])
        returned_gids = {g.get("门禁编号") for g in gate_rows}
        if returned_gids != valid_gids or len(gate_rows) != len(valid_gids):
            errors.append(f"门禁项缺失、重复或越界；应返回 {sorted(valid_gids)}")
        gmap = {}
        for row in gate_rows:
            gid = row.get("门禁编号")
            if gid in gate_by_id:
                decision = _resolve_gate(gate_by_id[gid], row, visible_text, spans)
                if decision is None:
                    errors.append(f"{gid} 缺少有效判定、具体说明或答卷证据")
                else:
                    gmap[gid] = decision
        fluff_valid = [
            fq for fq in res.get("零分冗余原句", [])
            if isinstance(fq, str) and fq.strip() and is_verbatim_quote(fq, visible_text)
        ]

        if not errors:
            cmap = {cid: {"状态": "未写", "答卷引文": ""} for cid in valid_cids}
            for row in res.get("卡片核对", []):
                if row["状态"] != "未写":
                    cmap[row["卡号"]] = {"状态": row["状态"], "答卷引文": row.get("答卷引文", "")}
            return {"卡片判定": cmap, "门禁判定": gmap, "零分冗余原句": fluff_valid}
        prompt = UNIVERSAL_CHECK_PROMPT + "\n上轮输出未通过校验，请修正：" + "；".join(errors[:8])
    raise RuntimeError(f"答卷检核连续3次未通过校验：{errors[:8]}")


def compute_universal_ledger(rubric, total_grids, overflow_text, pass_data):
    """纯 Python 确定性算分并生成逐分可查的扣分诊断账本。"""
    full_score = rubric["硬约束"]["满分"]
    word_limit = rubric["硬约束"]["字数上限"]
    pools = rubric["分池配置"]
    block_quotas = pools["各任务块封顶配额"]

    cmap = pass_data["卡片判定"]
    gmap = pass_data["门禁判定"]

    cluster_sums = {cr["簇编号"]: 0.0 for cr in rubric["功能簇封顶表"]}
    card_ledger = []
    for sc in rubric["采分卡片表"]:
        if sc["满分值"] <= 0:
            continue
        cid = sc["卡号"]
        dec = cmap.get(cid, {"状态": "未写", "答卷引文": ""})
        st = dec["状态"]
        if st == "完整":
            pts = sc["满分值"]
        elif st == "笼统":
            pts = round(sc["满分值"] * sc["笼统系数"], 2)
        else:
            pts = 0.0
        cluster_sums[sc["所属簇"]] += pts
        card_ledger.append({
            "卡号": cid,
            "所属簇": sc["所属簇"],
            "对应任务块": sc["对应任务块"],
            "全局段号": sc["全局段号"],
            "卡片性质": sc["卡片性质"],
            "计分角色": sc["计分角色"],
            "核心语义": sc["核心语义"],
            "材料原文": sc["原文"],
            "满分值": sc["满分值"],
            "判定": st,
            "实得分": pts,
            "失分": round(sc["满分值"] - pts, 2),
            "答卷引文": dec["答卷引文"],
        })

    # 先做功能簇封顶，再按任务块做配额封顶，最后汇总到内容总池
    block_raw_totals = {b: 0.0 for b in block_quotas}
    cluster_breakdown = []
    for cr in rubric["功能簇封顶表"]:
        if cr["簇封顶分"] <= 0:
            continue
        gid = cr["簇编号"]
        raw_g = round(cluster_sums[gid], 2)
        cap_g = min(cr["簇封顶分"], raw_g)
        block_raw_totals[cr["所属任务块"]] = block_raw_totals.get(cr["所属任务块"], 0.0) + cap_g
        cluster_breakdown.append({
            "簇编号": gid,
            "簇名称": cr["簇名称"],
            "所属任务块": cr["所属任务块"],
            "簇内累计": raw_g,
            "簇封顶分": cr["簇封顶分"],
            "实得簇分": round(cap_g, 2),
        })

    block_breakdown = []
    content_sum = 0.0
    for b, quota in block_quotas.items():
        raw_b = round(block_raw_totals.get(b, 0.0), 2)
        capped_b = min(quota, raw_b)
        content_sum += capped_b
        block_breakdown.append({
            "任务块": b,
            "块内簇累计": raw_b,
            "块配额上限": quota,
            "实得块分": round(capped_b, 2),
        })
    content_score = min(pools["内容总分池"], round(content_sum, 2))

    # 外壳客观门禁分
    shell_score = 0.0
    gate_ledger = []
    hard_deductions = []
    requirement_deductions = []
    failed_core = []
    for g in rubric["外壳客观门禁列表"]:
        gid = g["门禁编号"]
        decision = gmap.get(gid, {})
        passed = bool(decision.get("通过"))
        level = decision.get("程度")
        pts = g["分值"] * REQUIREMENT_LEVELS[level] if level is not None else (g["分值"] if passed else 0.0)
        shell_score += pts
        entry = {
            "门禁编号": gid,
            "名称": g["名称"],
            "分值": g["分值"],
            "通过": passed,
            "实得分": pts,
            "答卷引文": decision.get("答卷引文", ""),
        }
        if "核查标准" in g:
            entry.update({"核查标准": g["核查标准"], "程度": level, "说明": decision.get("说明", "")})
            if not passed:
                requirement_deductions.append(f"【题干要求】{g['名称']}，{level}（{pts:g}/{g['分值']:g}分）。{entry['说明']}")
            if g.get("核心任务") and level == "未达成":
                failed_core.append(g["名称"])
        elif not passed:
            hard_deductions.append(f"未满足【{g['名称']}】（丢 {g['分值']} 分）")
        gate_ledger.append(entry)

    # 超字数阶梯扣分
    over_penalty = 0.0
    if word_limit and total_grids > word_limit + 5:
        excess = total_grids - word_limit
        steps = math.ceil(excess / 25)
        over_penalty = min(round(full_score * 0.15, 1), float(steps * 1.0))
        msg = f"总字数 {total_grids} 格，超上限 {word_limit} 格共 {excess} 格，扣 {over_penalty} 分"
        if overflow_text.strip():
            msg += f"（第 {rubric['硬约束']['物理截断格数']} 格后的 {count_exam_grids(overflow_text)} 格已被物理截断不计分）"
        hard_deductions.append(msg)

    ceiling_ratio = rubric.get("核心要求限分比例", CORE_REQUIREMENT_CEILING)
    score_limit = math.floor(full_score * ceiling_ratio * 2) / 2 if failed_core else full_score
    uncapped_total = max(0.0, min(full_score, content_score + shell_score - over_penalty))
    raw_total = min(score_limit, uncapped_total)
    if raw_total < uncapped_total:
        requirement_deductions.append(f"核心要求【{'、'.join(failed_core)}】未达成，本题总分上限为 {score_limit:g} 分")
    final_score = round(raw_total * 2) / 2

    # 生成面向考生的结构化精准扣分点评（完全基于数学账本，零废话）
    missed_core = sorted(
        [c for c in card_ledger if c["判定"] in {"未写", "冲突"} and c["满分值"] >= 0.75],
        key=lambda x: (-x["满分值"], x["卡号"]),
    )
    vague_items = [c for c in card_ledger if c["判定"] == "笼统"]
    conflict_items = [c for c in card_ledger if c["判定"] == "冲突"]
    fluff_spans = pass_data.get("零分冗余原句", [])
    wasted_grids = sum(count_exam_grids(s) for s in fluff_spans)

    diagnostic_lines = []
    for b in block_breakdown:
        if b["实得块分"] < b["块配额上限"]:
            gap = round(b["块配额上限"] - b["实得块分"], 2)
            diagnostic_lines.append(f"【{b['任务块']}】实得 {b['实得块分']}/{b['块配额上限']} 分（尚差 {gap} 分）")
    for hd in hard_deductions:
        diagnostic_lines.append(f"【格式/结构/字数失分】{hd}")
    diagnostic_lines.extend(requirement_deductions)
    for c in conflict_items:
        diagnostic_lines.append(
            f"【事实冲突】你写的“{c['答卷引文']}”与材料第{c['全局段号']}段核心意思冲突（{c['核心语义']}，本项 0/{c['满分值']:g} 分）"
        )
    for c in missed_core[:6]:
        if c["判定"] == "未写":
            diagnostic_lines.append(
                f"【漏答要点】[{c['对应任务块']}] 漏写：{c['核心语义']}（本项 0/{c['满分值']:g} 分，依据：资料第{c['全局段号']}段“{c['材料原文'][:38]}”）"
            )
    for c in vague_items[:4]:
        diagnostic_lines.append(
            f"【部分采分】你写的“{c['答卷引文']}”未完整表达{c['核心语义']}（本项 {c['实得分']:g}/{c['满分值']:g} 分）"
        )
    if fluff_spans:
        joined = " / ".join(f"“{s[:28]}”" for s in fluff_spans)
        diagnostic_lines.append(f"【无效占格约 {wasted_grids} 格】以下句子未命中任何采分卡（0分）：{joined}")

    return {
        "最终得分": final_score,
        "满分": full_score,
        "总格数": total_grids,
        "字数上限": word_limit,
        "被截断溢出文字": overflow_text,
        "分项得分": {
            "内容总得分": content_score,
            "内容池上限": pools["内容总分池"],
            "外壳门禁得分": round(shell_score, 2),
            "外壳池上限": pools["实证外壳门禁分池"],
            "超字数扣分": over_penalty,
            "题干核心任务限分": round(uncapped_total - raw_total, 2),
        },
        "各任务块得分": block_breakdown,
        "功能簇得分": cluster_breakdown,
        "外壳门禁明细": gate_ledger,
        "逐卡账本": card_ledger,
        "零分冗余原句": fluff_spans,
        "精准扣分点评": diagnostic_lines,
    }


def _separate_fixed_title(rubric, visible_text):
    expected = rubric.get("字面指令槽位", {}).get("字面指定标题")
    if not expected or not any(g["门禁编号"] == "F_FIXED_TITLE" for g in rubric["外壳客观门禁列表"]):
        return rubric, visible_text, {}

    first, _, rest = visible_text.partition("\n")
    correct = is_verbatim_quote(first, expected) and is_verbatim_quote(expected, first)
    # 单独一行的短标题交给标题规则，正文其他评分项只读取后续正文。
    standalone = bool(rest.strip() and 0 < len(first.strip()) <= max(25, len(expected))
                      and not re.search(r"[，。！？：；,.!?:;]", first)
                      and not re.match(r"^\s*(?:第[一二三四五六七八九十\d]+|[一二三四五六七八九十\d]+[、.)）]|[（(])", first))
    review_rubric = {**rubric, "外壳客观门禁列表": [g for g in rubric["外壳客观门禁列表"] if g["门禁编号"] != "F_FIXED_TITLE"]}
    title_result = {"F_FIXED_TITLE": {"通过": correct, "答卷引文": first if correct or standalone else ""}}
    return review_rubric, rest.lstrip() if correct or standalone else visible_text, title_result


def grade_universal_answer(rubric, answer_text, api_key=None, dual_pass=True):
    cleaned = clean_reference_body(answer_text)
    total_grids = count_exam_grids(cleaned)
    cutoff = rubric["硬约束"]["物理截断格数"]
    visible_text, overflow_text = truncate_by_exam_grids(cleaned, cutoff)
    review_rubric, visible_text, title_result = _separate_fixed_title(rubric, visible_text)

    if not dual_pass:
        p1 = _run_single_universal_pass(review_rubric, visible_text, False, api_key=api_key)
        p1["门禁判定"].update(title_result)
        res = compute_universal_ledger(rubric, total_grids, overflow_text, p1)
        res["双盲审计"] = None
        return res

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_run_single_universal_pass, review_rubric, visible_text, False, api_key)
        f2 = pool.submit(_run_single_universal_pass, review_rubric, visible_text, True, api_key)
        p1, p2 = f1.result(), f2.result()
    p1["门禁判定"].update(title_result)
    p2["门禁判定"].update(title_result)

    s1 = compute_universal_ledger(rubric, total_grids, overflow_text, p1)
    s2 = compute_universal_ledger(rubric, total_grids, overflow_text, p2)

    card_lookup = {sc["卡号"]: sc for sc in rubric["采分卡片表"] if sc["满分值"] > 0}
    gate_lookup = {g["门禁编号"]: g for g in rubric["外壳客观门禁列表"]}

    disputed_cards = []
    for cid, sc in card_lookup.items():
        if p1["卡片判定"][cid]["状态"] != p2["卡片判定"][cid]["状态"]:
            disputed_cards.append({
                "卡号": cid,
                "核心语义": sc["核心语义"],
                "材料原文锚点": sc["原文"],
                "正序判定": p1["卡片判定"][cid],
                "逆序判定": p2["卡片判定"][cid],
            })
    disputed_gates = []
    for gid, g in gate_lookup.items():
        first, second = p1["门禁判定"][gid], p2["门禁判定"][gid]
        if first.get("程度", first["通过"]) != second.get("程度", second["通过"]):
            disputed_gates.append({
                "门禁编号": gid,
                "名称": g["名称"],
                **({"核查标准": g["核查标准"]} if "核查标准" in g else {}),
                "正序判定": first,
                "逆序判定": second,
            })

    merged = {
        "卡片判定": dict(p1["卡片判定"]),
        "门禁判定": dict(p1["门禁判定"]),
        "零分冗余原句": p1["零分冗余原句"] or p2["零分冗余原句"],
    }
    if disputed_cards or disputed_gates:
        spans = index_answer_evidence(visible_text)
        arb = call_llm_json(
            UNIVERSAL_ARB_PROMPT,
            {"题干与要求": rubric["题干与要求"], "考生可见答卷": {key: span[2] for key, span in spans.items()},
             "分歧卡片": disputed_cards, "分歧门禁": disputed_gates},
            api_key=api_key,
            temperature=0.0,
            max_tokens=4000,
        )
        for item in arb.get("卡片裁定", []):
            cid = item.get("卡号")
            st = item.get("状态")
            q = resolve_answer_evidence(item, visible_text, spans)
            if cid in card_lookup and st in {"完整", "笼统", "冲突", "未写"}:
                if st == "未写" or q:
                    merged["卡片判定"][cid] = {"状态": st, "答卷引文": "" if st == "未写" else q}
        disputed_ids = {g["门禁编号"] for g in disputed_gates}
        decided = set()
        for item in arb.get("门禁裁定", []):
            gid = item.get("门禁编号")
            if gid in disputed_ids and (decision := _resolve_gate(gate_lookup[gid], item, visible_text, spans)) is not None:
                merged["门禁判定"][gid] = decision
                decided.add(gid)
        missing = {g["门禁编号"] for g in disputed_gates if "核查标准" in g} - decided
        if missing:
            raise ValueError(f"题干要求复核缺少有效裁定：{sorted(missing)}")

    final_res = compute_universal_ledger(rubric, total_grids, overflow_text, merged)
    final_res["双盲审计"] = {
        "正序独立分": s1["最终得分"],
        "逆序独立分": s2["最终得分"],
        "双盲分差": round(abs(s1["最终得分"] - s2["最终得分"]), 2),
        "分歧项目数": len(disputed_cards) + len(disputed_gates),
    }
    return final_res


UPGRADE_ANSWER_PROMPT = """你是申论答卷升格编辑。保留原卷有证据的有效信息，按题干完成改写。
严格遵守以下手术规则：
1. 【重组事项】：按题干确定结构。如果原稿靠故事时间串联，题干却要求写供日常执行的指南，必须先将同类动作集中到对应事项，拆开混写的不同职责，再用可反复执行的通用规范表述。删除偶发人名和时间，但保留真实的执行时间要求、核查对象和处置动作。评价、报道等需要实例的任务可以保留必要的事实。原有条理已经符合任务的段落优先保留。
   只有材料明确规定为常态时，才可写“每日”、固定时点、频次、统一阈值等制度性要求；一次案例中的时刻和做法不能直接改写为普遍规定。不要把不同工作职责合成一个混合事项。
2. 【精准删冗】：直接删掉【零分冗余与错误句】，腾出宝贵方格。
3. 【补漏升格】：将【待补齐的高价值失分点】与【未通过的外壳门禁（如指定标题、首段明确观点表态）】用最精炼的短句嵌入对应段落或条目中；将【笼统折损点】补上具体动作词。
   字数冲突时，优先题干必答任务、主体立场和关键事实。候选补漏点按优先级取舍，允许删除低优先级的铺垫、同义反复和例证细节。
4. 【严守字数铁律】：修改后的整篇答案按申论方格纸计算，总字数必须严格不超过【目标字数上限】！宁可精简助词“的、了、着”，也绝不允许超格！
5. 在 "手术改动说明" 数组中列出 2~4 条清晰对比（格式："删去‘...’ -> 补入‘...’（补齐XX采分点）"）。

返回 JSON：
{
  "升格后正文": "...",
  "手术改动说明": ["..."]
}"""


UPGRADE_REVIEW_PROMPT = """独立复核升格稿。只报告可以指认的严重问题：新增无依据的具体事实、
把一次个案升级为例行制度，或在供日常执行的指南中把不同职责混成一项。
合法的等价概括、依据明确的合理反推对策、题干要求的时间顺序，均不列为问题。
从编号片段中选出造成问题的原句，解释具体哪里有问题。无需给分。
输出 JSON：{"问题":[{"证据编号":["1"],"说明":"..."}]}，无严重问题时返回空数组。"""


def _review_upgrade(rubric, candidate, api_key):
    spans = index_answer_evidence(candidate)
    sources = list(dict.fromkeys(card["原文"] for card in rubric["采分卡片表"]))
    review = call_llm_json(
        UPGRADE_REVIEW_PROMPT,
        {"题干": rubric["题干与要求"], "材料依据": sources,
         "稿件原句": {key: span[2] for key, span in spans.items()}},
        api_key=api_key,
    )
    issues = []
    for item in review["问题"]:
        quote = resolve_answer_evidence(item, candidate, spans)
        if not quote or not item["说明"].strip():
            raise ValueError("升格复核返回的问题缺少有效证据或说明")
        issues.append(f"{quote}：{item['说明']}")
    return issues


def upgrade_user_answer_closed_loop(rubric, original_answer, initial_grade_result, api_key=None):
    """在用户原答案上做微创修改，并通过方格计字检验 + 重新走批改引擎验分闭环。"""
    word_limit = rubric["硬约束"]["字数上限"] or 400
    kept_hits = [
        {"已得分点": c["核心语义"], "考生原句": c["答卷引文"], "材料依据": c["材料原文"]}
        for c in initial_grade_result["逐卡账本"]
        if c["判定"] == "完整"
    ]
    missing_targets = [
        {
            "所属任务块": c["对应任务块"],
            "待补要点": c["核心语义"],
            "材料依据": c["材料原文"],
            "当前状态": c["判定"],
            "分值": c["满分值"],
        }
        for c in sorted(initial_grade_result["逐卡账本"], key=lambda x: -x["失分"])
        if c["失分"] >= 0.5
    ][:10]
    failed_gates = [
        g["名称"] for g in initial_grade_result["外壳门禁明细"] if not g["通过"]
    ]

    payload = {
        "题干与要求": rubric["题干与要求"],
        "目标字数上限": word_limit - 5,
        "考生原答卷": clean_reference_body(original_answer),
        "需保留的已满分原句": kept_hits,
        "建议删减的零分冗余或超限部分": initial_grade_result["零分冗余原句"],
        "必须补齐的外壳门禁": failed_gates,
        "待补齐的高价值失分点": missing_targets,
    }

    prompt = UPGRADE_ANSWER_PROMPT
    feedback = []
    for attempt in range(3):
        res = call_llm_json(prompt, payload, api_key=api_key, temperature=0.0, max_tokens=4000)
        cand = res.get("升格后正文", "").strip()
        grids = count_exam_grids(cand)
        if cand and grids <= word_limit:
            feedback = _review_upgrade(rubric, cand, api_key)
            if not feedback:
                upgraded_text = cand
                change_notes = res.get("手术改动说明", [])
                break
        else:
            target = max(1, word_limit - max(10, round(word_limit * 0.08)))
            feedback = [f"正文实占 {grids} 格。改写目标 {target} 格，优先删过渡、次要细节和重复结论。"]
            payload["目标字数上限"] = target
            preferred = payload["待补齐的高价值失分点"]
            payload["待补齐的高价值失分点"] = preferred[:max(1, (len(preferred) + 1) // 2)]
        payload["待修订的上一稿"] = cand
        prompt = UPGRADE_ANSWER_PROMPT + "\n上一稿存在以下问题，请逐项修复：\n" + "\n".join(feedback)
    else:
        raise ValueError(f"三次升格均未通过复核：{'；'.join(feedback)}")

    # 闭环重测验分：把改好的答案重新送进同一个批改引擎真实验分，绝不自吹自擂
    regraded = grade_universal_answer(rubric, upgraded_text, api_key=api_key)
    return {
        "原卷得分": initial_grade_result["最终得分"],
        "原卷格数": initial_grade_result["总格数"],
        "升格后得分": regraded["最终得分"],
        "升格后格数": regraded["总格数"],
        "实测提分": round(regraded["最终得分"] - initial_grade_result["最终得分"], 2),
        "升格后正文": upgraded_text,
        "手术改动说明": change_notes,
        "升格后分项得分": regraded["分项得分"],
    }


def select_universal_references(references, ref_limit):
    unique_refs = {}
    for ref in references:
        unique_refs.setdefault(ref["正文"], ref)
    return list(unique_refs.values())[:ref_limit]


def prefab_universal_question(question_md_path, api_key=None, ref_limit=15, mode_label="通用零预设预制", *, exclude_human_scored=False):
    bundle = load_question_bundle(question_md_path, exclude_human_scored=exclude_human_scored)
    refs = select_universal_references(bundle["参考答案"], ref_limit)
    return prefab_universal_bundle(bundle, refs, api_key=api_key, mode_label=mode_label), bundle


def prefab_universal_bundle(bundle, refs, api_key=None, mode_label="通用零预设预制"):
    constraints = parse_hard_constraints(bundle["题干与要求"])
    spec = _extract_universal_spec(bundle["题干与要求"], api_key)
    card_pack = extract_universal_cards(spec, bundle["材料段落"], api_key=api_key)
    aligned = align_universal_references(
        card_pack["全部卡片"], spec["字面任务块"], refs, api_key=api_key
    )
    return compile_universal_rubric(bundle, spec, constraints, card_pack, aligned, mode_label=mode_label)
