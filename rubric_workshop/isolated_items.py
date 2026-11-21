from concurrent.futures import ThreadPoolExecutor
import re
from rubric_workshop.blind_batch import call_llm_json, count_pure_exam_chars


def build(source, *, calibrate=False):
    prompt = """你是命题组，只读题目、要求、材料。先用题干动词确定答题深度，在考生字限内独立写一份最简完整答案，再按它真正交付的意义划分内容。每项只问一个是非问题，例如是否表达了某种关系。两个可分别为真为假的独立功能必须分别给分，不能以一个名称捆绑。方面题以有对象的功能、实践或关系为最低充分；机制题才要求实现途径；案例题须分对象；应用文以读者需要知道、能采取的行动为内容，组织项单独问整体沟通效果。措辞、案例、步骤细节可互相替代。每项返回[分区,名称,完整得分所需的一个意思,部分达成的边界,可省略或替代表达,材料段号数组,权重整数1至3]。组织项依据题干可无材料段。分区别称严格复制预算键。只返回 {"rows":[...]}。"""
    payload = dict(source)
    if calibrate:
        limit = int(re.search(r"不超过\s*(\d+)\s*字", source["question"])[1])
        for attempt in range(3):
            draft = call_llm_json(
                '仅据所给材料，按题干要求写一份覆盖不同方面的完整考场作答。留出卷面空间。返回 {"answer":"正文"}。',
                payload,
            )["answer"]
            if isinstance(draft, str) and 0 < count_pure_exam_chars(draft) <= limit:
                break
            payload["字限提醒"] = limit
        else:
            raise ValueError("完整作答超过题面字限")
        payload["材料盲写稿"] = draft
        prompt += "先复核盲稿是否漏了题干任务，可补漏。建立分值前先按题干要求的概括层次合并：两个事实可分别为真，只说明事实不同；只有删去一个会使读者少知道一个独立答案方面，才可独立给分。方面题先抽取材料自己总结的方面及不同功能；举例或成因不能升级为必写。每项满分只写一个最小充分命题，不得用多重并列举例定义满分。在上下文已明确总对象时，分项无需重复限定语。流程按读者负责的环节区分，工作人员的后续处理另看题意。部分分必须指出只满足最小命题的哪一部分。"
    for attempt in range(3):
        try:
            rows = call_llm_json(prompt, payload)["rows"]
            items = []
            for zone, name, full, partial, optional, refs, weight in rows:
                if (
                    zone not in source["budget"]
                    or not all(
                        isinstance(v, str) and v.strip()
                        for v in (name, full, partial, optional)
                    )
                    or type(weight) != int
                    or weight not in (1, 2, 3)
                ):
                    raise ValueError("请核对分区、短答及权重")
                if (
                    not isinstance(refs, list)
                    or (zone == "内容" and not refs)
                    or any(ref not in source["materials"] for ref in refs)
                ):
                    raise ValueError("段号请使用原材料的字符串键")
                items.append(
                    dict(
                        zone=zone,
                        name=name,
                        full=full,
                        partial=partial,
                        optional=optional,
                        refs=refs,
                        weight=weight,
                    )
                )
            if len({i["name"] for i in items}) != len(items):
                raise ValueError("名称重复")
            for zone, budget in source["budget"].items():
                group = [i for i in items if i["zone"] == zone]
                units = round(budget * 4)
                if not group or len(group) > units:
                    raise ValueError("分区项目数不符")
                total = sum(i["weight"] for i in group)
                shares = [
                    divmod((units - len(group)) * i["weight"], total) for i in group
                ]
                left = units - len(group) - sum(q for q, _ in shares)
                bonus = sorted(range(len(group)), key=lambda j: -shares[j][1])[:left]
                for j, i in enumerate(group):
                    i["points"] = (1 + shares[j][0] + (j in bonus)) / 4
            return dict(
                isolated_items=True,
                items=items,
                **({"material_draft": draft} if calibrate else {}),
            )
        except (ValueError, KeyError, TypeError) as exc:
            if attempt == 2:
                raise
            payload["校正"] = str(exc)


def compile_boundaries(source, rubric):
    data = {
        "题干": source["question"],
        "材料": source["materials"],
        "冻结表": rubric["items"],
    }
    prompt = '保留现有各项预算与任务。为每项写两个可以判断真假的正向得分条件，每个值半项分：第一条是答案实际交代的最小有效信息，第二条是题干深度所需的完整关系；若表达不可再分，两个条件相同。先归并相互替代的例证，允许用一句有效概括同时满足两个条件。各项互斥，不能把其他项目的措辞当作本项最小有效信息，不能凭未超字数给空白卷分。空喊要做某事不能获得如何做的分。概括方面题具体对象和功能已清楚时不要要求成因；同一句描述如果充分，可以同时证明两条。给出 {"conditions":[["项目名","第一条正向命题","第二条正向命题"]]}，顺序同冻结表。'
    for attempt in range(3):
        rows = call_llm_json(prompt, data).get("conditions")
        if (
            isinstance(rows, list)
            and len(rows) == len(rubric["items"])
            and all(
                isinstance(row, list)
                and len(row) == 3
                and row[0] == item["name"]
                and all(isinstance(v, str) and v.strip() for v in row)
                for row, item in zip(rows, rubric["items"])
            )
        ):
            return dict(rubric, positive_conditions=[row[1:] for row in rows])
        data["校正"] = "核对行数、项目名称及条件文本"
    raise ValueError("得分条件与冻结表不匹配")


def grade_conditions(source, rubric, sentences):
    items = rubric["items"]
    conditions = rubric["positive_conditions"]
    data = {
        "题干": source["question"],
        "答卷": list(enumerate(sentences, 1)),
        "判据": [[item["name"], *pair] for item, pair in zip(items, conditions)],
    }
    prompt = '逐项独立审读。每条命题独立判断整篇答卷是否实际表达了所需关系，允许上下文承接、十分具体的举例及概括等义；题干提供的话题本身不能充当考生的答案。凡未表达、主体错置、观点被后文否定，都返回空数组。两条条件如果同义必须同判。对每个命题返回直接支持它的句子编号数组。组织、字数等要求在作答确实履行该文种功能时检查。逐项给出理由，指出实际给分关系或具体漏答。返回 {"rows":[["原项目名",[第一条支持句编号],[第二条支持句编号],"理由"]]}。'
    for attempt in range(3):
        try:
            rows = (
                call_llm_json(prompt, data)["rows"]
                if sentences
                else [[i["name"], [], [], "空白答卷"] for i in items]
            )
            if len(rows) != len(items):
                raise ValueError("项目数量不符")
            marks = []
            for item, conditions_pair, (name, first, second, why) in zip(
                items, conditions, rows
            ):
                if (
                    name != item["name"]
                    or not isinstance(why, str)
                    or not why.strip()
                    or not all(
                        isinstance(v, list)
                        and all(type(i) == int and 1 <= i <= len(sentences) for i in v)
                        for v in (first, second)
                    )
                ):
                    raise ValueError("项目名或证据无效")
                if conditions_pair[0] == conditions_pair[1] and bool(first) != bool(
                    second
                ):
                    raise ValueError("相同条件结论矛盾")
                marks.append(
                    dict(
                        name=name,
                        score=item["points"] * (bool(first) + bool(second)) / 2,
                        why=why,
                        citations=[
                            sentences[i - 1] for i in sorted(set(first + second))
                        ],
                        conditions=[
                            dict(condition=c, quotes=[sentences[i - 1] for i in refs])
                            for c, refs in zip(conditions_pair, (first, second))
                        ],
                    )
                )
            return marks
        except (ValueError, TypeError, KeyError) as exc:
            if attempt == 2:
                raise
            data["校正"] = str(exc)


def grade(source, rubric, answer):
    from scoring.engine import truncate_by_exam_grids

    limit = int(re.search(r"不超过\s*(\d+)\s*字", source["question"])[1])
    answer = truncate_by_exam_grids(answer, limit)[0]
    sentences = [
        x.strip() for x in re.findall(r"[^。！？；\n]+[。！？；]?", answer) if x.strip()
    ]

    if rubric.get("positive_conditions"):
        return grade_conditions(source, rubric, sentences)
    excerpts = {}
    if rubric.get("select_evidence") and sentences:
        names = [item["name"] for item in rubric["items"] if item["zone"] == "内容"]
        request = {
            "答卷": list(enumerate(sentences, 1)),
            "评分项": [item for item in rubric["items"] if item["zone"] == "内容"],
        }
        for attempt in range(3):
            selected = call_llm_json(
                '从答卷为每项选出直接支持、相反或用于消解指代的全部原句。只有属于本项的实际信息才选。仅提及整题主题不能支持各项。返回 {"items":{"逐字项目名":[句号编号]}}，没有相关原句给空数组。',
                request,
            ).get("items")
            if (
                isinstance(selected, dict)
                and set(selected) == set(names)
                and all(
                    isinstance(ids, list)
                    and all(type(n) == int and 1 <= n <= len(sentences) for n in ids)
                    for ids in selected.values()
                )
            ):
                excerpts = {
                    name: [sentences[n - 1] for n in sorted(set(ids))]
                    for name, ids in selected.items()
                }
                break
            request["校正"] = "核对全部项目名及整数编号"
        else:
            raise ValueError("取证结果与评分项不匹配")

    def judge(item):
        relevant = excerpts.get(item["name"], sentences)
        evidence = {k: source["materials"][k] for k in item["refs"]}
        data = dict(
            题目=source["question"],
            材料=evidence,
            规则={k: item[k] for k in ("zone", "name", "full", "partial", "optional")},
            答卷=relevant,
        )
        prompt = '只评给定一项。先找全文所有相关语句。考生用上位概括或例证明确表达本项核心关系即完整，不追加其他项、材料中的例子或未明问的解释。结合上下文指代、主体、事实范围；无关套话为零。组织项看整体读者效果。返回 {"level":2,"refs":[1],"why":"简短给分或缺失理由"}，2完整1部分0缺失错误，编号从1开始。'
        for attempt in range(3):
            try:
                r = (
                    call_llm_json(prompt, data)
                    if relevant
                    else dict(level=0, refs=[], why="答卷没有表达本项所需的信息")
                )
                level = r["level"]
                refs = r["refs"]
                if (
                    type(level) != int
                    or level not in (0, 1, 2)
                    or not isinstance(refs, list)
                    or (level and not refs)
                    or any(type(n) != int or not 1 <= n <= len(relevant) for n in refs)
                    or not isinstance(r["why"], str)
                    or not r["why"].strip()
                ):
                    raise ValueError("档位或证据编号无效")
                return dict(
                    name=item["name"],
                    score=item["points"] * level / 2,
                    why=r["why"],
                    citations=[relevant[n - 1] for n in refs],
                )
            except (ValueError, KeyError, TypeError) as exc:
                if attempt == 2:
                    raise
                data["校正"] = str(exc)

    with ThreadPoolExecutor(max_workers=3) as pool:
        return list(pool.map(judge, rubric["items"]))
