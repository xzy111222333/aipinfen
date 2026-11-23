import concurrent.futures
import json
import math
import os
import re
import urllib.error
import urllib.request
from http.client import IncompleteRead
from pathlib import Path
from .grid_sheet import measure_grid_sheet


MODEL_DEFAULT = "deepseek-flash"
API_URL = "https://api.deepseek.com/chat/completions"
RELAY_URL = "https://chyyds.com/v1/chat/completions"
RELAY_MODEL = "deepseek-v4.1-flash"
RELAY_KEY = "SHENLUN_CHYYDS_API_KEY"


def _read_api_setting(name):
    env_key = os.environ.get(name, "").strip()
    if env_key:
        return env_key
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(rf"^\s*{name}\s*=\s*(\S+)\s*$", line)
            if m:
                val = m.group(1).strip().strip('"').strip("'")
                if val:
                    return val
    return None


def _resolve_api_key(api_key=None):
    if api_key:
        return api_key
    val = _read_api_setting("DEEPSEEK_API_KEY")
    if val:
        os.environ["DEEPSEEK_API_KEY"] = val
        return val
    raise KeyError("未找到 DEEPSEEK_API_KEY，请在环境变量或项目根目录 .env 文件中配置")


JIJI_URL = "https://www.jiji.cc/v1/responses"
MDKJ_URL = "https://mdkj.lol/v1/chat/completions"
MDKJ_KEY = "SHENLUN_MDKJ_API_KEY"


def llm_cache_model(api_key=None):
    if api_key is None and _read_api_setting(MDKJ_KEY):
        return MDKJ_URL, RELAY_MODEL
    if api_key is None and _read_api_setting("SHENLUN_JIJI_API_KEY"):
        return JIJI_URL, RELAY_MODEL
    return (RELAY_URL, RELAY_MODEL) if api_key is None and _read_api_setting(RELAY_KEY) else MODEL_DEFAULT


def _read_relay_response(response):
    if response.headers.get_content_type() != "text/event-stream":
        return json.load(response)
    for line in response:
        if line.startswith(b"data:"):
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("type") in ("response.completed", "response.failed", "response.incomplete"):
                return event["response"]
    raise ConnectionError("Response stream ended before completion")


def call_llm_json(system_prompt, user_payload, api_key=None, model=None, temperature=0.0, max_tokens=16000):
    mdkj_key = _read_api_setting(MDKJ_KEY) if api_key is None else None
    relay_key = _read_api_setting(RELAY_KEY) if api_key is None else None
    jiji_key = (None if mdkj_key else _read_api_setting("SHENLUN_JIJI_API_KEY")) if api_key is None else None
    url = MDKJ_URL if mdkj_key else JIJI_URL if jiji_key else RELAY_URL if relay_key else API_URL
    relay_key = mdkj_key or jiji_key or relay_key
    key = relay_key or _resolve_api_key(api_key)
    model = model or (RELAY_MODEL if relay_key else MODEL_DEFAULT)
    headers = {"Content-Type": "application/json"}
    if relay_key:
        headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    last_err = None
    for retry in range(3):
        sys_msg = system_prompt if retry == 0 else (system_prompt + "\n请严格输出合法的RFC8259标准JSON，切勿包含尾随逗号或未转义引号。")
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if jiji_key:
            headers["x-openai-actor-authorization"] = "local-image-extension"
            body = dict(model=model, instructions=sys_msg, input=json.dumps(user_payload, ensure_ascii=False),
                        max_output_tokens=max_tokens, store=False, stream=True, temperature=temperature)
        req = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
        )
        req.add_unredirected_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = _read_relay_response(resp) if jiji_key else json.load(resp)
            if relay_key and raw.get("model") != model:
                raise RuntimeError(f"中转站返回的模型与请求不一致，请求模型：{model}")
            if jiji_key:
                if raw.get("status") != "completed":
                    raise ValueError(f"中转响应未完成: {raw.get('status')}")
                content = "".join(part["text"] for item in raw.get("output", []) if item.get("type")=="message"
                                  for part in item.get("content", []) if part.get("type")=="output_text")
                return json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
            choice = raw["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError(f"模型输出被截断: {choice['finish_reason']}")
            content = choice["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("模型响应缺少有效正文")
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I).strip()
            # 清理偶发的尾随逗号容错后再做严格 json.loads
            cleaned_json = re.sub(r",\s*([}\]])", r"\1", content)
            return json.loads(cleaned_json)
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace").replace(key, "[redacted]")[:400]
            if err.code in {429, 500, 502, 503, 504, 524} and retry < 2:
                last_err = RuntimeError(f"API HTTP {err.code}: {detail}")
                continue
            raise RuntimeError(f"API HTTP {err.code}: {detail}") from None
        except (ValueError, urllib.error.URLError, TimeoutError, ConnectionError, IncompleteRead) as err:
            last_err = err
            if retry < 2:
                continue
            raise
    raise last_err


def is_verbatim_quote(quote, source_text):
    """校验引文是否为原文的连续逐字子串（忽略标点与空白差异，但严禁改字、跳词拼接或省略号）。"""
    if not isinstance(quote, str) or not quote.strip():
        return False
    if "..." in quote or "……" in quote:
        if quote not in source_text:
            return False
    norm_q = re.sub(r"[^\w\u4e00-\u9fff]+", "", quote)
    norm_s = re.sub(r"[^\w\u4e00-\u9fff]+", "", source_text)
    return bool(norm_q) and (norm_q in norm_s)


def index_answer_evidence(text):
    return {
        str(number): (match.start(), match.end(), match.group())
        for number, match in enumerate(re.finditer(r"[^。！？；\n]+[。！？；]*", text), start=1)
        if match.group().strip()
    }


def resolve_answer_evidence(row, text, spans):
    identifiers = row.get("证据编号", [])
    if isinstance(identifiers, list) and identifiers:
        ranges = [spans.get(str(identifier)) for identifier in identifiers]
        if all(ranges):
            return text[min(span[0] for span in ranges):max(span[1] for span in ranges)].strip()
        return ""
    quote = row.get("答卷引文", "")
    return quote if is_verbatim_quote(quote, text) else ""


def count_exam_grids(text):
    """按申论方格纸规则计字：每个汉字或中文标点占1格，连续2个英文字符或数字占1格，忽略空白符。"""
    cleaned = re.sub(r"\s+", "", text)
    if not cleaned:
        return 0
    grids = 0
    ascii_run = 0
    for ch in cleaned:
        if ord(ch) < 128 and ch.isalnum():
            ascii_run += 1
        else:
            if ascii_run:
                grids += math.ceil(ascii_run / 2)
                ascii_run = 0
            grids += 1
    if ascii_run:
        grids += math.ceil(ascii_run / 2)
    return grids


def truncate_by_exam_grids(text, max_grids):
    """模拟物理答题卡边界：超出方格上限的尾部文字直接截断，不进入内容采分。"""
    if max_grids is None or max_grids <= 0:
        return text, ""
    grids = 0
    ascii_run = 0
    cut_idx = len(text)
    for idx, ch in enumerate(text):
        if ch.isspace():
            continue
        if ord(ch) < 128 and ch.isalnum():
            next_grids = grids + math.ceil((ascii_run + 1) / 2)
            if next_grids > max_grids:
                cut_idx = idx
                break
            ascii_run += 1
        else:
            if ascii_run:
                grids += math.ceil(ascii_run / 2)
                ascii_run = 0
            if grids + 1 > max_grids:
                cut_idx = idx
                break
            grids += 1
    return text[:cut_idx], text[cut_idx:]


def clean_reference_body(text):
    """剥离题库或考生整理稿中混入的非作答元注释、字数统计尾注与行内【讲评】批注。"""
    raw_lines = (text or "").splitlines()
    filtered = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            filtered.append("")
            continue
        if stripped in {"### 参考答案", "【参考范文】", "参考范文", "参考答案"}:
            continue
        if stripped.startswith(("参考答案说明：", "参考答案说明:", "答题思路：", "答题思路:")):
            break
        # 剔除整理版在正文前附带的考后复盘说明行
        if re.match(r"^\d+\s*分钟给大作文", stripped):
            continue
        line = re.sub(r"[（(]\s*(?:正文部分)?(?:共计?|计空格共?|计空格)\s*\d+\s*字[^）)]*[）)]", "", line)
        line = re.sub(r"[（(]\s*\d+\s*字[,，]?\s*考场[^）)]*[）)]", "", line)
        # 剔除行内【12】计数标记或【比喻论证】【提出总论点】等整理批注（不误伤《》）
        line = re.sub(r"[\[【](?:\d+|[^\]】]{2,18}(?:论证|论点|背景|意义|紧扣资料|具体展开))[\]】]", "", line)
        line = re.sub(r"^(?:标题|结尾)[：:]\s*", "", line.strip()) if len(stripped) > 6 else line
        if line.strip():
            filtered.append(line)

    # 若正文前附有“总论点/分论点”提纲且紧接着重复出现独立标题与完整正文，剥离前置提纲避免重复计字
    for idx in range(1, min(8, len(filtered) - 2)):
        head_blob = "\n".join(filtered[:idx])
        if ("总论点：" in head_blob or "分论点一：" in head_blob) and len(filtered[idx].strip()) <= 24:
            if filtered[idx].strip() == filtered[0].strip():
                filtered = filtered[idx:]
                break

    return "\n".join(filtered).strip()


def is_authentic_human_or_scored_source(source_name):
    """判断某条题库参考答案是否属于真人实战回忆卷或带分考生卷（建表时必须剥离，严防测试集泄漏）。"""
    s = (source_name or "").strip()
    if not s:
        return False
    # 拦截裸分值（如 ccm 70.5 / ccm70.5 / 蓝鲸镇少年与海 72）
    if re.search(r'(?:\s|^|[a-zA-Z\u4e00-\u9fff])[6-9]\d(?:\.\d)?(?:\s|$)', s):
        return True
    patterns = [
        r'\d+(?:\.\d+)?\s*分',
        r'实战',
        r'xhs',
        r'小红书',
        r'网友',
        r'考生',
        r'选手',
        r'大神',
        r'回忆',
        r'上岸',
        r'未署名',
        r'b站高分',
        r'高资',
        r'逆袭工厂',
        r'公考苏老师',
        r'田心小钱',
        r'还得冰美式',
        r'hilunaaa',
        r'大厂考公',
        r'蓝鲸镇',
        r'yuzu',
    ]
    return any(re.search(pat, s, re.I) for pat in patterns)


def load_question_bundle(question_md_path, include_all_materials=False, exclude_human_scored=False, exclude_sources=None):
    """从题库 Markdown 读取题干、关联材料段落与参考答案；支持全卷材料加载与防泄漏过滤。"""
    q_path = Path(question_md_path)
    raw = q_path.read_text(encoding="utf-8-sig")
    folder = q_path.parent

    ref_section = raw.split("## 题目", 1)[0]
    mat_links = re.findall(r"\[(材料\s*\d+)\]\(([^)]+)\)", ref_section)
    if include_all_materials:
        all_mat_files = sorted(
            folder.glob("材料*.md"),
            key=lambda fp: int(re.search(r"\d+", fp.stem).group()) if re.search(r"\d+", fp.stem) else 999,
        )
        if all_mat_files:
            mat_links = [(fp.stem, fp.name) for fp in all_mat_files]

    question_block = raw.split("## 题目", 1)[1].split("## 答案", 1)[0].strip()

    specified_nums = set()
    if not include_all_materials:
        # First check for explicit material references
        for m in re.finditer(r"(?:给定)?(?:资料|材料)\s*(\d+(?:\s*[-—~至到、，,和与及]\s*\d+)*)", question_block):
            expr = m.group(1)
            for rng in re.finditer(r"(\d+)\s*[-—~至到]\s*(\d+)", expr):
                specified_nums.update(range(int(rng.group(1)), int(rng.group(2)) + 1))
            cleaned_expr = re.sub(r"\d+\s*[-—~至到]\s*\d+", " ", expr)
            for single in re.findall(r"\d+", cleaned_expr):
                specified_nums.add(int(single))

        # Entity-based narrowing for non-essay questions without explicit references
        if not specified_nums:
            entities = re.findall(r"([\u4e00-\u9fff]{2,4}[村县市公司|区|夜市|活动])", question_block)
            if entities:
                for mat_file in folder.glob("材料*.md"):
                    content = mat_file.read_text(encoding="utf-8-sig")
                    if any(e in content for e in entities):
                        mat_num = int(re.search(r"\d+", mat_file.stem).group())
                        specified_nums.add(mat_num)

    paragraphs = []
    seq = 1
    for label, rel_file in mat_links:
        mat_num = int(re.search(r"\d+", label).group())
        if specified_nums and mat_num not in specified_nums:
            continue
        mat_raw = (folder / rel_file).read_text(encoding="utf-8-sig")
        for idx, block in enumerate(
            (b.strip() for b in re.split(r"\n[ \t]*\n", mat_raw) if b.strip() and not b.lstrip().startswith("# ")),
            start=1,
        ):
            paragraphs.append({
                "全局段号": seq,
                "材料名": f"资料{mat_num}",
                "段号": idx,
                "原文": block,
            })
            seq += 1

    excl_keywords = [k.strip().lower() for k in (exclude_sources or []) if k and k.strip()]
    answers = []
    if "## 答案" in raw:
        ans_part = raw.split("## 答案", 1)[1]
        chunks = re.split(r"\n### ", ans_part)
        for chunk in chunks[1:]:
            if "\n" not in chunk:
                continue
            title_line, body = chunk.split("\n", 1)
            src_name = title_line.strip()
            if exclude_human_scored and is_authentic_human_or_scored_source(src_name):
                continue
            if excl_keywords and any(kw in src_name.lower() for kw in excl_keywords):
                continue
            cleaned = clean_reference_body(body)
            if cleaned:
                answers.append({
                    "来源": src_name,
                    "正文": cleaned,
                    "格数": count_exam_grids(cleaned),
                })

    return {
        "题目路径": str(q_path),
        "题干与要求": question_block,
        "材料段落": paragraphs,
        "参考答案": answers,
    }


def parse_hard_constraints(question_text):
    score_match = re.search(r"[（(]\s*(\d+(?:\.\d+)?)\s*分\s*[）)]", question_text)
    if not score_match:
        raise ValueError(f"题干未找到明确分值: {question_text}")
    full_score = float(score_match.group(1))

    range_match = re.search(r"(\d+)\s*[-—–~～至]\s*(\d+)\s*字", question_text)
    explicit_max = re.search(r"(?:不超过|不超|限|控制在|少于)\s*(\d+)\s*字|(\d+)\s*字\s*(?:以内|以下)", question_text)
    fallback_max = re.search(r"(\d+)\s*字(?:左右)?", question_text)
    min_words = None
    max_words = None
    if range_match:
        min_words = int(range_match.group(1))
        max_words = int(range_match.group(2))
    elif explicit_max:
        max_words = int(explicit_max.group(1) or explicit_max.group(2))
    elif fallback_max:
        max_words = int(fallback_max.group(1))

    # 物理答题卡通常多留1行（约6%，最少20格，最多30格）作为缓冲，超出缓冲格的部分物理截断
    buffer_grids = min(30, max(20, round(max_words * 0.06))) if max_words else None
    hard_cutoff = (max_words + buffer_grids) if max_words else None

    return {
        "满分": full_score,
        "最少字数": min_words,
        "字数上限": max_words,
        "物理截断格数": hard_cutoff,
    }


SPEC_PROMPT = """你只负责把申论题干与要求解析成结构化槽位，不得编造采分点或答案。
严格按以下枚举判断：
1. 题型类别：只能选 "归纳概括"、"分类归纳"、"原因对策"、"公文应用文"、"综合理解" 之一。
   - 若题干要求写发布词、倡议书、讲话稿、公开信、通知、简报、短评、方案等文种，选 "公文应用文"。
   - 若题干明确给出固定分类维度（如分别谈经济/社会/生态效益），选 "分类归纳"。
   - 若同时要求写原因/问题并提建议/对策，选 "原因对策"。
2. 文种子类：仅当题型类别为 "公文应用文" 时填写，只能选：
   - "书信倡议通知类"（如公开信、倡议书、通知、感谢信，通常需标题、称呼、落款日期）
   - "讲话发言发布词类"（如发布词、讲话稿、发言稿、宣讲稿、开幕词，重在标题/开场宣告与现场受众感，落款日期可写可不写）
   - "提纲简报方案类"（如调研提纲、工作简报、活动方案、编者按，无须落款称呼）
   非公文题填 null。
3. 发文身份：题干指定的机构或角色名称（如 "S市科委工作人员"、"各级政府"），未指定填 null。
4. 目标受众：公文面对的对象（如 "社会公众/广大市民"），非公文填 null。
5. 核心答题目标：用一句话概括题目问的核心对象与动作（如 "各级政府在稳就业方面采取的主要措施"）。
6. 是否要求号召动员：布尔值。仅当题干或要求明确含 "呼吁"、"号召"、"倡议"、"动员"、"欢迎报名/参与" 时为 true。
7. 是否纯问做法措施：布尔值。仅当题干明确只问“如何做/主要措施/具体做法/工作建议”（不问成效、背景或全面介绍）时为 true。
8. 强制归类维度：仅当题型要求按指定大类分块作答时列出大类名数组（如 ["经济效益","社会效益","生态效益"] 或 ["原因","建议"]），否则填空数组 []。
9. 受众错位陷阱提示：若题干有明确目标受众，结合题干说明哪类对象不能与目标受众混淆（不看答案，仅按角色常理一句话提示，无则填 null）。

返回 JSON：
{
  "题型类别": "...",
  "文种子类": null,
  "发文身份": "...",
  "目标受众": null,
  "核心答题目标": "...",
  "是否要求号召动员": false,
  "是否纯问做法措施": false,
  "强制归类维度": [],
  "受众错位陷阱提示": null
}"""


CARD_EXTRACT_PROMPT = """你是申论材料原子信息拆解器。只允许做机械拆句和指代还原，严禁筛选重点、严禁给分、严禁编造。
输入包含【题干信息】与【给定材料段落】。请逐段把材料拆解成最小原子事实卡片，遵守以下死规则：
1. 把代词（我们、他、该公司、此模式等）还原为材料里的真实名称。
2. 每张卡片的 "原文" 字段必须逐字复制对应 "全局段号" 原文中的连续子串，一字不差，不得用省略号。
3. "信息类型" 只能从以下6项中选1项：
   - "目的用意"：政策/项目明示的宗旨、初衷、预期目标
   - "具体做法"：某主体实施、筹备、规定或计划采取的具体行动、流程步骤、渠道方式
   - "机制特点"：模式定义、覆盖范围、门槛特征、核心优势
   - "成效结果"：已经产生的客观效果、改变或荣誉
   - "个体案例与体验"：具体人物个体感受或单一微观故事例子
   - "纯背景与纯算账数字"：引出话题的困境背景、纯企业内部无关动作、或孤立统计数字铺垫
4. "面向或惠及对象"：该信息主要针对谁（如 "社会公众"、"申报课题的科学家"、"滞留贫困劳动力"、"高校毕业生"、"在岗职工与企业" 等）。
5. "题干相关性"：仅做客观主语和方向核对，只能选：
   - "直接相关"：符合题干询问的主体范围或公文所需介绍/呼吁的项目内容
   - "受众或主体易混"：与材料主题有关，但执行主体非题干所问（如问政府措施却写企业自发远程办公/高管捐款），或针对对象与公文受众不同（如面向公众呼吁报名，材料该句却是支持科学家申报课题拿经费）
   - "背景铺垫"：纯引子问题或纯算账细节
6. "是否具体案例例证"：布尔值。若该卡片只是为了举例说明而引用的某一个往期具体案例细节、某个人物的微观感受、或某次统计的数字例证（如“以XX项目为例…”、“10岁豆豆觉得…”、“80多台红外相机…”、“全国已运送超80万人”），填 true；若属于通用项目宗旨、整体机制定义、通用操作流程、本次核心发布计划或普适政策措施，填 false。
7. 把讲同一业务环节或同一类作用的卡片归入同一个简短的 "所属簇名"（整篇材料控制在 4~7 个功能簇；建议把具体人物/往期案例例证集中归入案例体验簇，与主干机制/计划簇分开）。
8. 保持全卷卡片总数在 12~26 张之间，既不遗漏独立信息，也不把一句话拆成碎词。

返回 JSON：
{
  "功能簇列表": [{"簇编号": "G1", "簇名称": "...", "簇说明": "..."}],
  "卡片": [
    {
      "卡号": "M01",
      "所属簇": "G1",
      "全局段号": 1,
      "主体": "...",
      "面向或惠及对象": "...",
      "信息类型": "...",
      "题干相关性": "直接相关",
      "是否具体案例例证": false,
      "核心语义": "25字以内的客观事实陈述",
      "原文": "逐字连续引文"
    }
  ]
}"""


def extract_verified_cards(spec, paragraphs, api_key=None):
    para_map = {p["全局段号"]: p["原文"] for p in paragraphs}
    payload = {"题干解析": spec, "给定材料段落": paragraphs}
    prompt = CARD_EXTRACT_PROMPT
    for attempt in range(3):
        res = call_llm_json(prompt, payload, api_key=api_key, temperature=0.0)
        clusters = res.get("功能簇列表", [])
        cards = res.get("卡片", [])
        cluster_ids = {c["簇编号"] for c in clusters if c.get("簇编号") and c.get("簇名称")}
        errors = []
        if not (3 <= len(clusters) <= 9):
            errors.append(f"功能簇数量异常({len(clusters)})，应在3~9之间")
        if not (8 <= len(cards) <= 32):
            errors.append(f"卡片数量异常({len(cards)})，应在8~32之间")
        seen_ids = set()
        for c in cards:
            cid = c.get("卡号", "")
            pno = c.get("全局段号")
            quote = c.get("原文", "")
            if not cid or cid in seen_ids:
                errors.append(f"卡号缺失或重复: {cid}")
            seen_ids.add(cid)
            if c.get("所属簇") not in cluster_ids:
                errors.append(f"{cid} 所属簇 {c.get('所属簇')} 不在功能簇列表中")
            if pno not in para_map or not is_verbatim_quote(quote, para_map[pno]):
                errors.append(f"{cid} 引文不在全局第{pno}段原文中: {quote[:60]}")
            if c.get("题干相关性") not in {"直接相关", "受众或主体易混", "背景铺垫"}:
                errors.append(f"{cid} 题干相关性枚举非法")
        if not errors:
            return {"功能簇列表": clusters, "卡片": cards}
        prompt = CARD_EXTRACT_PROMPT + "\n上轮输出未通过校验，请修正以下问题后重新输出完整JSON：" + "；".join(errors[:10])
    raise RuntimeError("材料原子卡片提取连续3次未通过逐字校验")


ALIGN_ANSWERS_PROMPT = """你是参考答案与材料卡片的机械连线员。严禁自己评价答案好坏，只做客观连线。
输入包含【材料原子卡片列表】和一批【参考答案】。对每份参考答案：
1. 检查它命中了哪些材料卡片：
   - "完整"：明确写出了该卡片的核心动作/事实/概念（允许同义词或换说法）
   - "笼统"：只用高度概括的泛词一笔带过（例如材料是具体三项处置措施，答案只写“依法处理”4个字；或材料是具体培训形式，答案只写“学习技能”）
   未提到的卡片不要列入，节省篇幅。
2. 每条命中必须附带 "答案引文"，逐字复制该份参考答案中的连续原句片段，一字不差！
3. 记录该参考答案中没有任何材料卡片支撑的纯自造套话短语到 "无依据包装词"（最多列3条典型短语，无则空数组）。
4. 若为公文题，额外如实勾选客观格式与受众事实：
   - "有独立标题"：布尔值
   - "有独立称呼或开篇直呼受众"：布尔值
   - "有落款单位"：布尔值
   - "向目标受众发出明确号召"：布尔值
   - "受众错位情况"：只能选 "无"、"顺带提及非受众事项"（如面向市民发布词里顺带一句欢迎科学家申报立项，但仍重点呼吁市民报名）、"严重错位"（如把给科学家课题的专项经费说成市民报名即可领经费，或结尾只呼吁科学家申报而未呼吁公众报名）。

返回 JSON：
{
  "对齐结果": [
    {
      "答案编号": 0,
      "命中列表": [{"卡号": "M01", "程度": "完整", "答案引文": "..."}],
      "无依据包装词": [],
      "公文检查": {
        "有独立标题": true,
        "有独立称呼或开篇直呼受众": true,
        "有落款单位": true,
        "向目标受众发出明确号召": true,
        "受众错位情况": "无"
      }
    }
  ]
}"""


def align_reference_answers(cards, ref_answers, is_gongwen, api_key=None, batch_size=3):
    card_ids = {c["卡号"] for c in cards}
    compact_cards = [
        {
            "卡号": c["卡号"],
            "所属簇": c["所属簇"],
            "主体": c["主体"],
            "面向或惠及对象": c["面向或惠及对象"],
            "信息类型": c["信息类型"],
            "题干相关性": c["题干相关性"],
            "核心语义": c["核心语义"],
            "原文": c["原文"],
        }
        for c in cards
    ]

    batches = []
    for start in range(0, len(ref_answers), batch_size):
        chunk = [
            {"答案编号": idx, "正文": ref_answers[idx]["正文"]}
            for idx in range(start, min(start + batch_size, len(ref_answers)))
        ]
        batches.append(chunk)

    def run_batch(chunk):
        ans_map = {item["答案编号"]: item["正文"] for item in chunk}
        payload = {"是否公文题": is_gongwen, "材料卡片": compact_cards, "参考答案批次": chunk}
        prompt = ALIGN_ANSWERS_PROMPT
        for _ in range(3):
            res = call_llm_json(prompt, payload, api_key=api_key, temperature=0.0)
            rows = res.get("对齐结果", [])
            errors = []
            if sorted(r.get("答案编号") for r in rows) != sorted(ans_map.keys()):
                errors.append("返回的答案编号与输入批次不一致")
            for r in rows:
                aid = r.get("答案编号")
                body = ans_map.get(aid, "")
                for hit in r.get("命中列表", []):
                    cid = hit.get("卡号")
                    deg = hit.get("程度")
                    q = hit.get("答案引文", "")
                    if cid not in card_ids:
                        errors.append(f"答案{aid} 引用了不存在的卡号 {cid}")
                    if deg not in {"完整", "笼统"}:
                        errors.append(f"答案{aid} {cid} 程度非法: {deg}")
                    if not is_verbatim_quote(q, body):
                        errors.append(f"答案{aid} {cid} 引文非连续原句: {q[:40]}")
                if is_gongwen:
                    gw = r.get("公文检查") or {}
                    if gw.get("受众错位情况") not in {"无", "顺带提及非受众事项", "严重错位"}:
                        errors.append(f"答案{aid} 公文受众错位枚举非法")
            if not errors:
                return rows
            print(f"    [连线重试] 批次 {[x['答案编号'] for x in chunk]} 校验未过: {errors[:3]}", flush=True)
            prompt = ALIGN_ANSWERS_PROMPT + "\n上轮校验失败（注意引文严禁用省略号跳字，必须连续逐字复制该答案原句），请修正：" + "；".join(errors[:10])
        raise RuntimeError(f"参考答案连线连续3次校验失败: {errors[:5]}")

    aligned = []
    if batches:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
            for part in pool.map(run_batch, batches):
                aligned.extend(part)
    aligned.sort(key=lambda x: x["答案编号"])
    return aligned


def compile_quantitative_rubric(bundle, spec, constraints, card_pack, aligned_refs, mode_label):
    """纯 Python 数学编译器：根据材料结构、字数预算与参考答案对齐统计生成锁死评分卡。"""
    full_score = constraints["满分"]
    word_limit = constraints["字数上限"] or 400
    is_gongwen = spec["题型类别"] == "公文应用文"
    has_categories = bool(spec["强制归类维度"])

    if is_gongwen:
        format_pool = round(full_score * 0.10, 2)
        pragmatic_pool = round(full_score * 0.15, 2) if spec["是否要求号召动员"] else round(full_score * 0.10, 2)
        content_pool = round(full_score - format_pool - pragmatic_pool, 2)
        structure_pool = 0.0
    elif has_categories:
        format_pool = 0.0
        pragmatic_pool = 0.0
        structure_pool = round(full_score * 0.10, 2)
        content_pool = round(full_score - structure_pool, 2)
    else:
        format_pool = 0.0
        pragmatic_pool = 0.0
        structure_pool = 0.0
        content_pool = full_score

    n_refs = len(aligned_refs)
    cards = card_pack["卡片"]
    clusters = card_pack["功能簇列表"]

    # 统计每张卡的完整命中率、总命中率、同义表达词库，以及单答案模式下的“1对多缩水”现象
    card_stats = {
        c["卡号"]: {
            "full_hits": 0,
            "vague_hits": 0,
            "quotes": [],
            "overcompressed_in_single": False,
        }
        for c in cards
    }

    for row in aligned_refs:
        quote_to_cards = {}
        for hit in row.get("命中列表", []):
            cid = hit["卡号"]
            q = hit["答案引文"].strip()
            if hit["程度"] == "完整":
                card_stats[cid]["full_hits"] += 1
            else:
                card_stats[cid]["vague_hits"] += 1
            if q and q not in card_stats[cid]["quotes"] and len(card_stats[cid]["quotes"]) < 6:
                card_stats[cid]["quotes"].append(q)
            quote_to_cards.setdefault(q, []).append((cid, hit["程度"]))

        if n_refs <= 2:
            for q, mapped in quote_to_cards.items():
                # 抓出某笔式偷懒缩水：同一句短泛词笼统连上了>=2张独立做法卡（如“依法处理”连3张卡）
                vague_cids = [cid for cid, deg in mapped if deg == "笼统"]
                if len(vague_cids) >= 2 and count_exam_grids(q) <= 22:
                    for cid in vague_cids:
                        card_stats[cid]["overcompressed_in_single"] = True

    # 计算每张卡片的原始权重份额与分值（严格遵守字数容量守恒，不依赖AI主观给分）
    only_measures = bool(spec.get("是否纯问做法措施"))
    raw_weights = {}
    card_role = {}
    for c in cards:
        cid = c["卡号"]
        st = card_stats[cid]
        total_hits = st["full_hits"] + st["vague_hits"]
        rel = c["题干相关性"]
        ctype = c["信息类型"]
        is_case = bool(c.get("是否具体案例例证")) or ctype == "个体案例与体验"

        if n_refs >= 5:
            # 多机构共识模式：用客观投票率 + 材料相关性双重过滤
            freq = (st["full_hits"] + 0.5 * st["vague_hits"]) / n_refs
            any_freq = total_hits / n_refs
            if rel != "直接相关" and freq < 0.35:
                raw_weights[cid] = 0.0
                card_role[cid] = "非采分项(背景或易混对象)"
            elif any_freq < 0.12:
                raw_weights[cid] = 0.0
                card_role[cid] = "免考边缘细节(<12%提及)"
            elif freq >= 0.65:
                raw_weights[cid] = 2.2
                card_role[cid] = "核心共识必答点"
            elif freq >= 0.35:
                raw_weights[cid] = 1.4
                card_role[cid] = "高频有效得分点"
            else:
                raw_weights[cid] = 0.9
                card_role[cid] = "可选支撑/案例等价点"
        else:
            # 单答案/零答案纠偏模式：用“主干容量归一化 + 案例弹性外挂”防止大量故事细节稀释主干分值
            if rel != "直接相关":
                raw_weights[cid] = 0.0
                card_role[cid] = "非采分项(非指定主体或受众易混)"
            elif ctype == "纯背景与纯算账数字":
                raw_weights[cid] = 0.0
                card_role[cid] = "非采分项(纯背景数字)"
            elif only_measures and ctype == "成效结果" and total_hits == 0:
                raw_weights[cid] = 0.0
                card_role[cid] = "非采分项(纯结果统计，非做法措施)"
            elif is_case:
                raw_weights[cid] = 0.7 if total_hits > 0 else 0.5
                card_role[cid] = "案例等价选答点(多选一封顶，不稀释主干分)"
            elif st["overcompressed_in_single"]:
                raw_weights[cid] = 1.8
                card_role[cid] = "原文反缩水还原点(纠偏参考答案笼统表述)"
            elif st["full_hits"] > 0:
                raw_weights[cid] = 2.0
                card_role[cid] = "原文与参考答案双印证主干点"
            elif st["vague_hits"] > 0:
                raw_weights[cid] = 1.7
                card_role[cid] = "原文具体还原主干点"
            else:
                raw_weights[cid] = 1.4
                card_role[cid] = "材料主干补漏点(参考答案遗漏)"

    scored_cards = []
    if n_refs >= 5:
        elastic_target_sum = content_pool * 1.22
        total_raw = sum(raw_weights.values())
        if total_raw <= 0:
            raise RuntimeError("有效采分卡总权重为0")
        scale = elastic_target_sum / total_raw
        for c in cards:
            cid = c["卡号"]
            w = round(raw_weights[cid] * scale * 4) / 4
            if raw_weights[cid] > 0 and w < 0.5:
                w = 0.5
            st = card_stats[cid]
            scored_cards.append({
                **c,
                "计分角色": card_role[cid],
                "满分值": w,
                "笼统得分系数": 0.5,
                "参考命中统计": f"{st['full_hits']}完整/{st['vague_hits']}笼统 (共{n_refs}份)",
                "常见同义表述": st["quotes"],
            })
    else:
        # 单答案容量守恒定标：仅主干卡即支撑 112% 内容分池，微观案例卡作为额外 0.5~0.75 分等价补充
        backbone_raw = sum(
            raw_weights[c["卡号"]]
            for c in cards
            if raw_weights[c["卡号"]] > 0 and not (bool(c.get("是否具体案例例证")) or c["信息类型"] == "个体案例与体验")
        )
        if backbone_raw <= 0:
            backbone_raw = sum(raw_weights.values())
        if backbone_raw <= 0:
            raise RuntimeError("有效采分卡总权重为0")
        backbone_scale = (content_pool * 1.12) / backbone_raw
        for c in cards:
            cid = c["卡号"]
            is_case = bool(c.get("是否具体案例例证")) or c["信息类型"] == "个体案例与体验"
            if raw_weights[cid] <= 0:
                w = 0.0
            elif is_case:
                w = 0.75 if card_stats[cid]["full_hits"] > 0 else 0.5
            else:
                w = round(raw_weights[cid] * backbone_scale * 4) / 4
                if w < 0.5:
                    w = 0.5
            st = card_stats[cid]
            scored_cards.append({
                **c,
                "计分角色": card_role[cid],
                "满分值": w,
                "笼统得分系数": 0.4 if st["overcompressed_in_single"] else 0.5,
                "参考命中统计": f"{st['full_hits']}完整/{st['vague_hits']}笼统 (共{n_refs}份)",
                "常见同义表述": st["quotes"],
            })

    # 计算每个功能簇的封顶值：若某簇主要由具体案例组成，严格收紧簇上限，防止狂抄故事刷分
    cluster_rules = []
    for cl in clusters:
        gid = cl["簇编号"]
        g_cards = [sc for sc in scored_cards if sc["所属簇"] == gid and sc["满分值"] > 0]
        raw_sum = round(sum(sc["满分值"] for sc in g_cards), 2)
        if raw_sum == 0:
            cap = 0.0
        else:
            case_cnt = sum(
                1 for sc in g_cards
                if bool(sc.get("是否具体案例例证")) or sc["信息类型"] == "个体案例与体验"
            )
            case_ratio = case_cnt / len(g_cards)
            if case_ratio >= 0.6 and len(g_cards) >= 2:
                cap = min(
                    round(max(0.5, raw_sum * 0.65) * 2) / 2,
                    round(max(1.0, content_pool * 0.14) * 2) / 2,
                )
            elif n_refs >= 5:
                valid_ids = {sc["卡号"] for sc in g_cards}
                cluster_hit_refs = sum(
                    1 for r in aligned_refs
                    if any(h["卡号"] in valid_ids for h in r.get("命中列表", []))
                )
                cluster_freq = cluster_hit_refs / n_refs
                factor = 0.88 if cluster_freq >= 0.85 else 0.78
                cap = round(min(raw_sum, max(1.5, raw_sum * factor)) * 2) / 2
            else:
                cap = round(min(raw_sum, max(1.5, raw_sum * 0.90)) * 2) / 2
            cap = min(raw_sum, cap)
        cluster_rules.append({
            "簇编号": gid,
            "簇名称": cl["簇名称"],
            "簇说明": cl["簇说明"],
            "簇内卡片总分": raw_sum,
            "簇封顶分": cap,
        })

    # 汇总参考答案中被识别出的无依据包装词（供展示透明性）
    fluff_counter = {}
    for r in aligned_refs:
        for phrase in r.get("无依据包装词", []):
            p = phrase.strip()
            if p:
                fluff_counter[p] = fluff_counter.get(p, 0) + 1

    # identify incentive/value cards for 公文 "有号召力" 客观联动计分
    incentive_card_ids = [
        sc["卡号"] for sc in scored_cards
        if sc["满分值"] > 0 and sc["信息类型"] in {"机制特点", "个体案例与体验", "成效结果"}
    ]

    return {
        "预制模式": mode_label,
        "题目路径": bundle["题目路径"],
        "题干与要求": bundle["题干与要求"],
        "硬约束": constraints,
        "题干槽位": spec,
        "分池配置": {
            "内容分池上限": content_pool,
            "文体格式分池": format_pool,
            "场景号召与受众分池": pragmatic_pool,
            "分类条理分池": structure_pool,
            "卡片弹性总和": round(sum(sc["满分值"] for sc in scored_cards), 2),
            "簇封顶总和": round(sum(cr["簇封顶分"] for cr in cluster_rules), 2),
        },
        "字数与扣分死规则": {
            "标准上限格数": word_limit,
            "物理截断格数": constraints["物理截断格数"],
            "超字数规则": f"预留5格行尾标点容差；总格数超过{word_limit + 5}格起，每超25格扣1.0分，最多扣{round(full_score * 0.15, 1)}分；超过{constraints['物理截断格数']}格的溢出文字物理截断不予采分。",
        },
        "公文与号召力客观细则": (
            {
                "文种子类": spec["文种子类"],
                "格式计分项": (
                    "书信倡议通知类：独立标题(0.5)+顶格称呼(0.5)+落款单位与日期(1.0)=2.0分"
                    if spec["文种子类"] == "书信倡议通知类"
                    else "讲话发言发布词类：明确标题或首句发布宣告(1.0)+面向受众称呼语或首段直呼受众(1.0)=2.0分；落款日期兼容不强制扣分，但写错发文单位或年份扣0.5分"
                ),
                "号召力客观三要素": {
                    "明确发出受众行动呼吁": 1.0,
                    "客观参与价值支撑(由内容卡自动联动，命中>=2张特点/收获/案例卡得1.0分，1张得0.5分)": 1.0,
                    "受众与身份不错位红线(无错位得1.0分，顺带混提扣0.5分，把非受众政策张冠李戴给受众或只呼吁错对象得0分)": 1.0,
                    "联动激励卡号池": incentive_card_ids,
                },
            }
            if is_gongwen else None
        ),
        "功能簇封顶表": cluster_rules,
        "采分卡片表": scored_cards,
        "已剔除的参考答案无依据套话样本": sorted(fluff_counter.keys())[:12],
    }


GRADE_CHECKLIST_PROMPT = """你是严格按锁定清单核对答卷证据的检核员。你看不到分值，严禁打分或评价文采。
输入包含【核对卡片清单】、【题干槽位】和【考生可见答卷】（已按物理答题卡截断）。
请逐项完成客观事实核对：
1. 对每张核对卡片，判断考生答卷是否表达了其核心意思（完全允许大白话、同义词、换句式、跨段合并）：
   - "完整"：清楚表达了该卡核心动作/事实/概念
   - "笼统"：仅用宽泛词带过，未写出核心具体内容
   - "冲突"：与卡片事实相反（如把“非临港”写成“临港”，把“2026年发布”写成“2024年发布”，或把政府动作安错主体）
   - "未写"：未提及（为减少输出体积，状态为 "未写" 的卡片可省略不输出，程序自动记为未写）
   凡状态为 "完整"、"笼统"、"冲突" 的卡片，必须提供 "答卷引文"，逐字复制【考生可见答卷】中的连续原句片段，一字不差！
2. 若为公文题，填写 "公文客观核对"：
   - "有标题或首句明确发布宣告"：布尔值
   - "有独立标题"：布尔值
   - "有受众称呼或首段直接面向受众说话"：布尔值
   - "有落款单位与日期"：布尔值
   - "落款单位或年份是否写错"：布尔值（没写落款不算错，写了但写错单位或错年份才为true）
   - "明确呼吁目标受众报名或行动"：布尔值 + "呼吁引文"（逐字复制答卷中呼吁目标受众行动的原句；若只呼吁了非目标受众如科学家申报课题而未呼吁市民公众报名，必须填 false 且填空串）
   - "受众错位情况"：只能选：
     * "无"（全文核心面向目标受众；客观介绍“项目获政府专项经费支持”作为背景保障不算错位）
     * "顺带提及非受众事项"（文中专门分出祈使/欢迎句招揽非目标受众，如“欢迎各领域科学家积极申报立项”，但同时也呼吁了目标受众）
     * "严重错位"（将非受众政策张冠李戴给目标受众，如对市民说“诚邀大家申报，立项即可获专项经费”；或全文结尾只呼吁非目标受众而未呼吁目标受众）
     若非"无"须填 "错位引文"（逐字复制答卷原句）。
3. 若题干有强制归类维度（非空数组），填写 "归类核对"：
   - "是否分条或分块作答"：布尔值
   - "维度齐全且未严重混类"：布尔值
   - "错归类条数"：整数（把明显属于A类的内容塞进B类标题下的数量，无则填0）。

返回 JSON：
{
  "卡片核对": [{"卡号": "M01", "状态": "完整", "答卷引文": "..."}],
  "公文客观核对": null,
  "归类核对": null
}"""


ARBITRATE_PROMPT = """同一份考生答卷在两次独立检核中出现了局部判定分歧。
请严格比对【卡片原文与核心语义】、【题干槽位】和【考生可见答卷】：
1. 对每张分歧卡片做出最终客观裁定（只能选 "完整"、"笼统"、"冲突"、"未写"），非"未写"必须逐字复制【考生可见答卷】中的连续子串作为 "答卷引文"。
2. 若输入含 "公文核对分歧"，请对照答卷原文给出最终准确的 "公文客观核对" 对象（字段结构与检核A一致，引文须逐字在答卷中）。
返回 JSON：
{"裁定": [{"卡号": "M01", "状态": "完整", "答卷引文": "..."}], "公文客观核对": null}"""


def verify_checklist_pass(rubric, visible_text, shuffle_reverse=False, api_key=None):
    active_cards = [
        {
            "卡号": sc["卡号"],
            "主体": sc["主体"],
            "面向或惠及对象": sc["面向或惠及对象"],
            "信息类型": sc["信息类型"],
            "核心语义": sc["核心语义"],
            "材料原文锚点": sc["原文"],
            "常见同义说法参考": sc["常见同义表述"][:4],
        }
        for sc in rubric["采分卡片表"]
        if sc["满分值"] > 0
    ]
    if shuffle_reverse:
        active_cards = list(reversed(active_cards))

    valid_ids = {c["卡号"] for c in active_cards}
    spec = rubric["题干槽位"]
    is_gongwen = spec["题型类别"] == "公文应用文"
    has_cats = bool(spec["强制归类维度"])

    payload = {
        "题干槽位": spec,
        "核对卡片清单": active_cards,
        "考生可见答卷": visible_text,
    }
    prompt = GRADE_CHECKLIST_PROMPT
    for _ in range(3):
        res = call_llm_json(prompt, payload, api_key=api_key, temperature=0.0, max_tokens=8000)
        errors = []
        seen = set()
        for row in res.get("卡片核对", []):
            cid = row.get("卡号")
            st = row.get("状态")
            q = row.get("答卷引文", "")
            if cid not in valid_ids or cid in seen:
                errors.append(f"卡号非法或重复: {cid}")
            seen.add(cid)
            if st not in {"完整", "笼统", "冲突", "未写"}:
                errors.append(f"{cid} 状态非法: {st}")
            if st in {"完整", "笼统", "冲突"} and not is_verbatim_quote(q, visible_text):
                errors.append(f"{cid} 答卷引文不在考生答卷中: {q[:40]}")
        if is_gongwen:
            gw = res.get("公文客观核对")
            if not isinstance(gw, dict):
                errors.append("公文题必须返回 公文客观核对 对象")
            else:
                if gw.get("受众错位情况") not in {"无", "顺带提及非受众事项", "严重错位"}:
                    errors.append("受众错位情况枚举非法")
                cta_q = gw.get("呼吁引文", "")
                if gw.get("明确呼吁目标受众报名或行动") and not is_verbatim_quote(cta_q, visible_text):
                    errors.append(f"呼吁引文不在考生答卷中: {cta_q[:40]}")
                err_q = gw.get("错位引文", "")
                if gw.get("受众错位情况") != "无" and err_q and not is_verbatim_quote(err_q, visible_text):
                    errors.append(f"错位引文不在考生答卷中: {err_q[:40]}")
        if has_cats and not isinstance(res.get("归类核对"), dict):
            errors.append("分类题必须返回 归类核对 对象")

        if not errors:
            card_map = {cid: {"状态": "未写", "答卷引文": ""} for cid in valid_ids}
            for row in res.get("卡片核对", []):
                if row["状态"] != "未写":
                    card_map[row["卡号"]] = {"状态": row["状态"], "答卷引文": row.get("答卷引文", "")}
            return {
                "卡片判定": card_map,
                "公文客观核对": res.get("公文客观核对"),
                "归类核对": res.get("归类核对"),
            }
        prompt = GRADE_CHECKLIST_PROMPT + "\n上轮输出未通过逐字引文校验，请修正：" + "；".join(errors[:10])
    raise RuntimeError("答卷清单检核连续3次未通过逐字校验")


def compute_score_from_decisions(rubric, total_grids, overflow_text, pass_result):
    """纯 Python 确定性算分器：任何AI不参与数学加减。"""
    full_score = rubric["硬约束"]["满分"]
    word_limit = rubric["硬约束"]["字数上限"]
    pools = rubric["分池配置"]
    spec = rubric["题干槽位"]
    is_gongwen = spec["题型类别"] == "公文应用文"
    has_cats = bool(spec["强制归类维度"])

    card_decisions = pass_result["卡片判定"]
    card_details = []
    cluster_buckets = {cr["簇编号"]: 0.0 for cr in rubric["功能簇封顶表"]}

    for sc in rubric["采分卡片表"]:
        if sc["满分值"] <= 0:
            continue
        cid = sc["卡号"]
        dec = card_decisions.get(cid, {"状态": "未写", "答卷引文": ""})
        state = dec["状态"]
        if state == "完整":
            pts = sc["满分值"]
        elif state == "笼统":
            pts = round(sc["满分值"] * sc["笼统得分系数"], 2)
        else:
            pts = 0.0
        cluster_buckets[sc["所属簇"]] += pts
        card_details.append({
            "卡号": cid,
            "所属簇": sc["所属簇"],
            "核心语义": sc["核心语义"],
            "满分值": sc["满分值"],
            "判定": state,
            "原始得分": pts,
            "答卷引文": dec["答卷引文"],
        })

    cluster_breakdown = []
    sum_clusters = 0.0
    for cr in rubric["功能簇封顶表"]:
        gid = cr["簇编号"]
        if cr["簇封顶分"] <= 0:
            continue
        raw_g = round(cluster_buckets[gid], 2)
        capped_g = min(cr["簇封顶分"], raw_g)
        sum_clusters += capped_g
        cluster_breakdown.append({
            "簇编号": gid,
            "簇名称": cr["簇名称"],
            "簇内累计": raw_g,
            "簇封顶分": cr["簇封顶分"],
            "实得簇分": round(capped_g, 2),
        })

    content_score = min(pools["内容分池上限"], round(sum_clusters, 2))

    format_score = 0.0
    pragmatic_score = 0.0
    structure_score = 0.0
    deductions = []

    if is_gongwen:
        gw = pass_result["公文客观核对"] or {}
        subtype = spec["文种子类"]
        if subtype == "书信倡议通知类":
            if gw.get("有独立标题"):
                format_score += 0.5
            if gw.get("有受众称呼或首段直接面向受众说话"):
                format_score += 0.5
            if gw.get("有落款单位与日期"):
                format_score += 1.0
        else:
            if gw.get("有标题或首句明确发布宣告"):
                format_score += 1.0
            if gw.get("有受众称呼或首段直接面向受众说话"):
                format_score += 1.0
        if gw.get("落款单位或年份是否写错"):
            format_score = max(0.0, format_score - 0.5)
            deductions.append("落款单位或年份信息与题干冲突(-0.5)")
        format_score = min(pools["文体格式分池"], format_score)

        # 号召力与受众三要素客观联动算分
        cta_pts = 1.0 if gw.get("明确呼吁目标受众报名或行动") else 0.0
        inc_ids = set(rubric["公文与号召力客观细则"]["号召力客观三要素"]["联动激励卡号池"])
        inc_hits = sum(
            1 for cid in inc_ids
            if card_decisions.get(cid, {}).get("状态") in {"完整", "笼统"}
        )
        hook_pts = 1.0 if inc_hits >= 2 else (0.5 if inc_hits == 1 else 0.0)
        misalign = gw.get("受众错位情况", "无")
        if misalign == "无":
            aud_pts = 1.0
        elif misalign == "顺带提及非受众事项":
            aud_pts = 0.5
            deductions.append("公文混入面向非目标受众的申报/经费表述(-0.5)")
        else:
            aud_pts = 0.0
            deductions.append("严重受众错位：将非受众政策张冠李戴或未面向目标受众呼吁(-1.0)")
        pragmatic_score = min(pools["场景号召与受众分池"], round(cta_pts + hook_pts + aud_pts, 2))

    if has_cats:
        cat_check = pass_result["归类核对"] or {}
        structure_score = pools["分类条理分池"]
        if not cat_check.get("是否分条或分块作答"):
            structure_score = 0.0
            deductions.append(f"未分条归类作答(-{pools['分类条理分池']})")
        elif not cat_check.get("维度齐全且未严重混类") or cat_check.get("错归类条数", 0) > 0:
            wrong_n = max(1, int(cat_check.get("错归类条数", 1)))
            lost = min(structure_score, round(wrong_n * 0.5, 2))
            structure_score = round(structure_score - lost, 2)
            deductions.append(f"存在要点归类错位(-{lost})")

    over_penalty = 0.0
    # 留5格行尾标点共格容差，超出上限5格以上按每25格扣1分
    if word_limit and total_grids > word_limit + 5:
        excess = total_grids - word_limit
        steps = math.ceil(excess / 25)
        max_pen = round(full_score * 0.15, 1)
        over_penalty = min(max_pen, float(steps * 1.0))
        msg = f"总字数{total_grids}格，超出上限{word_limit}格共{excess}格，扣{over_penalty}分"
        if overflow_text.strip():
            msg += f"（超出物理答题卡{rubric['硬约束']['物理截断格数']}格的{count_exam_grids(overflow_text)}格已截断不计分）"
        deductions.append(msg)

    raw_total = content_score + format_score + pragmatic_score + structure_score - over_penalty
    raw_total = max(0.0, min(full_score, raw_total))
    # 申论阅卷标准半分为最小颗粒度
    final_score = round(raw_total * 2) / 2

    return {
        "最终得分": final_score,
        "精确折算前得分": round(raw_total, 2),
        "满分": full_score,
        "总格数": total_grids,
        "是否触发物理截断": bool(overflow_text.strip()),
        "分项得分": {
            "内容得分": content_score,
            "文体格式得分": format_score,
            "场景号召与受众得分": pragmatic_score,
            "分类条理得分": structure_score,
            "超字数扣分": over_penalty,
        },
        "扣分与风险提示": deductions,
        "功能簇得分明细": cluster_breakdown,
        "逐卡命中明细": card_details,
    }


def grade_answer_with_audit(rubric, answer_text, api_key=None, dual_pass=True):
    """完整批改流水线：物理方格门禁 -> 正反序双盲独立检核 -> 分歧单点仲裁 -> Python公式算分。"""
    # 使用更精准的方格测量逻辑
    measurement = measure_grid_sheet(answer_text, rubric["硬约束"]["字数上限"])
    visible_text = measurement["可见正文"]
    overflow_text = measurement["截断溢出正文"]
    total_grids = measurement["方格纸实占格数"]

    if not dual_pass:
        p1 = verify_checklist_pass(rubric, visible_text, shuffle_reverse=False, api_key=api_key)
        compiled = compute_score_from_decisions(rubric, total_grids, overflow_text, p1)
        return {**compiled, "双盲审计": None}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(verify_checklist_pass, rubric, visible_text, False, api_key)
        f2 = pool.submit(verify_checklist_pass, rubric, visible_text, True, api_key)
        p1, p2 = f1.result(), f2.result()

    s1 = compute_score_from_decisions(rubric, total_grids, overflow_text, p1)
    s2 = compute_score_from_decisions(rubric, total_grids, overflow_text, p2)

    # 找出两次独立检核中状态不一致的卡片，做聚焦单点仲裁
    card_lookup = {sc["卡号"]: sc for sc in rubric["采分卡片表"] if sc["满分值"] > 0}
    disputed = []
    for cid, sc in card_lookup.items():
        st1 = p1["卡片判定"][cid]["状态"]
        st2 = p2["卡片判定"][cid]["状态"]
        if st1 != st2:
            disputed.append({
                "卡号": cid,
                "核心语义": sc["核心语义"],
                "材料原文锚点": sc["原文"],
                "检核A判定": p1["卡片判定"][cid],
                "检核B判定": p2["卡片判定"][cid],
            })

    merged = {
        "卡片判定": dict(p1["卡片判定"]),
        "公文客观核对": p1["公文客观核对"],
        "归类核对": p1["归类核对"],
    }
    gw_disputed = (p1["公文客观核对"] != p2["公文客观核对"]) and (
        s1["分项得分"]["文体格式得分"] != s2["分项得分"]["文体格式得分"]
        or s1["分项得分"]["场景号召与受众得分"] != s2["分项得分"]["场景号召与受众得分"]
    )
    if disputed or gw_disputed:
        arb_payload = {
            "题干槽位": rubric["题干槽位"],
            "考生可见答卷": visible_text,
            "分歧卡片": disputed,
            "公文核对分歧": {"检核A": p1["公文客观核对"], "检核B": p2["公文客观核对"]} if gw_disputed else None,
        }
        arb_res = call_llm_json(ARBITRATE_PROMPT, arb_payload, api_key=api_key, temperature=0.0, max_tokens=4000)
        for item in arb_res.get("裁定", []):
            cid = item.get("卡号")
            st = item.get("状态")
            q = item.get("答卷引文", "")
            if cid in card_lookup and st in {"完整", "笼统", "冲突", "未写"}:
                if st == "未写" or is_verbatim_quote(q, visible_text):
                    merged["卡片判定"][cid] = {"状态": st, "答卷引文": "" if st == "未写" else q}
        if gw_disputed and isinstance(arb_res.get("公文客观核对"), dict):
            merged["公文客观核对"] = arb_res["公文客观核对"]

    final_compiled = compute_score_from_decisions(rubric, total_grids, overflow_text, merged)
    final_compiled["双盲审计"] = {
        "正序独立检核分": s1["最终得分"],
        "逆序独立检核分": s2["最终得分"],
        "原始双盲分差": round(abs(s1["最终得分"] - s2["最终得分"]), 2),
        "分歧卡片数": len(disputed),
        "分歧卡号": [d["卡号"] for d in disputed],
    }
    return final_compiled


def prefab_question(question_md_path, api_key=None, ref_limit=None, mode_label="多机构共识预制", hold_out_refs=None):
    """端到端题目预制入口：解析题干 -> 拆材料卡片 -> 对齐参考答案 -> Python数学编译锁定评分表。"""
    bundle = load_question_bundle(question_md_path)
    
    # 支持 hold-out 验证：根据来源过滤掉特定的参考答案
    if hold_out_refs:
        bundle["参考答案"] = [r for r in bundle["参考答案"] if r["来源"] not in hold_out_refs]
    
    constraints = parse_hard_constraints(bundle["题干与要求"])
    spec = call_llm_json(SPEC_PROMPT, {"题干与要求": bundle["题干与要求"]}, api_key=api_key, temperature=0.0)
    card_pack = extract_verified_cards(spec, bundle["材料段落"], api_key=api_key)

    used_refs = bundle["参考答案"] if ref_limit is None else bundle["参考答案"][:ref_limit]
    aligned = align_reference_answers(
        card_pack["卡片"],
        used_refs,
        is_gongwen=(spec["题型类别"] == "公文应用文"),
        api_key=api_key,
    )
    rubric = compile_quantitative_rubric(bundle, spec, constraints, card_pack, aligned, mode_label=mode_label)
    return rubric, bundle
