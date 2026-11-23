import json

import math

import os

import re

from pathlib import Path

RELAY_MODEL = "deepseek-v4.1-flash"

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

MDKJ_KEY = "SHENLUN_MDKJ_API_KEY"

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
