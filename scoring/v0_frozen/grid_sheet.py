import math
import re


ROW_WIDTH = 25


def count_pure_exam_chars(text):
    """统计纯申论字符数（不含空白换行）：汉字与中文标点计1，连续2个ASCII字母数字计1。"""
    cleaned = re.sub(r"\s+", "", text or "")
    if not cleaned:
        return 0
    total = 0
    ascii_run = 0
    for ch in cleaned:
        if ord(ch) < 128 and ch.isalnum():
            ascii_run += 1
        else:
            if ascii_run:
                total += math.ceil(ascii_run / 2)
                ascii_run = 0
            total += 1
    if ascii_run:
        total += math.ceil(ascii_run / 2)
    return total


def detect_answer_completeness(text, expected_min_chars=None):
    """识别答卷属于正常成文作答，还是考后回忆提纲/含未展开占位符的残卷。"""
    raw = (text or "").strip()
    pure_chars = count_pure_exam_chars(raw)
    if not raw:
        return {"形态": "空白卷", "是否残卷或提纲": True, "原因": ["正文为空"]}

    reasons = []
    explicit_markers = [
        r"(?:整体|写作|答题|论证)思路[：:—\-]",
        r"(?:总论点|中心论点)[：:]",
        r"分论点[一二三123：:]",
        r"(?:此处|正文|中间|后面|结尾)(?:有|存在)?(?:省略|遗忘|记不清|忘了)",
        r"(?:例证|论据|例子|结尾)(?:略|待补|记不清|忘了)",
        r"\.{4,}|…{2,}",
    ]
    for pat in explicit_markers:
        m = re.search(pat, raw)
        if m:
            reasons.append(f"含提纲或省略标记“{m.group(0)[:16]}”")

    if expected_min_chars and pure_chars < expected_min_chars * 0.52:
        reasons.append(f"纯字数仅 {pure_chars} 字，不足最低成文要求({expected_min_chars}字)的一半")

    is_incomplete = len(reasons) > 0
    return {
        "形态": "回忆提纲或残卷" if is_incomplete else "完整成文作答",
        "是否残卷或提纲": is_incomplete,
        "纯字数": pure_chars,
        "原因": reasons,
    }


def measure_grid_sheet(text, max_words, row_width=ROW_WIDTH, buffer_rows=None):
    """按真实申论25格方格纸测量：含标题居中独占整行、段首缩进2格、换行行尾留白损耗及答题卡底线截断。"""
    raw_lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    if not raw_lines:
        limit_rows = math.ceil(max_words / row_width) if max_words else None
        extra = buffer_rows if buffer_rows is not None else (4 if (max_words and max_words >= 700) else 3)
        return {
            "纯字数": 0,
            "段落行数": 0,
            "方格纸实占格数": 0,
            "方格纸实占行数": 0,
            "标准上限行数": limit_rows,
            "物理截断行数": (limit_rows + extra) if limit_rows else None,
            "换行留白损耗格数": 0,
            "可见正文": "",
            "截断溢出正文": "",
        }

    limit_rows = math.ceil(max_words / row_width) if max_words else None
    if buffer_rows is None:
        # 真实国考答题卡通常多给3~4行缓冲格，防止正常分段考生被误截断，同时拦截恶意超长粘贴
        buffer_rows = 4 if (max_words and max_words >= 700) else 3
    cutoff_rows = (limit_rows + buffer_rows) if limit_rows else None

    used_rows = 0
    visible_parts = []
    overflow_parts = []

    for idx, ln in enumerate(raw_lines):
        line_chars = count_pure_exam_chars(ln)
        is_standalone_title = (
            idx == 0
            and len(raw_lines) > 1
            and line_chars <= 22
            and not re.match(r"^[一二三四五六七八九十\d（(①②③]|^事项|^原因|^问题|^建议|^对策|^理由|^成效", ln)
            and not ln.endswith(("：", ":", "。", "；"))
        )

        if is_standalone_title:
            needed_rows = 1
        else:
            needed_rows = max(1, math.ceil((2 + line_chars) / row_width))

        if cutoff_rows is None or used_rows + needed_rows <= cutoff_rows:
            used_rows += needed_rows
            visible_parts.append(ln)
        elif used_rows < cutoff_rows:
            remain_rows = cutoff_rows - used_rows
            avail_chars = max(1, remain_rows * row_width - (0 if is_standalone_title else 2))
            vis_seg, over_seg = _slice_line_by_chars(ln, avail_chars)
            if vis_seg:
                visible_parts.append(vis_seg)
            if over_seg:
                overflow_parts.append(over_seg)
            used_rows = cutoff_rows
        else:
            overflow_parts.append(ln)
            used_rows += needed_rows

    pure_total = count_pure_exam_chars("\n".join(raw_lines))
    sheet_grids = used_rows * row_width
    return {
        "纯字数": pure_total,
        "段落行数": len(raw_lines),
        "方格纸实占格数": sheet_grids,
        "方格纸实占行数": used_rows,
        "标准上限行数": limit_rows,
        "物理截断行数": cutoff_rows,
        "换行留白损耗格数": max(0, sheet_grids - pure_total),
        "可见正文": "\n".join(visible_parts),
        "截断溢出正文": "\n".join(overflow_parts),
    }


def _slice_line_by_chars(line, max_chars):
    cnt = 0
    ascii_run = 0
    cut = len(line)
    for i, ch in enumerate(line):
        if ch.isspace():
            continue
        if ord(ch) < 128 and ch.isalnum():
            nxt = cnt + math.ceil((ascii_run + 1) / 2)
            if nxt > max_chars:
                cut = i
                break
            ascii_run += 1
        else:
            if ascii_run:
                cnt += math.ceil(ascii_run / 2)
                ascii_run = 0
            if cnt + 1 > max_chars:
                cut = i
                break
            cnt += 1
    return line[:cut], line[cut:]