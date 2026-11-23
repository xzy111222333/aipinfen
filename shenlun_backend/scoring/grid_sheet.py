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

def measure_benchmark_layout(text, row_width):
    """按显式段落和行首空格估算到最后一个文字为止的布局占格。"""
    used = 0
    for line in text.strip("\r\n").splitlines():
        if used % row_width:
            used += row_width - used % row_width
        content = line.lstrip(" \u3000")
        if not content.strip():
            used += row_width
        else:
            used += len(line) - len(content) + count_pure_exam_chars(content)
    return used
