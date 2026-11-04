"""文本长度统计与分段工具。

逻辑与 BypassAIGC 保持一致：中文按汉字数计量，纯英文按字母数计量；
先按换行分段，单段过长时再按句末标点切分。
"""

from __future__ import annotations

import re

_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_ENGLISH_RE = re.compile(r"[a-zA-Z]")


def count_text_length(text: str) -> int:
    """统计文本长度：有汉字按汉字数，纯英文按字母数。"""
    chinese_count = len(_CHINESE_RE.findall(text))
    if chinese_count > 0:
        return chinese_count
    return len(_ENGLISH_RE.findall(text))


def split_text_into_segments(text: str, max_chars: int = 800) -> list[str]:
    """把文本切成适合单次改写的段落列表。

    先按换行分段并丢弃空行；单段超过 max_chars 时，再按句末标点（。!?;）切分。
    """
    segments: list[str] = []

    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue

        if count_text_length(para) <= max_chars:
            segments.append(para)
            continue

        pieces = re.split(r"([。!?;])", para)
        current = ""
        for i in range(0, len(pieces), 2):
            sentence = pieces[i]
            if i + 1 < len(pieces):
                sentence += pieces[i + 1]

            if count_text_length(current + sentence) <= max_chars:
                current += sentence
            else:
                if current:
                    segments.append(current)
                current = sentence

        if current:
            segments.append(current)

    return segments
