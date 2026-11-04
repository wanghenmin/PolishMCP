"""分段与长度统计的纯逻辑测试（不联网）。"""

from __future__ import annotations

from aimcp.text import count_text_length, split_text_into_segments


def test_count_chinese_only() -> None:
    assert count_text_length("你好世界") == 4


def test_count_english_only() -> None:
    assert count_text_length("hello world") == 10


def test_count_mixed_prefers_chinese() -> None:
    assert count_text_length("abc你好") == 2


def test_split_by_newline_drops_blank_lines() -> None:
    text = "第一段\n第二段\n\n第三段"
    assert split_text_into_segments(text) == ["第一段", "第二段", "第三段"]


def test_split_keeps_short_paragraphs_intact() -> None:
    text = "短标题\n这是一段正常长度的正文内容"
    assert split_text_into_segments(text, max_chars=800) == text.split("\n")


def test_long_paragraph_is_split_by_sentence_and_lossless() -> None:
    text = "一句话。两句话。三句话。"
    segments = split_text_into_segments(text, max_chars=5)

    assert all(count_text_length(segment) <= 5 for segment in segments)
    assert "".join(segments) == text


def test_single_short_text_returns_one_segment() -> None:
    assert split_text_into_segments("只有一句话。") == ["只有一句话。"]
