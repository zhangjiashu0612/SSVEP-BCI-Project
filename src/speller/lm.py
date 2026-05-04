"""Language model for the SSVEP speller.

Three layers of prediction, all driven by integer-indexed lookups so the state
machine can stay simple:

  predict_char(prefix)         e.g. "w"   -> ["我", "为", "外", "完", "万", "无", ...]
  predict_word(char)           e.g. "我"  -> ["想要", "需要", "觉得", "希望", ...]
  predict_continuation(word)   e.g. "想要" -> ["喝水", "吃饭", "睡觉", "出去", ...]

Data sources (in priority order):
  1. JSON resources under `data/lm/` if present (built by scripts/build_lm.py)
  2. pypinyin's built-in dict for prefix→characters
  3. A small bundled fallback dict so the speller demo runs end-to-end on a
     fresh clone with zero external data — picking sensible candidates for
     the demo letters used in the README.

Resource files:
  data/lm/char_freq.json       {char: frequency}
  data/lm/char_to_words.json   {leading_char: [words sorted by freq]}
  data/lm/word_bigram.json     {word: [next words sorted by transition prob]}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# ---- bundled fallback (kept tiny — only seeds the demo letters) ----------

_FALLBACK_CHAR_FREQ: dict[str, float] = {
    # Top-frequency Chinese characters, hand-picked from common letter prefixes
    # so the demo for "w", "n", "h", "s" feels real even without resources.
    "的": 100, "一": 92, "是": 90, "不": 88, "了": 85, "在": 82, "有": 80,
    "人": 78, "我": 95, "他": 70, "这": 68, "中": 66, "大": 64, "来": 62,
    "上": 60, "国": 58, "个": 56, "到": 54, "说": 52, "们": 50,
    "为": 75, "子": 48, "和": 46, "你": 72, "地": 44, "出": 42, "也": 40,
    "时": 38, "道": 36, "得": 34, "可": 32, "好": 65, "想": 88, "要": 86,
    "会": 42, "多": 38, "能": 50, "都": 48, "自": 36, "看": 70, "去": 60,
    "什": 55, "么": 56, "吃": 65, "饭": 60, "喝": 60, "水": 55, "睡": 50,
    "觉": 48, "完": 45, "万": 35, "外": 50, "无": 40, "问": 60, "哇": 30,
    "可": 38, "需": 55, "希": 50, "望": 45, "认": 42, "是": 90,
}

_FALLBACK_CHAR_TO_WORDS: dict[str, list[str]] = {
    "我": ["想要", "需要", "觉得", "希望", "认为", "知道"],
    "你": ["好吗", "知道", "想要", "需要", "看到", "认识"],
    "他": ["的", "们", "说", "想", "看到", "知道"],
    "好": ["的", "吗", "看", "了", "啊", "极了"],
    "想": ["要", "到", "想", "起", "法", "念"],
    "要": ["求", "是", "知道", "做", "去", "来"],
    "吃": ["饭", "完", "了", "东西", "晚饭", "早饭"],
    "喝": ["水", "茶", "咖啡", "酒", "汤", "饮料"],
    "睡": ["觉", "眠", "着", "醒", "梦", "了"],
    "看": ["到", "见", "着", "完", "书", "电影"],
    "去": ["了", "过", "哪儿", "看", "做", "学校"],
    "什": ["么"],
    "完": ["成", "了", "全", "美", "毕"],
    "为": ["什么", "了", "什", "什么不"],
    "外": ["面", "国", "出", "婆", "公", "套"],
    "无": ["法", "数", "论", "限", "穷", "聊"],
    "问": ["题", "好", "答", "话", "候"],
    "万": ["一", "事", "能", "分", "岁"],
}

_FALLBACK_WORD_BIGRAM: dict[str, list[str]] = {
    "想要": ["喝水", "吃饭", "睡觉", "出去", "回家", "知道"],
    "需要": ["帮助", "时间", "更多", "什么", "你的", "考虑"],
    "觉得": ["很好", "怎么", "可以", "应该", "不错", "有点"],
    "希望": ["你能", "可以", "明天", "大家", "你们"],
    "认为": ["这个", "应该", "不是", "可以", "他是"],
    "知道": ["了", "什么", "怎么", "你的", "他的"],
    "什么": ["时候", "意思", "事", "东西", "样子"],
    "吃饭": ["了吗", "去吧", "时间", "之前"],
    "喝水": ["吧", "了", "去", "之后"],
    "睡觉": ["吧", "了", "之前", "时间"],
}


_LETTER_PINYIN_PREFIX_RE = re.compile(r"^[a-z]+$")


@dataclass
class LanguageModel:
    char_freq: dict[str, float]
    char_to_words: dict[str, list[str]]
    word_bigram: dict[str, list[str]]
    prefix_to_chars: dict[str, list[str]]  # pinyin-prefix → chars sorted desc by freq

    @classmethod
    def from_resources(cls, resources_dir: str | Path | None = None) -> "LanguageModel":
        char_freq = dict(_FALLBACK_CHAR_FREQ)
        char_to_words = {k: list(v) for k, v in _FALLBACK_CHAR_TO_WORDS.items()}
        word_bigram = {k: list(v) for k, v in _FALLBACK_WORD_BIGRAM.items()}
        if resources_dir is not None:
            base = Path(resources_dir)
            char_freq = _load_json_dict(base / "char_freq.json", char_freq)
            char_to_words = _load_json_listdict(base / "char_to_words.json", char_to_words)
            word_bigram = _load_json_listdict(base / "word_bigram.json", word_bigram)
        prefix_to_chars = _build_prefix_index(char_freq)
        return cls(char_freq=char_freq, char_to_words=char_to_words,
                   word_bigram=word_bigram, prefix_to_chars=prefix_to_chars)

    # ---- public prediction API -----------------------------------------

    def predict_char(self, prefix: str, k: int = 6) -> list[str]:
        """Letter prefix → top-k characters by frequency."""
        if not prefix or not _LETTER_PINYIN_PREFIX_RE.match(prefix):
            return []
        chars = self.prefix_to_chars.get(prefix, [])
        return chars[:k]

    def predict_word(self, char: str, k: int = 6) -> list[str]:
        """Single confirmed character → top-k 2+ char words starting with it."""
        if not char:
            return []
        return list(self.char_to_words.get(char, []))[:k]

    def predict_continuation(self, word: str, k: int = 6) -> list[str]:
        """Word → top-k bigram continuations."""
        if not word:
            return []
        return list(self.word_bigram.get(word, []))[:k]


# ---- helpers ---------------------------------------------------------------

def _load_json_dict(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        return {**fallback, **json.load(f)}


def _load_json_listdict(path: Path, fallback: dict[str, list[str]]) -> dict[str, list[str]]:
    if not path.exists():
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    out = {k: list(v) for k, v in fallback.items()}
    for k, v in loaded.items():
        out[k] = list(v) + [x for x in out.get(k, []) if x not in v]
    return out


def _build_prefix_index(char_freq: dict[str, float]) -> dict[str, list[str]]:
    """Use pypinyin to invert char→pinyin into pinyin-prefix→[chars sorted by freq]."""
    try:
        from pypinyin import lazy_pinyin, Style
    except ImportError:  # pragma: no cover
        return {}

    by_prefix: dict[str, list[tuple[str, float]]] = {}
    for ch, freq in char_freq.items():
        try:
            pys = lazy_pinyin(ch, style=Style.NORMAL)
        except Exception:
            continue
        if not pys:
            continue
        py = pys[0]
        if not py or not py[0].isalpha():
            continue
        for n in range(1, len(py) + 1):
            prefix = py[:n].lower()
            by_prefix.setdefault(prefix, []).append((ch, freq))

    # Also fold in pypinyin's full character dictionary so prefix lookups work
    # for chars not in our freq table — assigning them a small floor weight so
    # they rank below anything we explicitly scored.
    try:
        from pypinyin import pinyin_dict
        for codepoint, py_string in pinyin_dict.pinyin_dict.items():
            ch = chr(codepoint)
            if ch in char_freq:
                continue
            py = py_string.split(",")[0]
            py = "".join(c for c in py if c.isascii() and c.isalpha()).lower()
            if not py:
                continue
            for n in range(1, len(py) + 1):
                by_prefix.setdefault(py[:n], []).append((ch, 0.1))
    except Exception:
        pass

    out: dict[str, list[str]] = {}
    for prefix, items in by_prefix.items():
        seen: dict[str, float] = {}
        for ch, w in items:
            seen[ch] = max(seen.get(ch, 0.0), w)
        out[prefix] = [ch for ch, _ in sorted(seen.items(), key=lambda kv: -kv[1])]
    return out


def iter_letters() -> Iterable[str]:
    return (chr(c) for c in range(ord("a"), ord("z") + 1))
