"""Build LM resources for the speller.

Reads an optional Chinese text corpus from --corpus and produces:
  data/lm/char_freq.json
  data/lm/char_to_words.json
  data/lm/word_bigram.json

When --corpus is not supplied, generates a richer-than-fallback bundled set
from a small built-in seed list. Either way, the speller demo will pick up
these JSON files at runtime via LanguageModel.from_resources(...).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LM_DIR = ROOT / "data" / "lm"


def _is_chinese(ch: str) -> bool:
    return "一" <= ch <= "鿿"


def build(corpus_path: Path | None, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = corpus_path.read_text(encoding="utf-8") if corpus_path else _seed_text()

    # char freq
    chars = [c for c in text if _is_chinese(c)]
    char_freq = dict(Counter(chars))

    # word freq via jieba
    import jieba
    tokens = [w for w in jieba.lcut(text) if all(_is_chinese(c) for c in w) and len(w) >= 2]
    word_freq: Counter[str] = Counter(tokens)

    # leading-char → words
    char_to_words: dict[str, list[str]] = defaultdict(list)
    for w, _ in word_freq.most_common():
        char_to_words[w[0]].append(w)
    char_to_words = {k: v[:20] for k, v in char_to_words.items()}

    # word bigram
    bigram_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for a, b in zip(tokens, tokens[1:]):
        if len(a) >= 2 and len(b) >= 2:
            bigram_counts[a][b] += 1
    word_bigram = {a: [w for w, _ in cnt.most_common(20)] for a, cnt in bigram_counts.items()}

    (out_dir / "char_freq.json").write_text(
        json.dumps(char_freq, ensure_ascii=False, indent=0), encoding="utf-8")
    (out_dir / "char_to_words.json").write_text(
        json.dumps(char_to_words, ensure_ascii=False, indent=0), encoding="utf-8")
    (out_dir / "word_bigram.json").write_text(
        json.dumps(word_bigram, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"wrote {out_dir} with {len(char_freq)} chars, "
          f"{sum(len(v) for v in char_to_words.values())} words, "
          f"{len(word_bigram)} bigrams")


def _seed_text() -> str:
    """Tiny built-in corpus that gives plausible candidates for demo letters."""
    return (
        "我想要喝水。我想要吃饭。我需要帮助。我希望明天天气好。"
        "你好吗？你知道这件事吗？他想要去看电影。她需要更多时间。"
        "我们觉得这个想法很好。他们认为应该这样做。"
        "你想去哪儿？我们一起吃晚饭吧。喝点水休息一下。"
        "今天天气真好啊。明天我要早起去上班。我想睡觉了。"
        "什么时候开始？什么意思？没什么事。"
        "外面下雨了。无法出去玩。问题不大。"
    ) * 8  # repeat so jieba bigram counts > 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=None,
                    help="UTF-8 plain-text Chinese corpus; defaults to bundled seed text")
    ap.add_argument("--out", type=Path, default=LM_DIR)
    args = ap.parse_args()
    build(args.corpus, args.out)


if __name__ == "__main__":
    main()
