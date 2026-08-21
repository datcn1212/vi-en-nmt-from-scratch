"""Checks the output of src/prepare_iwslt.py in data/processed/
"""
import glob
import os
import random
import sys
import unicodedata

import sentencepiece as spm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import prepare_iwslt as prep 

HERE = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(HERE, "..", "data", "processed")
RAW_DIR = os.path.join(HERE, "..", "data", "raw")
SPLITS = [("train", "train"), ("dev", "tst2012"), ("test", "tst2013")]

# Precomposed Vietnamese vowels/consonant carrying a tone mark
TONE_CHARS = set(
    "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴĐ"
)


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def test_all_nfc():
    paths = glob.glob(os.path.join(PROCESSED_DIR, "*.vi")) + glob.glob(os.path.join(PROCESSED_DIR, "*.en"))
    assert paths, "no processed files found - run prepare_iwslt.py first"
    for path in paths:
        for line in read_lines(path):
            assert unicodedata.is_normalized("NFC", line), f"{path}: non-NFC line: {line!r}"


def test_alignment():
    for split, raw_name in SPLITS:
        vi_proc = read_lines(os.path.join(PROCESSED_DIR, f"{split}.vi"))
        en_proc = read_lines(os.path.join(PROCESSED_DIR, f"{split}.en"))
        assert len(vi_proc) == len(en_proc), f"{split}: line count mismatch, vi={len(vi_proc)} en={len(en_proc)}"

        # Spot-check
        vi_raw = prep.clean_lines(os.path.join(RAW_DIR, f"{raw_name}.vi"))
        en_raw = prep.clean_lines(os.path.join(RAW_DIR, f"{raw_name}.en"))
        vi_raw_index = {line: i for i, line in enumerate(vi_raw)}

        for i in random.sample(range(len(vi_proc)), min(5, len(vi_proc))):
            raw_i = vi_raw_index.get(vi_proc[i])
            assert raw_i is not None, f"{split}: processed vi[{i}] not found in cleaned raw source"
            assert en_raw[raw_i] == en_proc[i], f"{split}: pair at processed index {i} no longer corresponds"


def test_spm_roundtrip():
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(PROCESSED_DIR, "spm", "src_spm.model"))

    lines = sorted(set(read_lines(os.path.join(PROCESSED_DIR, "train.vi"))),
                    key=lambda line: len(set(line) & TONE_CHARS), reverse=True)
    sample = random.sample(lines[:20], 5)

    for text in sample:
        assert sp.decode(sp.encode(text)) == text, f"round-trip mismatch: {text!r}"
