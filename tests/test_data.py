"""Tests for src/data.py and src/vocab.py against the toy corpus. 
Reuses the SentencePiece models already trained in data/processed/spm/ 
- the tokenizer is frozen and shared everywhere, not retrained per dataset.
"""
import os
import sys

from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import read_parallel, TranslationDataset, collate_fn
from vocab import PAD_ID, SOS_ID, Vocab

HERE = os.path.dirname(__file__)
TOY_DIR = os.path.join(HERE, "..", "data", "toy")
SPM_DIR = os.path.join(HERE, "..", "data", "processed", "spm")


def _load(split, batch_size):
    src_vocab = Vocab(os.path.join(SPM_DIR, "src_spm.model"))
    tgt_vocab = Vocab(os.path.join(SPM_DIR, "tgt_spm.model"))
    pairs = read_parallel(os.path.join(TOY_DIR, f"{split}.vi"), os.path.join(TOY_DIR, f"{split}.en"))
    dataset = TranslationDataset(pairs, src_vocab, tgt_vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=lambda b: collate_fn(b, PAD_ID))
    batch = next(iter(loader))
    return batch, src_vocab, pairs


def test_shift():
    batch, _, _ = _load("dev", batch_size=5)
    tgt_in, tgt_out = batch["tgt_in"], batch["tgt_out"]
    assert (tgt_in[:, 1:] == tgt_out[:, :-1]).all()
    assert (tgt_in[:, 0] == SOS_ID).all()


def test_shapes():
    batch, _, _ = _load("dev", batch_size=5)
    assert batch["tgt_in"].shape == batch["tgt_out"].shape


def test_pad_mask():
    # Whole train split in one batch
    batch, src_vocab, pairs = _load("train", batch_size=40)
    src, mask = batch["src"], batch["src_pad_mask"]

    max_len = src.size(1)
    expected_pad = sum(max_len - len(src_vocab.encode(src_text)) for src_text, _ in pairs)
    assert expected_pad > 0, "batch has no padding - pick sentences of mixed length"
    assert mask.sum().item() == expected_pad


def test_roundtrip():
    batch, src_vocab, pairs = _load("dev", batch_size=5)
    src = batch["src"]
    for i in range(src.size(0)):
        decoded = src_vocab.decode(src[i].tolist())
        assert decoded == pairs[i][0], f"roundtrip mismatch: {decoded!r} vs {pairs[i][0]!r}"
