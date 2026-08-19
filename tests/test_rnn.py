"""Test gate for src/rnn_model.py: memorization capacity and attention masking,
both against the toy corpus - before either costs real training time.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rnn_model import RNNSeq2Seq
from data import read_parallel, TranslationDataset, collate_fn
from vocab import PAD_ID, Vocab

HERE = os.path.dirname(__file__)
TOY_DIR = os.path.join(HERE, "..", "data", "toy")
SPM_DIR = os.path.join(HERE, "..", "data", "processed", "spm")


def _load_batch(split, n=None):
    src_vocab = Vocab(os.path.join(SPM_DIR, "src_spm.model"))
    tgt_vocab = Vocab(os.path.join(SPM_DIR, "tgt_spm.model"))
    pairs = read_parallel(os.path.join(TOY_DIR, f"{split}.vi"), os.path.join(TOY_DIR, f"{split}.en"))
    if n is not None:
        pairs = pairs[:n]
    dataset = TranslationDataset(pairs, src_vocab, tgt_vocab)
    batch = collate_fn([dataset[i] for i in range(len(dataset))], PAD_ID)
    return batch, src_vocab, tgt_vocab


def test_overfit_20():
    batch, src_vocab, tgt_vocab = _load_batch("train", n=20)
    # dropout off: this checks memorization capacity/wiring, not generalization -
    # dropout noise would stop 20 sentences reaching a near-zero loss in 300 steps.
    model = RNNSeq2Seq(len(src_vocab), len(tgt_vocab), dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for _ in range(300):
        optimizer.zero_grad()
        logits = model(batch["src"], batch["src_pad_mask"], batch["tgt_in"])
        loss = criterion(logits.reshape(-1, logits.size(-1)), batch["tgt_out"].reshape(-1))
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.5, f"final loss {loss.item():.4f} did not drop below 0.5"


def test_attention_ignores_padding():
    batch, src_vocab, tgt_vocab = _load_batch("train")
    lengths = (~batch["src_pad_mask"]).sum(dim=1)
    short_idx, long_idx = lengths.argmin().item(), lengths.argmax().item()
    assert short_idx != long_idx, "need a real short/long pair to test padding"

    pair = torch.stack([batch["src"][short_idx], batch["src"][long_idx]])
    mask = torch.stack([batch["src_pad_mask"][short_idx], batch["src_pad_mask"][long_idx]])
    assert mask.any(), "the shorter sentence must actually be padded here"

    model = RNNSeq2Seq(len(src_vocab), len(tgt_vocab))
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden = model.encode(pair, mask)
        _, alpha = model.attention(hidden, encoder_outputs, mask)

    assert (alpha[mask] < 0.05).all(), f"max attention weight on padding: {alpha[mask].max().item():.4f}"
