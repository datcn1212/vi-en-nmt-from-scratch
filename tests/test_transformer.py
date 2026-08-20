"""Test gate for src/transformer_model.py: encoder masking, decoder overfit
capacity, and causal-mask correctness
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from decode import greedy_decode
from transformer_model import TransformerSeq2Seq
from data import read_parallel, TranslationDataset, collate_fn
from vocab import EOS_ID, PAD_ID, SOS_ID, Vocab

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


def test_encoder_ignores_padding():
    batch, src_vocab, tgt_vocab = _load_batch("train")
    src, mask = batch["src"], batch["src_pad_mask"]
    assert mask.any(), "batch must actually contain padding"

    model = TransformerSeq2Seq(len(src_vocab), len(tgt_vocab))
    model.eval()

    with torch.no_grad():
        out1 = model.encode(src, mask)
        src2 = src.clone()
        random_ids = torch.randint(4, len(src_vocab), src2.shape)
        src2[mask] = random_ids[mask]
        out2 = model.encode(src2, mask)

    assert torch.allclose(out1[~mask], out2[~mask], atol=1e-5)


@pytest.fixture(scope="module")
def overfit_transformer():
    batch, src_vocab, tgt_vocab = _load_batch("train", n=20)
    model = TransformerSeq2Seq(len(src_vocab), len(tgt_vocab), dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for _ in range(300):
        optimizer.zero_grad()
        logits = model(batch["src"], batch["src_pad_mask"], batch["tgt_in"])
        loss = criterion(logits.reshape(-1, logits.size(-1)), batch["tgt_out"].reshape(-1))
        loss.backward()
        optimizer.step()

    return model, batch, loss.item()


def test_overfit_20_transformer(overfit_transformer):
    _, _, final_loss = overfit_transformer
    assert final_loss < 0.5, f"final loss {final_loss:.4f} did not drop below 0.5"


def test_autoregressive_generation_transformer(overfit_transformer):
    model, batch, _ = overfit_transformer
    model.eval()

    matches = 0
    n = batch["src"].size(0)
    for i in range(n):
        src = batch["src"][i:i + 1]
        mask = batch["src_pad_mask"][i:i + 1]
        # tgt_out already excludes sos and keeps eos, the same format greedy_decode returns.
        target = [t for t in batch["tgt_out"][i].tolist() if t != PAD_ID]

        hyp = greedy_decode(model, src, mask, sos_id=SOS_ID, eos_id=EOS_ID)
        if hyp == target:
            matches += 1

    rate = matches / n
    assert rate >= 0.7, f"exact-match rate {rate:.2f} ({matches}/{n})"
