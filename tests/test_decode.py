"""Test gate for src/decode.py's beam_search_decode
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from decode import beam_search_decode, greedy_decode
from diagnostics import degeneracy_rate
from rnn_model import RNNSeq2Seq
from data import read_parallel, TranslationDataset, collate_fn
from vocab import EOS_ID, PAD_ID, SOS_ID, Vocab

HERE = os.path.dirname(__file__)
TOY_DIR = os.path.join(HERE, "..", "data", "toy")
SPM_DIR = os.path.join(HERE, "..", "data", "processed", "spm")


def _sequence_logprob(model, memory, src_pad_mask, seq, sos_id):
    # seq excludes sos, same format greedy_decode/beam_search_decode return.
    full = [sos_id] + seq
    total = 0.0
    for t in range(len(seq)):
        prefix = torch.tensor([full[:t + 1]])
        logits = model.decode_step(memory, src_pad_mask, prefix)
        log_probs = torch.log_softmax(logits, dim=-1)
        total += log_probs[0, full[t + 1]].item()
    return total


@pytest.fixture(scope="module")
def overfit_model():
    src_vocab = Vocab(os.path.join(SPM_DIR, "src_spm.model"))
    tgt_vocab = Vocab(os.path.join(SPM_DIR, "tgt_spm.model"))
    pairs = read_parallel(os.path.join(TOY_DIR, "train.vi"), os.path.join(TOY_DIR, "train.en"))[:20]
    dataset = TranslationDataset(pairs, src_vocab, tgt_vocab)
    batch = collate_fn([dataset[i] for i in range(len(dataset))], PAD_ID)

    model = RNNSeq2Seq(len(src_vocab), len(tgt_vocab), dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
    for _ in range(300):
        optimizer.zero_grad()
        logits = model(batch["src"], batch["src_pad_mask"], batch["tgt_in"])
        loss = criterion(logits.reshape(-1, logits.size(-1)), batch["tgt_out"].reshape(-1))
        loss.backward()
        optimizer.step()

    model.eval()
    return model, batch


def test_beam_beats_greedy_on_toy(overfit_model):
    model, batch = overfit_model
    for i in range(batch["src"].size(0)):
        src = batch["src"][i:i + 1]
        mask = batch["src_pad_mask"][i:i + 1]

        greedy_hyp = greedy_decode(model, src, mask, sos_id=SOS_ID, eos_id=EOS_ID)
        beam_hyp = beam_search_decode(model, src, mask, sos_id=SOS_ID, eos_id=EOS_ID, beam_width=5)

        with torch.no_grad():
            memory = model.encode(src, mask)
            greedy_score = _sequence_logprob(model, memory, mask, greedy_hyp, SOS_ID)
            beam_score = _sequence_logprob(model, memory, mask, beam_hyp, SOS_ID)

        assert beam_score >= greedy_score - 1e-4, (
            f"example {i}: beam score {beam_score:.4f} < greedy score {greedy_score:.4f}"
        )


def test_beam_no_degenerate(overfit_model):
    model, batch = overfit_model
    hyps = []
    for i in range(batch["src"].size(0)):
        src = batch["src"][i:i + 1]
        mask = batch["src_pad_mask"][i:i + 1]
        hyps.append(beam_search_decode(model, src, mask, sos_id=SOS_ID, eos_id=EOS_ID, beam_width=5))
    assert degeneracy_rate(hyps) == 0.0


def test_beam_equals_greedy_at_k1(overfit_model):
    model, batch = overfit_model
    for i in range(batch["src"].size(0)):
        src = batch["src"][i:i + 1]
        mask = batch["src_pad_mask"][i:i + 1]
        greedy_hyp = greedy_decode(model, src, mask, sos_id=SOS_ID, eos_id=EOS_ID)
        beam_hyp = beam_search_decode(model, src, mask, sos_id=SOS_ID, eos_id=EOS_ID, beam_width=1)
        assert beam_hyp == greedy_hyp, f"example {i}: beam={beam_hyp} greedy={greedy_hyp}"
