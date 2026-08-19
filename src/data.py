"""Shared data pipeline: reads parallel text, wraps it as a Dataset, and
collates batches with the shift and mask both architectures consume
identically.
"""
import torch
from torch.utils.data import Dataset


def read_parallel(src_path, tgt_path):
    with open(src_path, encoding="utf-8") as f:
        src_lines = f.read().splitlines()
    with open(tgt_path, encoding="utf-8") as f:
        tgt_lines = f.read().splitlines()
    assert len(src_lines) == len(tgt_lines), (
        f"line count mismatch: {len(src_lines)} vs {len(tgt_lines)}"
    )
    return list(zip(src_lines, tgt_lines))


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        return self.src_vocab.encode(src_text), self.tgt_vocab.encode(tgt_text)


def _pad_batch(seqs, pad_id):
    length = max(len(s) for s in seqs)
    out = torch.full((len(seqs), length), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def collate_fn(batch, pad_id):
    src_ids, tgt_ids = zip(*batch)

    src = _pad_batch(src_ids, pad_id)
    src_pad_mask = src == pad_id

    # Pad the full [sos,...,eos] sequence once, then slice tgt_in/tgt_out from
    # that same padded tensor. Padding tgt_in and tgt_out separately looks
    # equivalent but is not: at the real-content/padding boundary, the eos
    # that should land in tgt_out ends up replaced by padding instead, since
    # each side gets padded to its own batch-max independently of the other.
    tgt = _pad_batch(tgt_ids, pad_id)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]

    return {"src": src, "src_pad_mask": src_pad_mask, "tgt_in": tgt_in, "tgt_out": tgt_out}
