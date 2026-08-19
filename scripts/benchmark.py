"""Estimate minutes/epoch on this machine using synthetic data.

At start, there is no real corpus and no real model, so this times STAND-IN
models with the same cost driver as the real ones (the RNN's per-timestep
decode loop, the Transformer's parallel forward).

    python scripts/benchmark.py --arch rnn
    python scripts/benchmark.py --arch transformer
    python scripts/benchmark.py --arch rnn --from-src             # once src/ has the real model
    python scripts/benchmark.py --arch rnn --measure_seq_len ...  # once real tokenized data exists
"""
import argparse
import math
import os
import random
import statistics
import sys
import time
import sentencepiece as spm
import torch
import torch.nn as nn


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    # cuda/mps calls are async; without this the timer measures queuing, not compute.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class RNNStandIn(nn.Module):
    """Same cost driver as the real model: a per-timestep decode loop with attention."""

    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256):
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, emb_dim)
        self.tgt_embed = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.GRU(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.enc_to_dec = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_keys = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)
        self.cell = nn.GRUCell(emb_dim + 2 * hidden_dim, hidden_dim)
        self.out = nn.Linear(3 * hidden_dim, vocab_size)

    def forward(self, src, tgt_in):
        enc_out, h = self.encoder(self.src_embed(src))
        hidden = torch.tanh(self.enc_to_dec(torch.cat([h[0], h[1]], dim=-1)))
        keys = self.attn_keys(enc_out)
        emb = self.tgt_embed(tgt_in)
        steps = []
        for t in range(tgt_in.size(1)):
            scores = self.attn_v(torch.tanh(self.attn_query(hidden).unsqueeze(1) + keys)).squeeze(-1)
            alpha = torch.softmax(scores, dim=-1)
            context = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)
            hidden = self.cell(torch.cat([emb[:, t], context], dim=-1), hidden)
            steps.append(self.out(torch.cat([hidden, context], dim=-1)))
        return torch.stack(steps, dim=1)


class TransformerStandIn(nn.Module):
    """Uses nn.Transformer on purpose, so it can never be mistaken for real src/ code."""

    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2, d_ff=512):
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.core = nn.Transformer(
            d_model=d_model, nhead=num_heads,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=d_ff, batch_first=True,
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt_in):
        causal = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1), device=tgt_in.device)
        h = self.core(self.src_embed(src), self.tgt_embed(tgt_in), tgt_mask=causal)
        return self.out(h)


def effective_seq_len(src_path, tgt_path, spm_path, batch_size, samples=200, seed=0):
    """Batches pad to their longest member, not the mean length -- measure that instead."""
    sp = spm.SentencePieceProcessor()
    sp.load(spm_path)
    lengths = [[len(sp.encode(line.strip())) + 2 for line in open(p)] for p in (src_path, tgt_path)]
    per_pair = [max(a, b) for a, b in zip(*lengths)]
    rng = random.Random(seed)
    maxima = [max(rng.sample(per_pair, batch_size)) for _ in range(samples)]
    return round(statistics.mean(maxima))


def build_from_src(arch, vocab_size, pad_id):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    module = "rnn_model" if arch == "rnn" else "transformer_model"
    try:
        if arch == "rnn":
            from rnn_model import RNNSeq2Seq as Model
        else:
            from transformer_model import TransformerSeq2Seq as Model
    except ModuleNotFoundError:
        raise SystemExit(f"--from-src needs src/{module}.py, which does not exist yet. "
                          f"Drop the flag to use the stand-in.")
    return Model(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, pad_id=pad_id)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=["rnn", "transformer"], required=True)
    p.add_argument("--steps", type=int, default=50, help="timed steps, after warmup")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=30,
                    help="guess; batches actually pad to their longest member (usually more)")
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--train_pairs", type=int, default=133318,
                    help="published train-split size (stefan-it/nmt-en-vi); refine once counted locally")
    p.add_argument("--epochs", type=int, default=12, help="epochs to project a full run")
    p.add_argument("--from-src", dest="from_src", action="store_true",
                    help="time the real model in src/ instead of the stand-in")
    p.add_argument("--measure_seq_len", nargs=3, metavar=("SRC", "TGT", "SPM"),
                    help="derive seq_len from the real corpus instead of guessing")
    args = p.parse_args()

    device = pick_device()
    pad_id = 0

    if args.measure_seq_len:
        args.seq_len = effective_seq_len(*args.measure_seq_len, args.batch_size)
        print(f"measured seq_len (mean batch max): {args.seq_len}\n")

    if args.from_src:
        model = build_from_src(args.arch, args.vocab_size, pad_id)
    else:
        model = RNNStandIn(args.vocab_size) if args.arch == "rnn" else TransformerStandIn(args.vocab_size)
    model = model.to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    shape = (args.batch_size, args.seq_len)
    src = torch.randint(4, args.vocab_size, shape, device=device)
    tgt_in = torch.randint(4, args.vocab_size, shape, device=device)
    tgt_out = torch.randint(4, args.vocab_size, shape, device=device)
    src_pad_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    def one_step():
        optimizer.zero_grad()
        logits = model(src, src_pad_mask, tgt_in) if args.from_src else model(src, tgt_in)
        loss = criterion(logits.reshape(-1, args.vocab_size), tgt_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    for _ in range(args.warmup):
        one_step()
    sync(device)

    times = []
    for _ in range(args.steps):
        start = time.perf_counter()
        one_step()
        sync(device)
        times.append(time.perf_counter() - start)

    per_step = statistics.median(times)
    batches = math.ceil(args.train_pairs / args.batch_size)
    min_per_epoch = per_step * batches / 60

    print(f"arch       : {args.arch}  [{'real model from src/' if args.from_src else 'stand-in'}]")
    print(f"device     : {device}")
    print(f"batch/seq  : {args.batch_size} x {args.seq_len}")
    print(f"sec/step   : median {per_step:.4f}  (min {min(times):.4f}, max {max(times):.4f})")
    print(f"min/epoch  : {min_per_epoch:.1f}")
    print(f"{args.epochs} epochs  : {min_per_epoch * args.epochs / 60:.1f} h")
    if not args.from_src:
        print("[stand-in model -- re-run with --from-src once src/ has the real one]")
    if not args.measure_seq_len:
        print("[seq_len is a guess, not measured -- see --measure_seq_len once real data exists]")


if __name__ == "__main__":
    main()
