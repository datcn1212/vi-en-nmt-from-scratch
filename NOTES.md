# Notes

Working notes: design decisions, mistakes made and fixed, anything worth remembering later.

## Environment

- Device: MPS (Apple GPU), no CUDA (confirmed via `src/env_check.py`).

## Design decisions

- Special token ids fixed at pad=0, bos=1, eos=2, unk=3 (`prepare_iwslt.py`, `SentencePieceTrainer.train`) - baked into the trained model, must match wherever it's loaded later.
- SentencePiece trained only on the filtered train split, per language, never retrained on dev/test - avoids leaking subword frequency into the vocabulary.
- Length filter (`--max_words`) applies to train only; dev/test stay unfiltered since evaluation has to handle real sentence lengths.
- Toy corpus reuses the real SentencePiece models (`data/processed/spm/`) instead of training a separate tiny tokenizer - the tokenizer is frozen and shared everywhere, not re-trained per dataset.
- `test_overfit_20` runs with `dropout=0.0` - it checks memorization capacity and wiring, not generalization, and dropout noise would stop 20 sentences reaching near-zero loss in 300 steps.
- `test_attention_ignores_padding` checked for real discriminating power: `masked_fill` in `BahdanauAttention` temporarily commented out, rerun. Max attention weight on padding rose to 0.0522 (near-uniform over the ~19-position batch, as expected with no masking) against the < 0.05 bar - test fails correctly when masking is actually broken. Restored, both `test_rnn.py` tests pass again.
- `greedy_decode` doesn't call `model.eval()` internally, only `torch.no_grad()` (matching the spec literally) - mode switching is left to the caller, so it never has a surprising side effect on a model's train/eval state. Every call site in this repo sets `eval()` explicitly before decoding.
- Checkpoint `hyperparams` only stores training-loop knobs (arch, epochs, batch_size, lr, seed) - model architecture args (emb_dim, hidden_dim, dropout) aren't CLI-exposed in `train.py` yet, so they stay at `RNNSeq2Seq`'s own defaults. Anything that reloads a checkpoint should assume those same defaults until the CLI grows flags for them.

## Bugs found and fixed

- SentencePiece round-trip test caught silent data loss: default `character_coverage=0.9995` drops the rarest characters to `<unk>` on encode, unrecoverable on decode (e.g. "Ả" in "Ả Rập"). Fixed with `character_coverage=1.0` in `train_sentencepiece`.
- `collate_fn` padded `tgt_in` and `tgt_out` separately (two independent calls) - looks equivalent to padding once and slicing, but at the real-content/padding boundary it silently replaces eos in `tgt_out` with padding, breaking `tgt_in[:,1:] == tgt_out[:,:-1]`. Fixed by padding the full `tgt` sequence once, then slicing both from that tensor.
