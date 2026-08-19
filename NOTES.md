# Notes

Working notes: design decisions, mistakes made and fixed, anything worth remembering later.

## Environment

- Device: MPS (Apple GPU), no CUDA (confirmed via `src/env_check.py`).

## Design decisions

- Special token ids fixed at pad=0, bos=1, eos=2, unk=3 (`prepare_iwslt.py`, `SentencePieceTrainer.train`) - baked into the trained model, must match wherever it's loaded later.
- SentencePiece trained only on the filtered train split, per language, never retrained on dev/test - avoids leaking subword frequency into the vocabulary.
- Length filter (`--max_words`) applies to train only; dev/test stay unfiltered since evaluation has to handle real sentence lengths.
- Toy corpus reuses the real SentencePiece models (`data/processed/spm/`) instead of training a separate tiny tokenizer - the tokenizer is frozen and shared everywhere, not re-trained per dataset.

## Bugs found and fixed

- SentencePiece round-trip test caught silent data loss: default `character_coverage=0.9995` drops the rarest characters to `<unk>` on encode, unrecoverable on decode (e.g. "Ả" in "Ả Rập"). Fixed with `character_coverage=1.0` in `train_sentencepiece`.
- `collate_fn` padded `tgt_in` and `tgt_out` separately (two independent calls) - looks equivalent to padding once and slicing, but at the real-content/padding boundary it silently replaces eos in `tgt_out` with padding, breaking `tgt_in[:,1:] == tgt_out[:,:-1]`. Fixed by padding the full `tgt` sequence once, then slicing both from that tensor.
