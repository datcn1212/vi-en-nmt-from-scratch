# Experiment log

Short log of what ran, why, and what happened, not reconstructed afterward.

## How slow is each architecture, roughly

Why: rough minutes/epoch before writing real code, so plans aren't guesswork.

Ran (throwaway models, random data, real model size):
    python scripts/benchmark.py --arch rnn
    python scripts/benchmark.py --arch transformer

Result:
    RNN:         ~9 min/epoch  (~1.8h for 12 epochs)
    Transformer: ~2 min/epoch  (~0.4h for 12 epochs)

Note: sentence length was guessed (30 tokens); real batches pad longer, so likely underestimates - re-check once real tokenized data exists.

## Preprocessing: unescape, NFC, length filter, tokenizer

Why: clean aligned text before any model code, and a length-filter threshold grounded in the corpus's own distribution, not a copied number.

Ran:
    python src/prepare_iwslt.py --raw_dir data/raw --out_dir data/processed --max_words 150

Result:
    train word count p99 ~71-89 (en/vi) - 150 mainly cuts a long tail, not typical sentences
    dropped 95/133317 train pairs (0.07%)
    train/dev/test after filtering: 133222 / 1553 / 1268 pairs
    SentencePiece BPE, vocab 8000, trained on filtered train only
    subword max length (incl. bos/eos): 177, across all splits

Note: subword max (177) is the real sequence-length lower bound, not the word-based filter (150) - different units, don't conflate them later.

## Preprocessing tests: NFC, alignment, tokenizer round-trip

Why: verify preprocessing invariants hold before trusting the output for anything downstream.

Ran:
    python -m pytest tests/test_preprocess.py -v

Result: found a real bug on first run - SentencePiece's default character_coverage (0.9995) silently mapped a rare capitalized letter ("Ả" in "Ả Rập") to <unk>, unrecoverable on decode. Fixed with character_coverage=1.0, retrained, all 3 tests pass, reran 5x with different random samples to be sure.
