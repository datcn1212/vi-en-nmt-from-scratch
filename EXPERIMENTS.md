# Experiment log

Short log of what ran, why, and what happened, not reconstructed afterward.

## 2026-08-19 - how slow is each architecture, roughly

Why: want a rough minutes/epoch number before writing any real code, so later plans aren't guesswork.

Ran (throwaway models with random data, same rough size as the real ones):
    python scripts/benchmark.py --arch rnn
    python scripts/benchmark.py --arch transformer

Result:
    RNN:         ~9 min/epoch  (~1.8h for 12 epochs)
    Transformer: ~2 min/epoch  (~0.4h for 12 epochs)

Note: this used a guessed sentence length (30 tokens). Real batches pad to the longest sentence in the batch, which will likely be longer, so these are probably underestimates - worth re-checking once real tokenized data exists.
