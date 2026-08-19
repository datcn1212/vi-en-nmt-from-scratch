# Evaluation protocol

Locked in before any results exist, to prevent unconsciously bending the protocol once a model's performance is visible.

- Comparing configurations (hyperparameters, architectures, attention variants, anything) is judged on the dev set only.
- The test set is scored exactly once, at the end, after everything else is already decided.
- Whenever two models are compared, they use the same decoding settings (same beam width, same max length) - never beam search for one and greedy for the other.

## Decisions log

- Vocab size fixed at 8000, SentencePiece BPE, one model per language, trained on the filtered train split only, then frozen (dev/test only ever encoded with it, never retrained).
