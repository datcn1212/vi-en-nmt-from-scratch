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

## Data pipeline: Vocab, Dataset, collate, toy corpus

Why: shared pipeline both architectures will consume identically, verified on a small hand-written corpus before any model code exists.

Ran:
    python -m pytest tests/test_data.py -v

Result: found a real bug on first run - tgt_in and tgt_out were padded separately (two independent calls), which looks equivalent to padding once and slicing but is not: at the real-content/padding boundary, eos in tgt_out was replaced by padding instead of landing where tgt_in[1:] expected it, breaking the shift invariant. Fixed by padding the full tgt sequence once, then slicing tgt_in = tgt[:, :-1] and tgt_out = tgt[:, 1:] from the same tensor. All 4 tests pass, reran 5x; also checked the invariant by hand on the full 40-sentence train batch (widest length spread).

## RNN model: BahdanauAttention + RNNSeq2Seq, overfit and masking gates

Why: shape/mask bugs are cheap to catch on 20 toy sentences before they cost real training time on the full corpus.

Ran:
    python -m pytest tests/test_rnn.py -v

Result:
    test_overfit_20: loss 8.99 -> 0.0031 over 300 steps (dropout off), well under the 0.5 pass bar
    test_autoregressive_generation: 20/20 exact match, greedy decode against the overfit model
    test_attention_ignores_padding: passes with masking in place
    negative control: masked_fill in BahdanauAttention commented out, rerun -
    max attention weight on padding rose to 0.0522 (near-uniform over ~19 positions,
    as expected with no masking) - test correctly fails. Restored, full suite passes again.

## train.py pipeline check on real data

Why: catch CLI/device/checkpoint wiring bugs on a tiny slice before committing to a multi-hour run.

Ran:
    python src/train.py --arch rnn --train_src data/processed/train.vi --train_tgt data/processed/train.en \
      --dev_src data/processed/dev.vi --dev_tgt data/processed/dev.en \
      --src_spm data/processed/spm/src_spm.model --tgt_spm data/processed/spm/tgt_spm.model \
      --save_dir checkpoints/smoketest --epochs 2 --batch_size 32 --max_examples 64

Result:
    epoch 1 train_loss 8.9548 dev_loss 8.8628 lr 0.001000
    epoch 2 train_loss 8.6584 dev_loss 8.6109 lr 0.001000
    checkpoint written with all required fields (model_state_dict, arch, hyperparams,
    src_vocab_size, tgt_vocab_size, pad_id) - loaded back and checked by hand.

## RNN baseline: 12 epochs, full corpus

Why: first real result for this architecture - the dev loss curve decides which checkpoint gets used, and whether it overfits the way the toy-scale runs hinted at.

Ran:
    bash scripts/run_rnn_baseline.sh

Result:
    epoch 1   train_loss 3.8438  dev_loss 3.1085
    epoch 2   train_loss 2.8669  dev_loss 2.9117
    epoch 3   train_loss 2.5702  dev_loss 2.8329
    epoch 4   train_loss 2.3943  dev_loss 2.8068
    epoch 5   train_loss 2.2727  dev_loss 2.8050   (best - checkpoint)
    epoch 6   train_loss 2.1815  dev_loss 2.8124
    epoch 7   train_loss 2.1086  dev_loss 2.8154
    epoch 8   train_loss 2.0503  dev_loss 2.8225
    epoch 9   train_loss 2.0014  dev_loss 2.8326
    epoch 10  train_loss 1.9607  dev_loss 2.8414
    epoch 11  train_loss 1.9240  dev_loss 2.8631
    epoch 12  train_loss 1.8925  dev_loss 2.8764

Dev loss falls through epoch 5, then rises through epoch 12, while train loss keeps falling the whole run - overfits past epoch 5. Checkpoint stopped updating after epoch 5. Total wall clock ~11.4h for 12 epochs (57 min/epoch average).
