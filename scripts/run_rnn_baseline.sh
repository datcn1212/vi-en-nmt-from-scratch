#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints
source .venv/bin/activate
DATA=data/processed
python src/train.py --arch rnn \
  --train_src $DATA/train.vi --train_tgt $DATA/train.en \
  --dev_src $DATA/dev.vi --dev_tgt $DATA/dev.en \
  --src_spm $DATA/spm/src_spm.model --tgt_spm $DATA/spm/tgt_spm.model \
  --save_dir checkpoints/rnn_baseline --epochs 12 --batch_size 32 \
  > logs/train_rnn_baseline.log 2>&1
