#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints
source .venv/bin/activate
DATA=data/processed


# RNN + Xavier - one flag changed from the RNN baseline, everything else
# (12 epochs, batch 32, lr) held fixed.
python src/train.py --arch rnn \
  --train_src $DATA/train.vi --train_tgt $DATA/train.en \
  --dev_src $DATA/dev.vi --dev_tgt $DATA/dev.en \
  --src_spm $DATA/spm/src_spm.model --tgt_spm $DATA/spm/tgt_spm.model \
  --epochs 12 --batch_size 32 --xavier_init \
  --save_dir checkpoints/rnn_xavier \
  > logs/train_rnn_xavier.log 2>&1