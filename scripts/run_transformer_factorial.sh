#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints
source .venv/bin/activate
DATA=data/processed

# Cell 1 (flat lr, default init) already trained as checkpoints/transformer_baseline.
# Cell 2 (flat lr, Xavier) already trained as checkpoints/transformer_xavier.
# The 2 remaining cells run sequentially, not in parallel, so neither slows the other.

python src/train.py --arch transformer \
  --train_src $DATA/train.vi --train_tgt $DATA/train.en \
  --dev_src $DATA/dev.vi --dev_tgt $DATA/dev.en \
  --src_spm $DATA/spm/src_spm.model --tgt_spm $DATA/spm/tgt_spm.model \
  --epochs 12 --batch_size 32 --warmup_steps 4000 \
  --save_dir checkpoints/transformer_warmup \
  > logs/train_transformer_warmup.log 2>&1

python src/train.py --arch transformer \
  --train_src $DATA/train.vi --train_tgt $DATA/train.en \
  --dev_src $DATA/dev.vi --dev_tgt $DATA/dev.en \
  --src_spm $DATA/spm/src_spm.model --tgt_spm $DATA/spm/tgt_spm.model \
  --epochs 12 --batch_size 32 --warmup_steps 4000 --xavier_init \
  --save_dir checkpoints/transformer_warmup_xavier \
  > logs/train_transformer_warmup_xavier.log 2>&1
