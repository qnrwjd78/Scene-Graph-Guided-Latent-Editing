#!/bin/bash

# Training script for Stage 1 (GCN Pre-training)

DATASET="vg_clevr" # or "vg"

# Set paths based on dataset
if [ "$DATASET" == "vg_clevr" ]; then
    TRAIN_DATA="datasets/vg_clevr/images"
    VOCAB_JSON="datasets/vg_clevr/vocab.json"
    TRAIN_H5="datasets/vg_clevr/train.h5"
    VAL_H5="datasets/vg_clevr/val.h5"
else
    TRAIN_DATA="datasets/vg/images"
    VOCAB_JSON="datasets/vg/vocab.json"
    TRAIN_H5="datasets/vg/train.h5"
    VAL_H5="datasets/vg/val.h5"
fi

echo "Training Stage 1 on $DATASET..."
echo "Images: $TRAIN_DATA"
echo "Vocab: $VOCAB_JSON"

# Generate timestamp for unique experiment name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run training
python stage1_trainer.py \
    --train_data $TRAIN_DATA \
    --vocab_json $VOCAB_JSON \
    --train_h5 $TRAIN_H5 \
    --val_h5 $VAL_H5 \
    --batch_size 32 \
    --epochs 10 \
    --save_frequency 1 \
    --logs logs/stage1_$DATASET \
    --name stage1_${DATASET}_${TIMESTAMP} \
    --workers 4
