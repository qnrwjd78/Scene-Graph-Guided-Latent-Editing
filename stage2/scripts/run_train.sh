#!/bin/bash

# Training script for Stage 2 Adapter with Dual Input Strategy

# Parameters
# --adapter_lr: Learning rate for the Adapter (Train from scratch) -> 1e-3
# --lora_lr: Learning rate for LoRA (Fine-tuning) -> 1e-4
# --max_train_samples: Limit training samples for debugging (e.g., 1000). Remove for full training.
# --max_val_samples: Limit validation samples (e.g., 100).
# --batch_size: Batch size per GPU.
# --gradient_accumulation_steps: Accumulate gradients to simulate larger batch size.

# Run from the project root
python stage2/scripts/train_adapter.py \
    --epochs 10 \
    --batch_size 2 \
    --save_dir "checkpoints/ver_3/" \
    --gradient_accumulation_steps 8 \
    --lora_lr 2e-4 \
    --adapter_lr 2e-4 \
    --warmup_epochs 0 \
    --scheduler_warmup_steps 10 \
    --condition_dropout_prob 0.25 \
    --dataset vg \
    --resume "checkpoints/ver_2/adapter_epoch_3.pth" \
