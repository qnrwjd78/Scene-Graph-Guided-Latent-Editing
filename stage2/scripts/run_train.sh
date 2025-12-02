#!/bin/bash

# Training script for Stage 2 Adapter with Dual Input Strategy

# Parameters
# --adapter_lr: Learning rate for the Adapter (Train from scratch) -> 1e-3
# --lr: Learning rate for LoRA (Fine-tuning) -> 1e-4
# --max_train_samples: Limit training samples for debugging (e.g., 1000). Remove for full training.
# --max_val_samples: Limit validation samples (e.g., 100).
# --batch_size: Batch size per GPU.
# --gradient_accumulation_steps: Accumulate gradients to simulate larger batch size.

# Run from the project root
python stage2/scripts/train_adapter.py \
    --epochs 200 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --lr 1e-4 \
    --adapter_lr 1e-3 \
    --save_dir "checkpoints/dual_input_v2" \
    --max_train_samples 1600 \
    --max_val_samples 100
