#!/bin/bash

# Scene Graph Guided Latent Editing - Sampling Script

# Common Parameters
CHECKPOINT="checkpoints/ver_4/adapter_epoch_16.pth"
OUTPUT_DIR="outputs"
IMAGE_IDX=2000
DATASET="vg_clevr"

# 1. Simple Generation (Random Noise -> Image)
# 학습된 Adapter가 Scene Graph를 보고 이미지를 잘 그려내는지 확인합니다.
# --save_attn_map: 각 객체/관계가 이미지의 어디에 위치하는지 시각화하여 저장합니다.
echo "Running Simple Generation..."
python stage2/scripts/sample.py \
  --checkpoint_path $CHECKPOINT \
  --image_index $IMAGE_IDX \
  --save_attn_map \
  --dataset $DATASET \
  --output_dir "$OUTPUT_DIR/generation_ver_4/$IMAGE_IDX"

# 2. Editing (Inversion -> Edit)
# 원본 이미지를 Inversion한 뒤 복원(Reconstruction)하거나 편집합니다.
# --edit_type: reconstruction (복원), move_object (이동), replace_object (교체)
# --inversion_type: ddim (빠름), null_text (정확함)
# echo "Running Editing (Reconstruction)..."
# python stage2/scripts/sample_edit.py \
#   --checkpoint_path $CHECKPOINT \
#   --image_index 2000 \
#   --edit_type reconstruction \
#   --inversion_type ddim \
#   --save_attn_map \
#   --dataset $DATASET \
#   --output_dir "$OUTPUT_DIR/generation_ver_4/editing"
