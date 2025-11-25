# Stage 2: Scene Graph Guided Latent Editing

This directory contains the implementation of the inference and editing stage using Stable Diffusion.

## Structure

- **config/**: Configuration files.
- **models/**: 
  - `gcn_wrapper.py`: Wrapper for the Stage 1 Scene Graph Encoder.
  - `graph_adapter.py`: Trainable adapter to map graph embeddings to SD space.
- **processors/**: Custom attention processors for editing.
  - `masactrl_processor.py`: Handles feature warping and injection in Self-Attention.
  - `box_attn_processor.py`: Handles box-guided masking in Cross-Attention.
- **utils/**: Helper functions for inversion, warping, and data loading.
- **scripts/**:
  - `train_adapter.py`: Script to train the Graph Adapter.
  - `sample_edit.py`: Script to perform editing (Inversion -> Warping -> Generation).

## Usage

### 1. Train Adapter
First, train the adapter to align the Scene Graph Encoder with Stable Diffusion.
```bash
python scripts/train_adapter.py --epochs 10
```

### 2. Edit Image
Run the editing script with an input image.
```bash
python scripts/sample_edit.py --image_path path/to/image.jpg
```

## Requirements
- `diffusers`
- `transformers`
- `torch`
- `stage1` weights (pretrained/sip_vg.pt)
