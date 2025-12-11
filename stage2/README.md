# Stage 2: Scene Graph Guided Latent Editing

This directory contains the implementation of the inference and editing stage using Stable Diffusion.

## Structure (경량화)
- **models/**
  - `graph_adapter.py`: 트리플릿 GCN + IP-Adapter 스타일 cross-attn 브리지.
- **scripts/**
  - `train_adapter.py`: 캡션+그래프 병렬 조건으로 어댑터 학습(SD 고정).
  - `sample_edit.py`: 캡션 + 사전 인코딩된 그래프로 샘플 생성.

## Usage

### 1. Train Adapter (caption + scene graph)
- Prepare a torch file (list of dicts) where each dict has:
  `pixel_values` (tensor scaled to [-1,1]), `caption` (str),
  `node_feats`, `rel_feats`, `triples`, `obj_to_img`.
- Then run:
```bash
python scripts/train_adapter.py --data_path path/to/data.pt --epochs 10
```

### 2. Generate with caption + graph
Save a single-graph torch file containing `node_feats`, `rel_feats`, `triples`, `obj_to_img`, then:
```bash
python scripts/sample_edit.py --graph_path path/to/graph.pt --caption "a cat sits on a chair"
```

## Requirements
- `diffusers`
- `transformers`
- `torch`
- `diffusers`
- `transformers`
- `torch`
