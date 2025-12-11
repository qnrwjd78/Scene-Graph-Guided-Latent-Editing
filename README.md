# Scene-Graph Guided Latent Editing (CLIP + Stable Diffusion)

두 단계로 구성된 간단한 파이프라인입니다.
- **Stage 1 (새로 작성)**: CLIP 텍스트 임베딩을 입력으로 받아 트리플릿 GCN으로 그래프 임베딩을 학습하고, 이를 CLIP 이미지 임베딩과 대조학습으로 정렬합니다.
- **Stage 2 (갱신)**: 학습된 그래프 임베딩을 Stable Diffusion UNet의 cross-attention에 IP-Adapter 스타일로 주입하여, 캡션 + scene graph 이중 조건으로 이미지를 생성/편집합니다.

## 구조
- `stage1/`
  - `models/gcn_clip.py`: CLIP 정렬용 트리플릿 GCN + 글로벌 프로젝터.
  - `scripts/train_gcn_clip.py`: CLIP 텍스트 임베딩을 이용한 그래프-이미지 대조학습 스크립트.
- `stage2/`
  - `models/graph_adapter.py`: 트리플릿 GCN + graph K/V를 cross-attn에 추가하는 어댑터(IP-Adapter 스타일).
  - `scripts/train_adapter.py`: 캡션 + 그래프 병렬 조건으로 SD UNet 어댑터 학습(UNet/Text/VAE는 고정).
  - `scripts/sample_edit.py`: 캡션 + 사전 인코딩된 그래프를 넣어 샘플 생성.

## Stage 1: CLIP 그래프 인코더 학습
데이터 포맷(torch 저장 리스트):
```python
{
  "node_texts": List[str],
  "rel_texts": List[str],
  "triples": LongTensor [T,3] (subj_idx, rel_idx, obj_idx),
  "image_emb": Tensor [clip_dim]  # CLIP 이미지 임베딩(사전 계산)
}
```
학습 예시:
```bash
python stage1/scripts/train_gcn_clip.py --data_path data/stage1_graphs.pt --epochs 5
```
손실: 그래프 글로벌 임베딩과 CLIP 이미지 임베딩 간의 양방향 InfoNCE(temperature 사용).

## Stage 2: 그래프 어댑터 + Stable Diffusion
데이터 포맷(torch 저장 리스트):
```python
{
  "pixel_values": Tensor [3,H,W]  # [-1,1] 범위로 정규화된 이미지
  "caption": str,
  "node_feats": Tensor [N_nodes, node_dim],  # CLIP/VLM에서 얻은 노드 임베딩
  "rel_feats": Tensor [N_rels, rel_dim],     # 관계 임베딩
  "triples": LongTensor [T,3],
  "obj_to_img": LongTensor [N_nodes]
}
```
학습 예시:
```bash
python stage2/scripts/train_adapter.py --data_path data/stage2_pairs.pt --epochs 10
```

생성 예시(단일 그래프 torch 파일):
```bash
python stage2/scripts/sample_edit.py --graph_path data/sample_graph.pt --caption "a cat sits on a chair"
```

## 요구사항
- `torch`
- `transformers`
- `diffusers`
- (옵션) GPU CUDA 환경
