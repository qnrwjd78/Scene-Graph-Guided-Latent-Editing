import torch
import torch.nn as nn
from typing import Optional, Tuple


class GraphTripletGCNLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_states: torch.Tensor, rel_states: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
        if triples.numel() == 0:
            return node_states

        subj_idx, rel_idx, obj_idx = triples.unbind(dim=1)

        msg_subj_to_obj = node_states[subj_idx] + rel_states[rel_idx]
        msg_obj_to_subj = node_states[obj_idx] + rel_states[rel_idx]

        agg = torch.zeros_like(node_states)
        agg.scatter_add_(0, obj_idx.unsqueeze(-1).expand_as(msg_subj_to_obj), msg_subj_to_obj)
        agg.scatter_add_(0, subj_idx.unsqueeze(-1).expand_as(msg_obj_to_subj), msg_obj_to_subj)

        h = torch.cat([node_states, agg], dim=-1)
        h = self.proj(h)
        h = self.act(h)
        h = self.dropout(h)
        return node_states + h


class GraphTripletGCN(nn.Module):
    def __init__(
        self,
        node_dim: int = 768,
        rel_dim: int = 768,
        hidden_dim: int = 768,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden_dim)
        self.rel_in = nn.Linear(rel_dim, hidden_dim)
        self.layers = nn.ModuleList([GraphTripletGCNLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.rel_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_feats: torch.Tensor,
        rel_feats: torch.Tensor,
        triples: torch.Tensor,
        obj_to_img: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if obj_to_img is None:
            obj_to_img = torch.zeros(node_feats.size(0), device=node_feats.device, dtype=torch.long)

        node_states = self.node_in(node_feats)
        rel_states = self.rel_in(rel_feats)

        for layer in self.layers:
            node_states = layer(node_states, rel_states, triples)

        node_states = self.node_norm(node_states)
        rel_states = self.rel_norm(rel_states)

        batch = int(obj_to_img.max().item()) + 1 if obj_to_img.numel() > 0 else 1
        global_states = torch.zeros(batch, node_states.size(-1), device=node_states.device, dtype=node_states.dtype)
        global_states.scatter_add_(0, obj_to_img.unsqueeze(-1).expand_as(node_states), node_states)

        counts = torch.zeros(batch, device=node_states.device, dtype=node_states.dtype)
        counts.scatter_add_(0, obj_to_img, torch.ones_like(obj_to_img, dtype=node_states.dtype))
        counts = counts.clamp(min=1).unsqueeze(-1)
        global_states = global_states / counts

        return node_states, global_states


class GraphSceneEncoder(nn.Module):
    """
    CLIP-aligned scene graph encoder.
    - node_feats / rel_feats are CLIP text features
    - outputs global graph embedding aligned to CLIP image/text space
    """

    def __init__(
        self,
        clip_dim: int = 768,
        hidden_dim: int = 768,
        gcn_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gcn = GraphTripletGCN(
            node_dim=clip_dim,
            rel_dim=clip_dim,
            hidden_dim=hidden_dim,
            num_layers=gcn_layers,
            dropout=dropout,
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, clip_dim),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        rel_feats: torch.Tensor,
        triples: torch.Tensor,
        obj_to_img: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, global_states = self.gcn(node_feats, rel_feats, triples, obj_to_img)
        graph_global = self.global_proj(global_states)
        return graph_global
