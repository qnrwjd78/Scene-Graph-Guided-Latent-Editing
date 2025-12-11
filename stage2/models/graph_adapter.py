import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def _segment_reduce(x: torch.Tensor, indices: torch.Tensor, num_segments: int, reduce: str = "mean") -> torch.Tensor:
    """
    Minimal scatter-based segment reduction that avoids extra deps.
    x: (N, C) tensor to aggregate
    indices: (N,) long tensor with segment ids in [0, num_segments)
    """
    out = torch.zeros(num_segments, x.size(-1), device=x.device, dtype=x.dtype)
    out.scatter_add_(0, indices.unsqueeze(-1).expand_as(x), x)

    if reduce == "sum":
        return out

    if reduce == "mean":
        counts = torch.zeros(num_segments, device=x.device, dtype=x.dtype)
        ones = torch.ones_like(indices, dtype=x.dtype)
        counts.scatter_add_(0, indices, ones)
        counts = counts.clamp(min=1).unsqueeze(-1)
        return out / counts

    raise ValueError(f"Unsupported reduce '{reduce}'")


class GraphTripletGCNLayer(nn.Module):
    """
    Simple triplet-aware GCN layer:
    - message from subject -> object and object -> subject
    - relation embedding acts as a directional bias on both directions
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, node_states: torch.Tensor, rel_states: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
        """
        node_states: (N_nodes, H)
        rel_states:  (N_rels, H)
        triples:     (N_triples, 3) with columns (subj_idx, rel_idx, obj_idx)
        """
        if triples.numel() == 0:
            return node_states

        subj_idx, rel_idx, obj_idx = triples.unbind(dim=1)

        # Messages: subject + relation -> object, and object + relation -> subject
        msg_subj_to_obj = node_states[subj_idx] + rel_states[rel_idx]
        msg_obj_to_subj = node_states[obj_idx] + rel_states[rel_idx]

        agg = torch.zeros_like(node_states)
        agg.scatter_add_(0, obj_idx.unsqueeze(-1).expand_as(msg_subj_to_obj), msg_subj_to_obj)
        agg.scatter_add_(0, subj_idx.unsqueeze(-1).expand_as(msg_obj_to_subj), msg_obj_to_subj)

        h = torch.cat([node_states, agg], dim=-1)
        h = self.proj(h)
        h = self.act(h)
        h = self.dropout(h)
        # Residual connection to stabilize training
        return node_states + h


class GraphTripletGCN(nn.Module):
    """
    GCN encoder for scene graphs described by triplets.
    Assumes node/relation features are already CLIP-aligned (provided by VLM or pre-encoder).
    """

    def __init__(
        self,
        node_dim: int = 512,
        rel_dim: int = 512,
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
        """
        Args:
            node_feats: (N_nodes, D_node) flattened across batch
            rel_feats: (N_rels, D_rel)
            triples: (N_triples, 3) indices into node_feats/rel_feats (already offset for batch)
            obj_to_img: (N_nodes,) mapping each node to its image index. If None, assumes single graph.
        Returns:
            node_states: (N_nodes, H)
            global_states: (B, H) pooled per image using mean
        """
        if obj_to_img is None:
            obj_to_img = torch.zeros(node_feats.size(0), device=node_feats.device, dtype=torch.long)

        node_states = self.node_in(node_feats)
        rel_states = self.rel_in(rel_feats)

        for layer in self.layers:
            node_states = layer(node_states, rel_states, triples)

        node_states = self.node_norm(node_states)
        rel_states = self.rel_norm(rel_states)

        batch_size = int(obj_to_img.max().item()) + 1 if obj_to_img.numel() > 0 else 1
        global_states = _segment_reduce(node_states, obj_to_img, num_segments=batch_size, reduce="mean")
        return node_states, global_states


class GraphCrossAttnProcessor(nn.Module):
    """
    Cross-attention processor that appends graph key/value pairs to the usual text condition.
    Modeled after IP-Adapter style processors but kept minimal and dependency-free.
    """

    def __init__(self, graph_tokens: Optional[torch.Tensor] = None, gate_init: float = 1.0):
        super().__init__()
        # Gate allows scaling the graph contribution; learnable to let training tune the mix.
        self.graph_gate = nn.Parameter(torch.tensor(gate_init))
        self.graph_tokens: Optional[torch.Tensor] = graph_tokens

    def set_graph_tokens(self, graph_tokens: Optional[torch.Tensor]):
        self.graph_tokens = graph_tokens

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if self.graph_tokens is not None:
            graph_tokens = self.graph_tokens.to(hidden_states.dtype)
            # Ensure batch dimension matches; if a single graph is provided, broadcast it.
            if graph_tokens.size(0) == 1 and hidden_states.size(0) > 1:
                graph_tokens = graph_tokens.expand(hidden_states.size(0), -1, -1)

            graph_tokens = graph_tokens * self.graph_gate.tanh()  # bounded gate
            g_key = attn.to_k(graph_tokens)
            g_value = attn.to_v(graph_tokens)

            key = torch.cat([key, g_key], dim=1)
            value = torch.cat([value, g_value], dim=1)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class GraphAdapter(nn.Module):
    """
    End-to-end graph branch:
    - GCN over triplets with CLIP-aligned node/relation inputs
    - Projection into UNet cross-attention space (IP-Adapter style)
    - Helper to register custom attention processors on attn2 blocks
    """

    def __init__(
        self,
        node_dim: int = 512,
        rel_dim: int = 512,
        hidden_dim: int = 768,
        cross_attention_dim: int = 768,
        gcn_layers: int = 3,
        dropout: float = 0.0,
        max_graph_tokens: int = 32,
        gate_init: float = 1.0,
    ):
        super().__init__()
        self.gcn = GraphTripletGCN(
            node_dim=node_dim,
            rel_dim=rel_dim,
            hidden_dim=hidden_dim,
            num_layers=gcn_layers,
            dropout=dropout,
        )
        self.token_proj = nn.Linear(hidden_dim, cross_attention_dim)
        self.global_proj = nn.Linear(hidden_dim, cross_attention_dim)
        self.max_graph_tokens = max_graph_tokens
        self.gate_init = gate_init

    def _pad_graph_tokens(self, tokens: torch.Tensor, obj_to_img: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Pads variable-length node tokens to a dense (B, max_tokens, C) tensor.
        """
        device = tokens.device
        out = torch.zeros(batch_size, self.max_graph_tokens, tokens.size(-1), device=device, dtype=tokens.dtype)
        for b in range(batch_size):
            mask = obj_to_img == b
            cur = tokens[mask]
            length = min(cur.size(0), self.max_graph_tokens)
            if length > 0:
                out[b, :length] = cur[:length]
        return out

    def encode_graph(
        self,
        node_feats: torch.Tensor,
        rel_feats: torch.Tensor,
        triples: torch.Tensor,
        obj_to_img: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            graph_tokens: (B, max_graph_tokens, cross_attention_dim)
            graph_global: (B, cross_attention_dim) pooled per image
        """
        node_states, global_states = self.gcn(node_feats, rel_feats, triples, obj_to_img)
        obj_to_img = obj_to_img if obj_to_img is not None else torch.zeros(node_states.size(0), device=node_states.device, dtype=torch.long)
        batch_size = int(obj_to_img.max().item()) + 1 if obj_to_img.numel() > 0 else 1

        token_states = self.token_proj(node_states)
        graph_tokens = self._pad_graph_tokens(token_states, obj_to_img, batch_size)

        graph_global = self.global_proj(global_states)
        return graph_tokens, graph_global

    def build_graph_processors(self, graph_tokens: torch.Tensor):
        """
        Construct a processor instance with the provided graph tokens.
        """
        return GraphCrossAttnProcessor(graph_tokens=graph_tokens, gate_init=self.gate_init)

    def apply_to_unet(self, unet, graph_tokens: torch.Tensor):
        """
        Register graph-aware cross-attention processors on all attn2 blocks of the UNet.
        Keeps existing processors for attn1 (self-attn) intact.
        """
        processor = self.build_graph_processors(graph_tokens)
        attn_procs = {}
        for name, proc in unet.attn_processors.items():
            if name.endswith("attn2.processor"):
                attn_procs[name] = processor
            else:
                attn_procs[name] = proc
        unet.set_attn_processor(attn_procs)

    @staticmethod
    def update_graph_tokens(unet, graph_tokens: torch.Tensor):
        """
        Update graph tokens after they are refreshed without rebuilding the processor map.
        """
        for proc in unet.attn_processors.values():
            if isinstance(proc, GraphCrossAttnProcessor):
                proc.set_graph_tokens(graph_tokens)
