import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return x + self.net(x)

class SceneGraphEmbedder(nn.Module):
    def __init__(self, gcn_dim=512, model_dim=768, max_objs=30, max_seq_len=77, num_layers=3):
        super().__init__()
        
        # 1. Adapter (Translator): GCN(512) -> SD(768)
        self.input_proj = nn.Sequential(
            nn.Linear(gcn_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        
        # New Adapter Structure: Linear -> GELU -> Linear (No internal LN)
        self.adapter = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )
        
        # Final LayerNorm (Stabilize Output)
        self.final_ln = nn.LayerNorm(model_dim)
        
        # --- 3-Layer Hybrid Embeddings ---
        
        # (2) Token Type: 0(Obj), 1(Rel)
        self.type_emb = nn.Embedding(2, model_dim)
        
        # (3) Self Index: Object ID
        self.self_idx_emb = nn.Embedding(max_objs, model_dim)
        
        # (4) Pointer: Subject/Object ID pointer
        self.sub_ptr_emb = nn.Embedding(max_objs, model_dim)
        self.obj_ptr_emb = nn.Embedding(max_objs, model_dim)
        
        # Initialization (IP-Adapter Style: Truncated Normal std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                # Revert to std=0.02 (Small Init), but we will SCALE it by 30.0 in forward
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr):
        """
        Args:
            gcn_vectors: (Batch, Seq, 512)
            token_types: (Batch, Seq) - 0(Obj), 1(Rel)
            obj_idx:     (Batch, Seq) - Object ID (0 for Rel)
            sub_ptr:     (Batch, Seq) - Subject ID (0 for Obj)
            obj_ptr:     (Batch, Seq) - Object ID (0 for Obj)
        """

        # [Step 1] Project GCN to Model Dim
        x = self.input_proj(gcn_vectors)
        
        # [Step 2] Add Embeddings BEFORE Adapter (Scaled by 30.0)
        scale = 30.0
        
        # (2) Type Embedding
        x = x + (self.type_emb(token_types) * scale)
        
        # (3) Object: Self Index
        is_obj = (token_types == 0)
        obj_emb = self.self_idx_emb(obj_idx.clamp(max=self.self_idx_emb.num_embeddings - 1))
        x = torch.where(is_obj.unsqueeze(-1), x + (obj_emb * scale), x)
            
        # (4) Relation: Pointer
        is_rel = (token_types == 1)
        sub_emb = self.sub_ptr_emb(sub_ptr.clamp(max=self.sub_ptr_emb.num_embeddings - 1))
        obj_emb = self.obj_ptr_emb(obj_ptr.clamp(max=self.obj_ptr_emb.num_embeddings - 1))
        x = torch.where(is_rel.unsqueeze(-1), x + (sub_emb * scale) + (obj_emb * scale), x)
        
        # [Step 3] Adapter (Linear -> GELU -> Linear)
        x = self.adapter(x)
        
        # [Step 4] Final LayerNorm (Stabilize Output)
        x = self.final_ln(x)

        return x
