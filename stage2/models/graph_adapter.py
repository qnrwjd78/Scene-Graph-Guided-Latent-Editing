import torch
import torch.nn as nn

class SceneGraphEmbedder(nn.Module):
    def __init__(self, gcn_dim=512, model_dim=768, max_objs=30, max_seq_len=77):
        super().__init__()
        
        # 1. Adapter (Translator): GCN(512) -> SD(768)
        self.adapter = nn.Sequential(
            nn.Linear(gcn_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim)  # Add LayerNorm for stability
        )
        
        # --- 3-Layer Hybrid Embeddings ---
        
        # (2) Token Type: 0(Obj), 1(Rel)
        self.type_emb = nn.Embedding(2, model_dim)
        
        # (3) Self Index: Object ID
        self.self_idx_emb = nn.Embedding(max_objs, model_dim)
        
        # (4) Pointer: Subject/Object ID pointer
        self.sub_ptr_emb = nn.Embedding(max_objs, model_dim)
        self.obj_ptr_emb = nn.Embedding(max_objs, model_dim)
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr):
        """
        Args:
            gcn_vectors: (Batch, Seq, 512)
            token_types: (Batch, Seq) - 0(Obj), 1(Rel)
            obj_idx:     (Batch, Seq) - Object ID (0 for Rel)
            sub_ptr:     (Batch, Seq) - Subject ID (0 for Obj)
            obj_ptr:     (Batch, Seq) - Object ID (0 for Obj)
        """

        # [Step 1] Adapter (Clean Input for UNet)
        x_clean = self.adapter(gcn_vectors)
        
        # [Step 2] Hybrid Embeddings Injection (Mixed Input for LoRA)
        x_mixed = x_clean.clone()
        
        # (2) Type Embedding
        x_mixed = x_mixed + self.type_emb(token_types)
        
        # (3) Object: Self Index
        is_obj = (token_types == 0)
        # We use masking to apply embeddings only where appropriate
        # obj_idx is (Batch, Seq)
        obj_emb = self.self_idx_emb(obj_idx.clamp(max=self.self_idx_emb.num_embeddings - 1))
        x_mixed = torch.where(is_obj.unsqueeze(-1), x_mixed + obj_emb, x_mixed)
            
        # (4) Relation: Pointer
        is_rel = (token_types == 1)
        sub_emb = self.sub_ptr_emb(sub_ptr.clamp(max=self.sub_ptr_emb.num_embeddings - 1))
        obj_emb = self.obj_ptr_emb(obj_ptr.clamp(max=self.obj_ptr_emb.num_embeddings - 1))
        
        x_mixed = torch.where(is_rel.unsqueeze(-1), x_mixed + sub_emb + obj_emb, x_mixed)

        return x_clean, x_mixed
