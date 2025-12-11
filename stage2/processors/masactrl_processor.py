import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from diffusers.models.attention_processor import Attention
except ImportError:
    from diffusers.models.attention import CrossAttention as Attention

class MasaCtrlSelfAttnProcessor:
    def __init__(self, attn_store, is_inversion=False):
        self.attn_store = attn_store
        self.is_inversion = is_inversion
        # boxes should be set before forward pass during generation
        self.old_box = None 
        self.new_box = None
        self.warping_fn = None # Function to warp features

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # MasaCtrl Logic
        if self.is_inversion:
            # Store K, V
            self.attn_store.append(key.detach(), value.detach())
        else:
            # Retrieve and Warp
            stored_k, stored_v = self.attn_store.pop()
            
            if stored_k is None or stored_v is None:
                # Fallback if store is empty (e.g. first run or mismatch)
                # Just use current key/value (standard attention)
                pass
            else:
                stored_k = stored_k.to(query.device)
                stored_v = stored_v.to(query.device)
                if self.old_box is not None and self.new_box is not None and self.warping_fn is not None:
                    # Check if we need to warp (e.g. IoU check or just always warp if boxes provided)
                    # Here we assume we always warp if boxes are present
                    
                    # Reshape for warping: (B*Head, Seq, Dim) -> (B, Head, H, W, Dim)
                    # We need to know H, W. usually sqrt(Seq)
                    h = w = int(sequence_length ** 0.5)
                    
                    # Warp K
                    warped_k = self.warping_fn(stored_k, self.old_box, self.new_box, h, w)
                    # Warp V
                    warped_v = self.warping_fn(stored_v, self.old_box, self.new_box, h, w)
                    
                    key = warped_k
                    value = warped_v
                else:
                    # Use stored features as is (reconstruction)
                    key = stored_k
                    value = stored_v

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
