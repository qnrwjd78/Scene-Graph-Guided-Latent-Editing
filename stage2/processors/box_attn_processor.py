import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

class BoxGuidedCrossAttnProcessor:
    def __init__(self, target_box=None):
        self.target_box = target_box # [x1, y1, x2, y2] normalized 0-1

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

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # Box Masking Logic
        if self.target_box is not None:
            # attention_probs: (Batch*Head, Height*Width, Text_Seq_Len)
            # We want to mask the attention map such that the object token only attends to the box region.
            # Assuming the object token is the one we are interested in.
            # However, standard CrossAttn is (Query=Image, Key=Text).
            # So attention_probs[b, i, j] is how much pixel i attends to token j.
            
            # We need to know which token corresponds to the object. 
            # In this pipeline, we might be passing the object embedding as the condition.
            # If encoder_hidden_states contains [background_emb, object_emb], we want object_emb to only affect box pixels.
            
            # Let's assume encoder_hidden_states is just the object embedding (or sequence of them).
            # And we want to restrict its influence to the box.
            
            h = w = int(sequence_length ** 0.5)
            
            # Create box mask
            # target_box: [x1, y1, x2, y2]
            x1, y1, x2, y2 = self.target_box
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = int(y1 * h), int(y2 * h)
            
            mask = torch.zeros((h, w), device=attention_probs.device)
            mask[y1:y2, x1:x2] = 1.0
            mask = mask.view(-1) # (H*W)
            
            # Apply mask: If pixel is OUTSIDE box, set attention score to -inf (or very low)
            # But wait, CrossAttn is "Pixel attends to Text".
            # If we want "Text only affects Box Pixels", it means "Pixels outside box should NOT attend to Text".
            # So for pixels outside box, attention to this token should be 0.
            
            # attention_probs shape: (B*Head, H*W, SeqLen)
            # We broadcast mask to (1, H*W, 1)
            mask = mask.view(1, -1, 1)
            
            # If mask is 0 (outside), we want attention to be low? 
            # Actually, if we have multiple tokens, we only want to mask specific tokens.
            # But here we simplify: assume the condition is THE object.
            # So we mask the spatial dimension.
            
            # If we set attention scores to -inf for outside pixels, they will attend to nothing? 
            # Softmax will make them attend to something else if available, or uniform if all -inf.
            # Usually we want them to attend to a "background" token or null token.
            # If we only provide object embedding, this is tricky.
            
            # GLIGEN approach:
            # They add a gated self-attention or modify cross-attention.
            # Here, let's just multiply the attention weights by the mask (soft masking) 
            # or add a large negative value to scores before softmax (hard masking).
            # Since get_attention_scores usually does softmax internally or returns raw scores?
            # In Diffusers, get_attention_scores usually returns Softmaxed probs? 
            # No, usually it returns raw scores if we look at implementation, but here we are using the processor.
            # Wait, `get_attention_scores` in `Attention` class does `baddbmm` then `softmax`.
            # So `attention_probs` here are already probabilities (0-1).
            
            # So we can just multiply by mask.
            # Pixels outside box will have 0 attention to this token.
            attention_probs = attention_probs * mask
            
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
