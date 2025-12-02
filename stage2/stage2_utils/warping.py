import torch
import torch.nn.functional as F

def warp_tensor(tensor, old_box, new_box, h, w):
    """
    tensor: (Batch*Head, SeqLen, Dim) or (Batch, Head, SeqLen, Dim)
    old_box: [x1, y1, x2, y2] normalized
    new_box: [x1, y1, x2, y2] normalized
    h, w: spatial dimensions
    """
    # Handle input shape
    if tensor.dim() == 3:
        bh, seq, dim = tensor.shape
        # Assume Batch=1 for simplicity or handle properly
        # We need to separate Batch and Head to do spatial operations correctly if we want to be precise,
        # but usually warping is spatial so B*Head is fine as a batch dimension.
        b_dim = bh
    else:
        b, head, seq, dim = tensor.shape
        b_dim = b * head
        tensor = tensor.view(b_dim, seq, dim)
        
    # Reshape to spatial
    tensor_spatial = tensor.view(b_dim, h, w, dim).permute(0, 3, 1, 2) # (B, Dim, H, W)
    
    # Parse boxes
    ox1, oy1, ox2, oy2 = [int(v * s) for v, s in zip(old_box, [w, h, w, h])]
    nx1, ny1, nx2, ny2 = [int(v * s) for v, s in zip(new_box, [w, h, w, h])]
    
    # Clamp coordinates
    ox1, ox2 = max(0, ox1), min(w, ox2)
    oy1, oy2 = max(0, oy1), min(h, oy2)
    nx1, nx2 = max(0, nx1), min(w, nx2)
    ny1, ny2 = max(0, ny1), min(h, ny2)
    
    # Check valid crop
    if ox2 <= ox1 or oy2 <= oy1:
        return tensor # Invalid box, return original
        
    # Crop
    crop = tensor_spatial[:, :, oy1:oy2, ox1:ox2]
    
    # Resize to new box size
    new_h, new_w = ny2 - ny1, nx2 - nx1
    if new_h <= 0 or new_w <= 0:
        return tensor
        
    resized_crop = F.interpolate(crop, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # Paste
    # Create a canvas. 
    # Option 1: Start with zeros.
    # Option 2: Start with original tensor (copy background).
    # Usually for "Move", we want to move the object and fill the old place with background?
    # Or just paste the object onto the original feature map?
    # If we paste onto original, we have two cats.
    # We should probably start with a "cleaned" feature map or the original one.
    # For simplicity in this "Move" operation, let's paste onto the original, 
    # but ideally we should inpaint the old region. 
    # Since we don't have inpainting logic here, let's just paste.
    # (Advanced: Use attention mask to suppress old region)
    
    output = tensor_spatial.clone()
    
    # Optional: Zero out old region?
    # output[:, :, oy1:oy2, ox1:ox2] = 0 
    
    output[:, :, ny1:ny2, nx1:nx2] = resized_crop
    
    # Reshape back
    output = output.permute(0, 2, 3, 1).view(b_dim, h*w, dim)
    
    return output
