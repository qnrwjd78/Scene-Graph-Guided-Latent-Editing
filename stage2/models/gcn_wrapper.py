import torch
import torch.nn as nn
import sys
import os
import json

# Add stage1 to path to import sgCLIP
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'stage1'))

from sgCLIP.model import sgCLIP

class GCNWrapper(nn.Module):
    def __init__(self, vocab_file, checkpoint_path=None, embed_dim=512, graph_width=512, num_graph_layer=5):
        super().__init__()
        
        with open(vocab_file, 'r') as f:
            graph_vocab = json.load(f)
            
        model_cfg = {
            "graph_cfg": {
                "layers": num_graph_layer,
                "width": graph_width,
            },
            "embed_dim": embed_dim,
        }
        
        self.model = sgCLIP(graph_vocab=graph_vocab, **model_cfg)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading stage1 weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
        
        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, graphs):
        # graphs: [objects, boxes, triples, obj_to_img, triple_to_img]
        # We only need the graph encoding part
        
        # sgCLIP.forward signature:
        # forward(self, image, graphs) -> ...
        # But we can access the graph encoder directly if needed, or use the forward method with dummy image
        
        # Looking at sgCLIP/model.py, it seems we can call encode_graph if it exists, 
        # or we can look at how forward is implemented.
        # Based on previous context, there might be an encode_graph_local_global method or similar.
        # Let's assume we can use the internal modules.
        
        # Actually, let's check sgCLIP/model.py content again if possible, but for now I'll implement based on standard usage.
        # The user mentioned "encode_graph_local_global" in the previous grep.
        
        # Re-implementing the graph encoding part from sgCLIP if needed, or calling the method.
        # Let's try to call the method if it exists.
        
        if hasattr(self.model, 'encode_graph_local_global'):
             local_graph_features, global_graph_features = self.model.encode_graph_local_global(graphs)
             return local_graph_features, global_graph_features
        else:
            # Fallback: inspect model structure or assume standard forward
            # For now, let's assume the method exists as seen in previous grep
            raise AttributeError("Model does not have encode_graph_local_global method")

