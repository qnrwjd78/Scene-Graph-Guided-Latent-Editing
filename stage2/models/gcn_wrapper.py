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
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter out size mismatches
            model_state_dict = self.model.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"⚠️ Skipping {k}: shape mismatch (Checkpoint: {v.shape}, Model: {model_state_dict[k].shape})")
            
            self.model.load_state_dict(filtered_state_dict, strict=False)
            print(f"✅ Successfully loaded Stage 1 GCN weights from {checkpoint_path}")
        else:
            print(f"⚠️ WARNING: Stage 1 checkpoint not found at {checkpoint_path}. GCN is initialized randomly!")
            print("⚠️ This will cause the model to learn from garbage inputs. Please check 'pretrained/sip_vg.pt'.")
        
        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, graphs):
        # graphs: [objects, boxes, triples, obj_to_img, triple_to_img]
        
        # Infer batch size from obj_to_img
        obj_to_img = graphs[3]
        if obj_to_img.numel() > 0:
            batch_size = obj_to_img.max().item() + 1
        else:
            batch_size = 1 # Default or handle empty
            
        # Create dummy image for sgCLIP signature
        # sgCLIP uses img.shape to determine batch_size
        dummy_img = torch.zeros(batch_size, 3, 224, 224, device=obj_to_img.device)
        
        if hasattr(self.model, 'encode_graph_local_global'):
             local_graph_features, global_graph_features = self.model.encode_graph_local_global(dummy_img, graphs)
             return local_graph_features, global_graph_features
        else:
            raise AttributeError("Model does not have encode_graph_local_global method")

    def get_raw_features(self, graphs):
        return self.model.get_raw_features(graphs)

