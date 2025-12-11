import argparse
import os
from typing import Dict

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from models.graph_adapter import GraphAdapter


def load_graph(graph_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Expect a torch-saved dict with keys:
        - node_feats: (N_nodes, node_dim)
        - rel_feats: (N_rels, rel_dim)
        - triples: (N_triples, 3)
        - obj_to_img: (N_nodes,)
    """
    data = torch.load(graph_path, map_location=device)
    required = {"node_feats", "rel_feats", "triples", "obj_to_img"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Graph file missing keys: {missing}")
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--graph_path", type=str, required=True, help="Path to torch file with graph features.")
    parser.add_argument("--caption", type=str, required=True, help="Caption to condition the generation.")
    parser.add_argument("--output", type=str, default="graph_edit.png")
    parser.add_argument("--node_dim", type=int, default=512)
    parser.add_argument("--rel_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--gcn_layers", type=int, default=3)
    parser.add_argument("--max_graph_tokens", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = StableDiffusionPipeline.from_pretrained(args.sd_model)
    pipe.to(device)

    graph_data = load_graph(args.graph_path, device)

    adapter = GraphAdapter(
        node_dim=args.node_dim,
        rel_dim=args.rel_dim,
        hidden_dim=args.hidden_dim,
        cross_attention_dim=pipe.unet.config.cross_attention_dim,
        gcn_layers=args.gcn_layers,
        max_graph_tokens=args.max_graph_tokens,
    ).to(device)

    graph_tokens, _ = adapter.encode_graph(
        graph_data["node_feats"],
        graph_data["rel_feats"],
        graph_data["triples"],
        graph_data["obj_to_img"],
    )
    adapter.apply_to_unet(pipe.unet, graph_tokens)

    with torch.no_grad():
        image = pipe(
            prompt=args.caption,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
        ).images[0]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if isinstance(image, Image.Image):
        image.save(args.output)
    else:
        image.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
