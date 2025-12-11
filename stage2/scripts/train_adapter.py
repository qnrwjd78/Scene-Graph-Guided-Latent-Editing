import argparse
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader, Dataset

from models.graph_adapter import GraphAdapter, GraphCrossAttnProcessor


class GraphCaptionDataset(Dataset):
    """
    Minimal placeholder dataset.
    Each item should be a dict with:
        - pixel_values: (3, H, W) float tensor in [-1, 1] (SD expects this range after preprocessing)
        - caption: str
        - node_feats: (N_nodes, node_dim) tensor (CLIP/VLM encoded objects)
        - rel_feats: (N_rels, rel_dim) tensor (CLIP/VLM encoded relations)
        - triples: (N_triples, 3) long tensor with (subj_idx, rel_idx, obj_idx)
        - obj_to_img: (N_nodes,) long tensor mapping nodes to image index (all zeros if unbatched)

    Replace this with your real dataset that yields the same keys.
    """

    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(batch: List[Dict]) -> Dict:
    """
    Collate variable-sized graphs by offsetting node/triple indices.
    """
    pixel_values, captions = [], []
    node_feats, rel_feats, triples, obj_to_img = [], [], [], []
    node_offset = 0
    for i, item in enumerate(batch):
        pixel_values.append(item["pixel_values"])
        captions.append(item["caption"])

        n = item["node_feats"].shape[0]
        r = item["rel_feats"].shape[0]

        node_feats.append(item["node_feats"])
        rel_feats.append(item["rel_feats"])

        cur_triples = item["triples"].clone()
        cur_triples[:, 0] += node_offset
        cur_triples[:, 2] += node_offset
        triples.append(cur_triples)

        cur_obj_to_img = torch.full((n,), i, dtype=torch.long)
        obj_to_img.append(cur_obj_to_img)

        node_offset += n

    pixel_values = torch.stack(pixel_values, dim=0)
    node_feats = torch.cat(node_feats, dim=0)
    rel_feats = torch.cat(rel_feats, dim=0)
    triples = torch.cat(triples, dim=0)
    obj_to_img = torch.cat(obj_to_img, dim=0)

    return {
        "pixel_values": pixel_values,
        "captions": captions,
        "node_feats": node_feats,
        "rel_feats": rel_feats,
        "triples": triples,
        "obj_to_img": obj_to_img,
    }


def encode_text(pipe: StableDiffusionPipeline, captions: List[str], device: torch.device):
    text_inputs = pipe.tokenizer(
        captions,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    text_embeddings = pipe.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
    return text_embeddings


def get_graph_proc_params(unet):
    params = []
    seen = set()
    for proc in unet.attn_processors.values():
        if isinstance(proc, GraphCrossAttnProcessor):
            for p in proc.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
    return params


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = StableDiffusionPipeline.from_pretrained(args.sd_model, torch_dtype=torch.float16 if args.fp16 else torch.float32)
    pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    adapter = GraphAdapter(
        node_dim=args.node_dim,
        rel_dim=args.rel_dim,
        hidden_dim=args.hidden_dim,
        cross_attention_dim=pipe.unet.config.cross_attention_dim,
        gcn_layers=args.gcn_layers,
        dropout=args.dropout,
        max_graph_tokens=args.max_graph_tokens,
    ).to(device)

    # Replace this with a real dataset. Here we just raise to remind the user.
    if not args.data_path:
        raise ValueError("Provide --data_path with your prepared graph+caption data.")
    # Implement your own data loading here.
    dataset = torch.load(args.data_path)  # Expect a list of dicts matching GraphCaptionDataset contract
    dataloader = DataLoader(GraphCaptionDataset(dataset), batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    processors_initialized = False
    global_step = 0
    optimizer = None

    for epoch in range(args.epochs):
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            captions = batch["captions"]
            node_feats = batch["node_feats"].to(device)
            rel_feats = batch["rel_feats"].to(device)
            triples = batch["triples"].to(device)
            obj_to_img = batch["obj_to_img"].to(device)

            # Encode graph -> tokens for cross-attn
            graph_tokens, _ = adapter.encode_graph(node_feats, rel_feats, triples, obj_to_img)

            # Install processors once, then only refresh tokens
            if not processors_initialized:
                adapter.apply_to_unet(pipe.unet, graph_tokens)
                params = list(adapter.parameters()) + get_graph_proc_params(pipe.unet)
                optimizer = torch.optim.Adam(params, lr=args.lr)
                processors_initialized = True
            else:
                GraphAdapter.update_graph_tokens(pipe.unet, graph_tokens)

            # Text condition
            text_embeddings = encode_text(pipe, captions, device)

            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
            loss = F.mse_loss(model_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if args.log_steps and global_step % args.log_steps == 0:
                print(f"Step {global_step} | Epoch {epoch} | Loss {loss.item():.4f}")

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(
                {
                    "adapter": adapter.state_dict(),
                    "unet_attn_processors": pipe.unet.attn_processors,
                },
                os.path.join(args.save_dir, f"graph_adapter_epoch{epoch}.pt"),
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--data_path", type=str, required=False, help="Path to a torch-saved list of graph+caption dicts.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--node_dim", type=int, default=512)
    parser.add_argument("--rel_dim", type=int, default=512)
    parser.add_argument("--gcn_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max_graph_tokens", type=int, default=32)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--log_steps", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="checkpoints_graph_adapter")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
