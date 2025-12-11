import argparse
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

from stage1.models.gcn_clip import GraphSceneEncoder


class GraphCLIPDataset(Dataset):
    """
    Expect a torch-saved list of dicts. Each dict should include:
      - node_texts: List[str]
      - rel_texts:  List[str]
      - triples:    LongTensor (N_triples, 3) with subj_idx, rel_idx, obj_idx
      - image_emb:  Tensor (clip_dim,) precomputed CLIP image embedding (optional, see note)
    If image_emb is missing, you must provide image pixel data and modify this loader.
    """

    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(batch: List[Dict]):
    node_texts, rel_texts, triples, obj_to_img, image_embs = [], [], [], [], []
    node_offset, rel_offset = 0, 0
    for i, item in enumerate(batch):
        n = len(item["node_texts"])
        r = len(item["rel_texts"])

        node_texts.extend(item["node_texts"])
        rel_texts.extend(item["rel_texts"])

        t = item["triples"].clone()
        t[:, 0] += node_offset
        t[:, 2] += node_offset
        t[:, 1] += rel_offset
        triples.append(t)

        obj_to_img.append(torch.full((n,), i, dtype=torch.long))
        node_offset += n
        rel_offset += r

        if "image_emb" in item:
            image_embs.append(item["image_emb"])
        else:
            raise ValueError("image_emb missing; please provide CLIP image embeddings or extend loader to images.")

    triples = torch.cat(triples, dim=0) if triples else torch.zeros((0, 3), dtype=torch.long)
    obj_to_img = torch.cat(obj_to_img, dim=0) if obj_to_img else torch.zeros((0,), dtype=torch.long)
    image_embs = torch.stack(image_embs, dim=0)

    return {
        "node_texts": node_texts,
        "rel_texts": rel_texts,
        "triples": triples,
        "obj_to_img": obj_to_img,
        "image_embs": image_embs,
    }


def encode_texts(processor: CLIPProcessor, model: CLIPModel, texts: List[str], device: torch.device) -> torch.Tensor:
    if len(texts) == 0:
        return torch.zeros((0, model.config.projection_dim), device=device)
    batch = processor.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=processor.tokenizer.model_max_length,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**batch)
    return text_features


def contrastive_loss(graph_emb: torch.Tensor, image_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    g = F.normalize(graph_emb, dim=-1)
    i = F.normalize(image_emb, dim=-1)
    logits = g @ i.t() / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_g = F.cross_entropy(logits.t(), targets)
    return (loss_i + loss_g) * 0.5


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading data from {args.data_path}")
    data = torch.load(args.data_path)
    dataloader = DataLoader(
        GraphCLIPDataset(data),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    print(f"Loading CLIP model {args.clip_model}")
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    clip_dim = clip_model.config.projection_dim
    encoder = GraphSceneEncoder(
        clip_dim=clip_dim,
        hidden_dim=args.hidden_dim,
        gcn_layers=args.gcn_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            node_feats = encode_texts(clip_processor, clip_model, batch["node_texts"], device)
            rel_feats = encode_texts(clip_processor, clip_model, batch["rel_texts"], device)

            triples = batch["triples"].to(device)
            obj_to_img = batch["obj_to_img"].to(device)
            image_embs = batch["image_embs"].to(device)

            graph_embs = encoder(node_feats, rel_feats, triples, obj_to_img)
            loss = contrastive_loss(graph_embs, image_embs, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if args.log_steps and global_step % args.log_steps == 0:
                print(f"Step {global_step} | Epoch {epoch} | Loss {loss.item():.4f}")

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt = {
                "encoder": encoder.state_dict(),
                "clip_model": args.clip_model,
                "config": vars(args),
            }
            path = os.path.join(args.save_dir, f"gcn_clip_epoch{epoch}.pt")
            torch.save(ckpt, path)
            print(f"Saved {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to torch-saved list of graph dicts.")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--gcn_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--save_dir", type=str, default="checkpoints_stage1")
    parser.add_argument("--log_steps", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
