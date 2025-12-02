import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import networkx as nx
import torch
from PIL import Image


def visualize(h5_path, vocab_path, index, out=None, show=False, save_raw=False, image_dir=None):
    vocab = json.load(open(vocab_path))
    h5 = h5py.File(h5_path, "r")

    objs = torch.tensor(h5["object_names"][index])
    rel_sub = torch.tensor(h5["relationship_subjects"][index])
    rel_obj = torch.tensor(h5["relationship_objects"][index])
    rel_pred = torch.tensor(h5["relationship_predicates"][index])

    names = vocab["object_idx_to_name"]
    preds = vocab["pred_idx_to_name"]

    G = nx.DiGraph()
    for i, o in enumerate(objs):
        if o < 0:
            continue
        G.add_node(i, label=names[o])
    for s, p, o in zip(rel_sub, rel_pred, rel_obj):
        if s < 0 or o < 0 or p < 0:
            continue
        G.add_edge(int(s), int(o), label=preds[p])

    pos = nx.spring_layout(G, seed=0, k=0.05)  # smaller k -> nodes closer
    plt.figure(figsize=(5.5, 4.5))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={n: d["label"] for n, d in G.nodes(data=True)},
        node_color="#88c",
        node_size=400,
        font_size=7,
    )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels={(u, v): d["label"] for u, v, d in G.edges(data=True)},
        font_color="red",
        font_size=6,
    )
    plt.axis("off")
    plt.tight_layout()
    # 그래프 저장 경로가 없으면 기본 이름 부여 (그래프와 raw를 같은 위치에 저장하기 위함)
    out_path = Path(out) if out else Path(f"graph_{index}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved graph to {out_path}")

    if image_dir:
        if "image_paths" in h5:
            img_rel = h5["image_paths"][index]
            img_rel = img_rel.decode() if isinstance(img_rel, (bytes, bytearray)) else str(img_rel)
            img_path = Path(image_dir) / img_rel
            if img_path.exists():
                raw = Image.open(img_path).convert("RGB")
                raw_out = out_path.with_name(out_path.stem + "_raw" + out_path.suffix)
                raw.save(raw_out)
                print(f"Saved raw image to {raw_out}")
            else:
                print(f"[warn] raw image not found: {img_path}")
        else:
            print("[warn] image_paths not found in h5; cannot save raw image.")
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="datasets/vg/val.h5")
    parser.add_argument("--vocab", default="datasets/vg/vocab.json")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save_raw", action="store_true", help="Save raw image alongside graph (requires image dir)")
    parser.add_argument("--image_dir", default="datasets/vg/images", help="Path to VG images")
    args = parser.parse_args()
    visualize(
        args.h5,
        args.vocab,
        args.index,
        out=args.out,
        show=args.show,
        save_raw=args.save_raw,
        image_dir=args.image_dir,
    )


if __name__ == "__main__":
    main()