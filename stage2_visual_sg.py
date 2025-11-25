import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stage2'))
import json
import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import networkx as nx
import torch
from PIL import Image


def sampler_style_graph(h5, vocab, index, max_objects, use_orphaned_objects, include_relationships):
    """
    Mirror the sampling logic used in testset_ddim_sampler.VgSceneGraphDataset so
    graph indices match what the sampler sees.
    """
    objs_per_image = int(h5["objects_per_image"][index])
    rels_per_image = int(h5["relationships_per_image"][index])

    obj_idxs_with_rels = set()
    obj_idxs_without_rels = set(range(objs_per_image))
    for r_idx in range(rels_per_image):
        s = int(h5["relationship_subjects"][index, r_idx])
        o = int(h5["relationship_objects"][index, r_idx])
        obj_idxs_with_rels.add(s)
        obj_idxs_without_rels.discard(s)
        obj_idxs_with_rels.add(o)
        obj_idxs_without_rels.discard(o)

    obj_idxs = list(obj_idxs_with_rels)
    obj_idxs_without_rels = list(obj_idxs_without_rels)
    # Keep off-by-one from the sampler (sample up to max_objects objects, then add __image__)
    if len(obj_idxs) > max_objects - 1:
        obj_idxs = random.sample(obj_idxs, max_objects)
    if len(obj_idxs) < max_objects - 1 and use_orphaned_objects:
        num_to_add = max_objects - 1 - len(obj_idxs)
        num_to_add = min(num_to_add, len(obj_idxs_without_rels))
        obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)
    O = len(obj_idxs) + 1  # plus __image__

    objs = torch.full((O,), -1, dtype=torch.long)
    obj_idx_mapping = {}
    for i, obj_idx in enumerate(obj_idxs):
        objs[i] = int(h5["object_names"][index, obj_idx])
        obj_idx_mapping[obj_idx] = i
    objs[O - 1] = vocab["object_name_to_idx"]["__image__"]

    triples = []
    if include_relationships:
        for r_idx in range(rels_per_image):
            s_raw = int(h5["relationship_subjects"][index, r_idx])
            o_raw = int(h5["relationship_objects"][index, r_idx])
            p_raw = int(h5["relationship_predicates"][index, r_idx])
            s = obj_idx_mapping.get(s_raw)
            o = obj_idx_mapping.get(o_raw)
            if s is not None and o is not None:
                triples.append([s, p_raw, o])

    in_image = vocab["pred_name_to_idx"]["__in_image__"]
    for i in range(O - 1):
        triples.append([i, in_image, O - 1])
    return objs, torch.tensor(triples, dtype=torch.long)


def visualize(
    h5_path,
    vocab_path,
    index,
    out=None,
    show=False,
    save_raw=False,
    image_dir=None,
    sampler_style=True,
    max_objects=30,
    use_orphaned_objects=True,
    include_relationships=True,
    ignore_dummies=True,
):
    vocab = json.load(open(vocab_path))
    h5 = h5py.File(h5_path, "r")

    if sampler_style:
        objs, triples = sampler_style_graph(
            h5,
            vocab,
            index,
            max_objects=max_objects,
            use_orphaned_objects=use_orphaned_objects,
            include_relationships=include_relationships,
        )
        edges = triples
    else:
        objs = torch.tensor(h5["object_names"][index])
        rel_sub = torch.tensor(h5["relationship_subjects"][index])
        rel_obj = torch.tensor(h5["relationship_objects"][index])
        rel_pred = torch.tensor(h5["relationship_predicates"][index])
        edges = torch.stack([rel_sub, rel_pred, rel_obj], dim=1)

    names = vocab["object_idx_to_name"]
    preds = vocab["pred_idx_to_name"]

    G = nx.DiGraph()
    for i, o in enumerate(objs):
        o_idx = int(o)
        if o_idx < 0:
            continue
        if ignore_dummies and names[o_idx] == "__image__":
            continue
        G.add_node(i, label=names[o_idx])
    for s, p, o in edges:
        s_idx, p_idx, o_idx = int(s), int(p), int(o)
        if s_idx < 0 or o_idx < 0 or p_idx < 0:
            continue
        if ignore_dummies and preds[p_idx] == "__in_image__":
            continue
        G.add_edge(s_idx, o_idx, label=preds[p_idx])

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
    out_path = Path(out) if out else Path(f"graph_{index}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved graph to {out_path}")

    if save_raw and image_dir:
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
    parser.add_argument("--h5", default="datasets/vg/test.h5", help="H5 split; matches testset_ddim_sampler default")
    parser.add_argument("--vocab", default="datasets/vg/vocab.json")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save_raw", action="store_true", help="Save raw image alongside graph (requires image dir)")
    parser.add_argument("--image_dir", default="datasets/vg/images", help="Path to VG images")
    parser.add_argument("--max_objects", type=int, default=30, help="Match VgSceneGraphDataset max_objects")
    parser.add_argument("--use_orphaned_objects", dest="use_orphaned_objects", action="store_true")
    parser.add_argument("--no_orphaned_objects", dest="use_orphaned_objects", action="store_false")
    parser.set_defaults(use_orphaned_objects=True)
    parser.add_argument("--no_relationships", dest="include_relationships", action="store_false")
    parser.set_defaults(include_relationships=True)
    parser.add_argument(
        "--raw",
        dest="sampler_style",
        action="store_false",
        help="Visualize raw h5 entries (original behavior)",
    )
    parser.add_argument(
        "--keep_dummies",
        dest="ignore_dummies",
        action="store_false",
        help="Show __image__ nodes and __in_image__ edges",
    )
    parser.set_defaults(sampler_style=True, ignore_dummies=True)
    args = parser.parse_args()
    visualize(
        args.h5,
        args.vocab,
        args.index,
        out=args.out,
        show=args.show,
        save_raw=args.save_raw,
        image_dir=args.image_dir,
        sampler_style=args.sampler_style,
        max_objects=args.max_objects,
        use_orphaned_objects=args.use_orphaned_objects,
        include_relationships=args.include_relationships,
        ignore_dummies=args.ignore_dummies,
    )


if __name__ == "__main__":
    main()
