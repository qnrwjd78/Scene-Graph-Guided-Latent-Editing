"""
Test EDICT editing with Stable Diffusion v1.5: Replace horse with zebra
Uses runwayml/stable-diffusion-v1-5 instead of pretrained_150
"""

import torch
import json
import os
import sys
import PIL
from PIL import Image
import numpy as np
import h5py
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers import StableDiffusionPipeline, DDIMScheduler
from models.gcn_wrapper import GCNWrapper
from models.graph_adapter import GraphAdapter


class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)


def load_test_sample(test_idx, vocab_path, h5_path, image_dir, image_size=512):
    """Load a specific test sample"""
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    transform = transforms.Compose([
        Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    # Load h5 data
    with h5py.File(h5_path, 'r') as f:
        image_paths = list(f['image_paths'])
        data = {k: torch.IntTensor(np.asarray(v)) for k, v in f.items() if k != 'image_paths'}

    # Get image
    img_path = os.path.join(image_dir, str(image_paths[test_idx], encoding="utf-8"))
    with open(img_path, 'rb') as f:
        with PIL.Image.open(f) as image:
            WW, HH = image.size
            pil_image = image.convert('RGB')
            image_tensor = transform(pil_image)

    # Process scene graph
    obj_idxs_with_rels = set()
    for r_idx in range(data['relationships_per_image'][test_idx]):
        s = data['relationship_subjects'][test_idx, r_idx].item()
        o = data['relationship_objects'][test_idx, r_idx].item()
        obj_idxs_with_rels.add(s)
        obj_idxs_with_rels.add(o)

    obj_idxs = list(obj_idxs_with_rels)[:9]  # Max 9 objects (+ __image__ = 10)
    O = len(obj_idxs) + 1

    objs = torch.LongTensor(O).fill_(-1)
    boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
    obj_idx_mapping = {}

    for i, obj_idx in enumerate(obj_idxs):
        objs[i] = data['object_names'][test_idx, obj_idx].item()
        x, y, w, h = data['object_boxes'][test_idx, obj_idx].tolist()
        x0, y0 = float(x) / WW, float(y) / HH
        x1, y1 = float(x + w) / WW, float(y + h) / HH
        boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
        obj_idx_mapping[obj_idx] = i

    objs[O - 1] = vocab['object_name_to_idx']['__image__']

    # Process relationships
    triples = []
    for r_idx in range(data['relationships_per_image'][test_idx].item()):
        s = data['relationship_subjects'][test_idx, r_idx].item()
        p = data['relationship_predicates'][test_idx, r_idx].item()
        o = data['relationship_objects'][test_idx, r_idx].item()
        s = obj_idx_mapping.get(s, None)
        o = obj_idx_mapping.get(o, None)
        if s is not None and o is not None:
            triples.append([s, p, o])

    # Add __in_image__ relationships
    in_image = vocab['pred_name_to_idx']['__in_image__']
    for i in range(O - 1):
        triples.append([i, in_image, O - 1])

    triples = torch.LongTensor(triples)

    return pil_image, image_tensor, objs, boxes, triples, vocab


def replace_horse_with_zebra(objs, vocab):
    """Replace 'horse' with 'zebra' in scene graph"""
    obj_name_to_idx = vocab['object_name_to_idx']
    obj_idx_to_name = vocab['object_idx_to_name']

    new_objs = objs.clone()

    # Find horse and replace with zebra
    horse_found = False
    for i, obj_idx in enumerate(new_objs[:-1]):  # Exclude __image__
        if obj_idx_to_name[obj_idx.item()] == 'horse':
            horse_found = True
            print(f"  Found horse at index {i}")
            new_objs[i] = obj_name_to_idx['zebra']
            print(f"  Replaced horse with zebra")
            break

    if not horse_found:
        print("  Warning: No horse found in scene graph!")
        print("  Objects:", [obj_idx_to_name[idx.item()] for idx in new_objs[:-1]])

    return new_objs


def prepare_graph_conditioning(gcn, adapter, objs, boxes, triples, device):
    """Prepare scene graph conditioning using GCN + Adapter"""
    objs = objs.unsqueeze(0).to(device)
    boxes = boxes.unsqueeze(0).to(device)
    triples = triples.unsqueeze(0).to(device)

    with torch.no_grad():
        obj_to_img = torch.zeros(objs.size(1), dtype=torch.long).to(device)
        triple_to_img = torch.zeros(triples.size(1), dtype=torch.long).to(device)
        objs_flat = objs.view(-1)
        boxes_flat = boxes.view(-1, 4)
        triples_flat = triples.view(-1, 3)

        graphs = [objs_flat, boxes_flat, triples_flat, obj_to_img, triple_to_img]
        local_feats, global_feats = gcn(graphs)

        cond_embeddings = adapter(local_feats)

        # Pad to 77 for SD compatibility
        B, N, C = cond_embeddings.shape
        if N < 77:
            padding = torch.zeros(B, 77 - N, C, device=device)
            cond_embeddings = torch.cat([cond_embeddings, padding], dim=1)
        elif N > 77:
            cond_embeddings = cond_embeddings[:, :77, :]

    return cond_embeddings


def edict_inversion(pipe, image, base_cond, num_steps=50, mix_weight=0.93):
    """
    EDICT-style inversion with coupling
    """
    device = pipe.device

    # Encode image to latent
    with torch.no_grad():
        latent = pipe.vae.encode(image).latent_dist.sample() * pipe.vae.config.scaling_factor

    pipe.scheduler.set_timesteps(num_steps)
    timesteps = reversed(pipe.scheduler.timesteps)

    print(f"  Running EDICT inversion with mix_weight={mix_weight}...")

    x_t = latent
    for i, t in enumerate(tqdm(timesteps, desc="Inverting")):
        # Get model prediction
        with torch.no_grad():
            noise_pred = pipe.unet(x_t, t, encoder_hidden_states=base_cond).sample

        # DDIM inversion step
        alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
        if i < len(timesteps) - 1:
            t_prev = list(reversed(pipe.scheduler.timesteps))[i + 1]
            alpha_prod_t_prev = pipe.scheduler.alphas_cumprod[t_prev]
        else:
            alpha_prod_t_prev = torch.tensor(0.0)

        beta_prod_t = 1 - alpha_prod_t

        # Predict x0
        pred_original_sample = (x_t - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

        # EDICT coupling
        if i > 0:
            pred_original_sample = mix_weight * pred_original_sample + (1 - mix_weight) * x_0_coupled

        x_0_coupled = pred_original_sample

        # Get next sample
        if i < len(timesteps) - 1:
            pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
            x_t = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

    return x_t


def edict_sampling(pipe, x_T, edit_cond, num_steps=50, mix_weight=0.93, guidance_scale=7.5):
    """
    EDICT-style sampling with coupling
    """
    device = pipe.device

    pipe.scheduler.set_timesteps(num_steps)
    timesteps = pipe.scheduler.timesteps

    print(f"  Running EDICT sampling with mix_weight={mix_weight}...")

    # Unconditional embeddings for CFG
    uncond_embeddings = torch.zeros_like(edit_cond)

    x_t = x_T
    x_0_coupled = None

    for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
        # CFG
        latent_model_input = torch.cat([x_t] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        concat_embeds = torch.cat([uncond_embeddings, edit_cond])

        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=concat_embeds).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Get alpha values
        alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t

        # Predict x0
        pred_original_sample = (x_t - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

        # EDICT coupling
        if x_0_coupled is not None:
            pred_original_sample = mix_weight * pred_original_sample + (1 - mix_weight) * x_0_coupled

        x_0_coupled = pred_original_sample

        # DDIM step
        x_t = pipe.scheduler.step(noise_pred, t, x_t, eta=0.0).prev_sample

    return x_t


def main():
    parser = argparse.ArgumentParser(description='EDICT + SD v1.5: Replace horse with zebra')
    parser.add_argument('--test_idx', type=int, default=50,
                       help='Test sample index (default: 50, has horse)')
    parser.add_argument('--output_dir', type=str, default='./edict_sd_results',
                       help='Output directory')
    parser.add_argument('--mix_weight', type=float, default=0.93,
                       help='EDICT mix weight (default: 0.93)')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of diffusion steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='Classifier-free guidance scale')
    args = parser.parse_args()

    print("="*70)
    print("EDICT + Stable Diffusion v1.5: Replace horse with zebra")
    print(f"Test Index: {args.test_idx}")
    print("="*70)

    # Configuration
    vocab_path = '../datasets/vg/vocab.json'
    h5_path = '../datasets/vg/val.h5'
    image_dir = '../datasets/vg/images'

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Load models
    print(f"\n1. Loading Stable Diffusion v1.5...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    print(f"   Loading GCN and Adapter...")
    gcn = GCNWrapper(
        vocab_file="../datasets/vg/vocab.json",
        checkpoint_path="../pretrained/sip_vg.pt"
    ).to(device)
    gcn.eval()

    adapter = GraphAdapter().to(device)
    adapter_path = "checkpoints/adapter.pth"
    if os.path.exists(adapter_path):
        checkpoint = torch.load(adapter_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            adapter.load_state_dict(checkpoint["model_state_dict"])
        else:
            adapter.load_state_dict(checkpoint)
        print(f"   Loaded adapter from {adapter_path}")
    else:
        print(f"   Warning: Adapter not found, using random weights")
    adapter.eval()

    # Step 2: Load test sample
    print(f"\n2. Loading test sample {args.test_idx}...")
    pil_image, image_tensor, base_objs, base_boxes, base_triples, vocab = load_test_sample(
        args.test_idx, vocab_path, h5_path, image_dir, image_size=512
    )

    obj_idx_to_name = vocab['object_idx_to_name']
    print(f"\n   Base scene graph objects:")
    for i, obj_idx in enumerate(base_objs[:-1]):
        print(f"     {i}: {obj_idx_to_name[obj_idx.item()]}")

    # Save original image
    pil_image.save(os.path.join(args.output_dir, f'test_{args.test_idx}_original.png'))
    print(f"\n   Saved original image")

    # Step 3: Replace horse with zebra
    print(f"\n3. Creating edited scene graph (horse → zebra)...")
    edit_objs = replace_horse_with_zebra(base_objs.clone(), vocab)

    # Step 4: Prepare scene graph conditioning
    print(f"\n4. Preparing scene graph conditioning...")
    base_cond = prepare_graph_conditioning(gcn, adapter, base_objs, base_boxes, base_triples, device)
    edit_cond = prepare_graph_conditioning(gcn, adapter, edit_objs, base_boxes, base_triples, device)

    # Step 5: Prepare image for inversion
    print(f"\n5. Preparing image for inversion...")
    # Convert to [-1, 1] range for SD
    image_sd = image_tensor.unsqueeze(0).to(device) * 2 - 1

    # Step 6: EDICT Inversion
    print(f"\n6. Running EDICT inversion...")
    x_T = edict_inversion(pipe, image_sd, base_cond, num_steps=args.num_steps, mix_weight=args.mix_weight)
    print(f"   Inversion complete")

    # Step 7: Reconstruction (for comparison)
    print(f"\n7. Reconstructing with base scene graph...")
    recon_latent = edict_sampling(pipe, x_T, base_cond, num_steps=args.num_steps,
                                  mix_weight=args.mix_weight, guidance_scale=args.guidance_scale)

    with torch.no_grad():
        recon_image = pipe.vae.decode(recon_latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
        recon_image = pipe.image_processor.postprocess(recon_image, output_type="pil", do_denormalize=[True])[0]

    recon_image.save(os.path.join(args.output_dir, f'test_{args.test_idx}_edict_recon.png'))
    print(f"   Saved reconstruction (should look like original)")

    # Step 8: EDICT Sampling with edited scene graph
    print(f"\n8. Running EDICT sampling with edited scene graph (horse → zebra)...")
    edited_latent = edict_sampling(pipe, x_T, edit_cond, num_steps=args.num_steps,
                                   mix_weight=args.mix_weight, guidance_scale=args.guidance_scale)

    with torch.no_grad():
        edited_image = pipe.vae.decode(edited_latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
        edited_image = pipe.image_processor.postprocess(edited_image, output_type="pil", do_denormalize=[True])[0]

    edited_image.save(os.path.join(args.output_dir, f'test_{args.test_idx}_edict_zebra.png'))

    print(f"\n{'='*70}")
    print(f"Complete! Results saved to {args.output_dir}/")
    print(f"{'='*70}")
    print(f"\nFiles:")
    print(f"  - test_{args.test_idx}_original.png      (original image)")
    print(f"  - test_{args.test_idx}_edict_recon.png   (EDICT reconstruction)")
    print(f"  - test_{args.test_idx}_edict_zebra.png   (EDICT edited: horse → zebra)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
