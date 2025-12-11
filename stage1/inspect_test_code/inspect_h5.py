import h5py
import json
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Inspect scene graph in HDF5 file.")
    parser.add_argument("index", type=int, help="Index to inspect")
    parser.add_argument("--h5", type=str, default="datasets/vg/test.h5", help="Path to HDF5 file")
    parser.add_argument("--vocab", type=str, default="datasets/vg/vocab.json", help="Path to vocab json")
    args = parser.parse_args()

    if not os.path.exists(args.h5):
        print(f"Error: {args.h5} not found.")
        return
    
    with open(args.vocab, 'r') as f:
        vocab = json.load(f)
        
    obj_idx_to_name = vocab['object_idx_to_name']
    pred_idx_to_name = vocab['pred_idx_to_name']

    print(f"Loading {args.h5}...")
    with h5py.File(args.h5, 'r') as f:
        # Check keys
        # print("Keys:", list(f.keys()))
        
        # Assuming structure based on VGDataset
        # Usually split_img_indices, images, objects, relationships...
        
        # In many VG implementations:
        # img_to_first_box, img_to_last_box
        # img_to_first_rel, img_to_last_rel
        
        # Check keys
        keys = list(f.keys())
        # print("Keys:", keys)
        
        if 'objects_per_image' in keys:
            # Structure: objects_per_image, object_names, etc.
            # We need to calculate offsets
            
            objects_per_image = f['objects_per_image'][:]
            relationships_per_image = f['relationships_per_image'][:]
            
            # Get Image ID
            if 'image_ids' in f:
                image_id = f['image_ids'][args.index]
                print(f"--- Index {args.index} (Image ID: {image_id}) ---")
            else:
                print(f"--- Index {args.index} (Image ID: Unknown) ---")

            print(f"objects_per_image shape: {objects_per_image.shape}")
            print(f"object_names shape: {f['object_names'].shape}")
            
            if args.index >= len(objects_per_image):
                print(f"Index {args.index} out of bounds.")
                return

            # Check if object_names is (N_images, Max_Objs)
            if f['object_names'].shape[0] == objects_per_image.shape[0]:
                 print("Structure: (N_images, Max_Objs)")
                 # Direct indexing
                 labels = f['object_names'][args.index]
                 # Filter out padding (usually 0 or -1, but 0 is __image__ in vocab, so maybe -1?)
                 # Let's see raw values first
                 print(f"Raw labels for index {args.index}: {labels}")
                 
                 # Assuming 0 is padding if it's at the end? Or maybe it's just valid objects.
                 # But we have objects_per_image count.
                 count = objects_per_image[args.index]
                 labels = labels[:count]
                 
                 obj_names = []
                 print("\nObjects:")
                 for i, label in enumerate(labels):
                    label_str = str(label)
                    if label_str in obj_idx_to_name:
                         name = obj_idx_to_name[label_str]
                    elif label in obj_idx_to_name:
                         name = obj_idx_to_name[label]
                    elif isinstance(obj_idx_to_name, list) and label < len(obj_idx_to_name):
                         name = obj_idx_to_name[label]
                    else:
                         name = f"UNKNOWN({label})"
                    obj_names.append(name)
                    print(f"[{i}] {name}")
                 
                 # Relationships
                 # Check relationship structure too
                 if f['relationship_subjects'].shape[0] == objects_per_image.shape[0]:
                      # (N_images, Max_Rels)
                      rel_count = relationships_per_image[args.index]
                      sub_idxs = f['relationship_subjects'][args.index][:rel_count]
                      obj_idxs = f['relationship_objects'][args.index][:rel_count]
                      pred_idxs = f['relationship_predicates'][args.index][:rel_count]
                      
                      print("\nRelationships:")
                      for i in range(len(sub_idxs)):
                        s_local = sub_idxs[i]
                        o_local = obj_idxs[i]
                        p_val = pred_idxs[i]
                        
                        s_name = obj_names[s_local] if s_local < len(obj_names) else f"GLOBAL({s_local})"
                        o_name = obj_names[o_local] if o_local < len(obj_names) else f"GLOBAL({o_local})"
                        
                        p_val_str = str(p_val)
                        if p_val_str in pred_idx_to_name:
                            p_name = pred_idx_to_name[p_val_str]
                        elif p_val in pred_idx_to_name:
                            p_name = pred_idx_to_name[p_val]
                        elif isinstance(pred_idx_to_name, list) and p_val < len(pred_idx_to_name):
                            p_name = pred_idx_to_name[p_val]
                        else:
                            p_name = f"UNKNOWN({p_val})"
                        
                        print(f"{s_name} --[{p_name}]--> {o_name}")
                 else:
                      print("Relationship structure mismatch (not per-image).")

                 return

            # Calculate offsets (Old logic)
            obj_start = int(np.sum(objects_per_image[:args.index]))
            obj_end = obj_start + int(objects_per_image[args.index])
            
            rel_start = int(np.sum(relationships_per_image[:args.index]))
            rel_end = rel_start + int(relationships_per_image[args.index])
            
            print(f"--- Index {args.index} ---")
            print(f"Objects range: {obj_start} - {obj_end}")
            print(f"Relationships range: {rel_start} - {rel_end}")
            
            # Objects
            # object_names seems to be indices into vocab if it's integers
            print(f"object_names shape: {f['object_names'].shape}, dtype: {f['object_names'].dtype}")
            labels = f['object_names'][obj_start:obj_end]
            print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
            labels = labels.flatten()
            print(f"Labels flattened: {labels}")
            
            obj_names = []
            print("\nObjects:")
            for i, label in enumerate(labels):
                # Vocab keys might be strings
                label_str = str(label)
                if label_str in obj_idx_to_name:
                     name = obj_idx_to_name[label_str]
                elif label in obj_idx_to_name:
                     name = obj_idx_to_name[label]
                elif isinstance(obj_idx_to_name, list) and label < len(obj_idx_to_name):
                     name = obj_idx_to_name[label]
                else:
                     name = f"UNKNOWN({label})"
                     
                obj_names.append(name)
                print(f"[{i}] {name}")
                
            # Relationships
            if rel_end > rel_start:
                sub_idxs = f['relationship_subjects'][rel_start:rel_end].flatten()
                obj_idxs = f['relationship_objects'][rel_start:rel_end].flatten()
                pred_idxs = f['relationship_predicates'][rel_start:rel_end].flatten()
                
                print("\nRelationships:")
                for i in range(len(sub_idxs)):
                    s_local = sub_idxs[i]
                    o_local = obj_idxs[i]
                    p_val = pred_idxs[i]
                    
                    s_name = obj_names[s_local] if s_local < len(obj_names) else f"GLOBAL({s_local})"
                    o_name = obj_names[o_local] if o_local < len(obj_names) else f"GLOBAL({o_local})"
                    
                    p_val_str = str(p_val)
                    if p_val_str in pred_idx_to_name:
                        p_name = pred_idx_to_name[p_val_str]
                    elif p_val in pred_idx_to_name:
                        p_name = pred_idx_to_name[p_val]
                    elif isinstance(pred_idx_to_name, list) and p_val < len(pred_idx_to_name):
                        p_name = pred_idx_to_name[p_val]
                    else:
                        p_name = f"UNKNOWN({p_val})"
                    
                    print(f"{s_name} --[{p_name}]--> {o_name}")
            else:
                print("\nNo relationships.")
                
        elif 'img_to_first_box' in keys:
             # ... (Keep previous logic if needed, but we know it failed)
             pass
        else:
             print("Unknown HDF5 structure. Keys:", keys)
             return

if __name__ == "__main__":
    main()
