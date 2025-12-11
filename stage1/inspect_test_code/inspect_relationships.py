import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Inspect relationships for a specific image index.")
    parser.add_argument("index", type=int, nargs='?', help="The index of the image to inspect (0-based index in the JSON list).")
    parser.add_argument("--file", type=str, default="datasets/vg/relationships.json", help="Path to relationships.json")
    parser.add_argument("--image_id", type=int, help="Search by Image ID instead of index.")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        return

    print(f"Loading {args.file}...")
    with open(args.file, 'r') as f:
        data = json.load(f)

    target_item = None
    target_index = -1

    if args.image_id is not None:
        print(f"Searching for Image ID: {args.image_id}...")
        # Create a map for faster lookup if needed, but linear scan is fine for one-off
        for i, item in enumerate(data):
            if item.get('image_id') == args.image_id:
                target_item = item
                target_index = i
                break
        if target_item is None:
            print(f"Error: Image ID {args.image_id} not found.")
            return
    elif args.index is not None:
        if args.index < 0 or args.index >= len(data):
            print(f"Error: Index {args.index} is out of bounds. Total images: {len(data)}")
            return
        target_item = data[args.index]
        target_index = args.index
    else:
        print("Error: Must provide either index or --image_id")
        return

    image_id = target_item.get('image_id', 'Unknown')
    relationships = target_item.get('relationships', [])

    print(f"--- Image Index: {target_index} (Image ID: {image_id}) ---")
    print(f"Total Relationships: {len(relationships)}")
    
    for i, rel in enumerate(relationships):
        subject = rel.get('subject', {})
        predicate = rel.get('predicate', 'UNKNOWN')
        object_ = rel.get('object', {})
        
        s_name = subject.get('name', subject.get('names', ['?'])[0])
        o_name = object_.get('name', object_.get('names', ['?'])[0])
        
        print(f"[{i}] {s_name} --[{predicate}]--> {o_name}")
        # print(f"    Full Data: {rel}") # Uncomment to see full details

if __name__ == "__main__":
    main()
