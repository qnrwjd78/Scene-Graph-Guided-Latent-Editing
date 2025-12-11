import json

file_path = '/root/Scene-Graph-Guided-Latent-Editing/datasets/vg/relationships.json'

print(f"Loading {file_path}...")
with open(file_path, 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} images.")

num_relationships = []
in_image_relationships = 0
total_relationships = 0

for img in data:
    rels = img['relationships']
    num_relationships.append(len(rels))
    total_relationships += len(rels)
    for r in rels:
        # Check for "in image" type relationships
        # The user mentioned [object]-__in__-__image__
        # Let's check the predicate
        if 'predicate' in r:
            if 'in image' in r['predicate'].lower() or '__in__' in r['predicate'].lower():
                in_image_relationships += 1

min_rels = min(num_relationships)
max_rels = max(num_relationships)
avg_rels = sum(num_relationships) / len(num_relationships)
num_relationships.sort()
median_rels = num_relationships[len(num_relationships) // 2]

print(f"Minimum relationships: {min_rels}")
print(f"Maximum relationships: {max_rels}")
print(f"Average relationships: {avg_rels}")
print(f"Median relationships: {median_rels}")
print(f"Total relationships: {total_relationships}")
print(f"Relationships with 'in image' or '__in__' in predicate: {in_image_relationships}")

# Print some example predicates to see what they look like
print("\nExample predicates:")
predicates = {}
for i in range(min(100, len(data))):
    for r in data[i]['relationships']:
        p = r['predicate']
        predicates[p] = predicates.get(p, 0) + 1

for p, count in list(predicates.items())[:20]:
    print(f"{p}: {count}")
