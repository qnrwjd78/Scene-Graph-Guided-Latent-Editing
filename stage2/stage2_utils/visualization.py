import matplotlib.pyplot as plt
import numpy as np

def draw_scene_graph_matplotlib(objs, triples, vocab, save_path):
    """
    Draws the scene graph using matplotlib with a simple circular layout.
    """
    obj_names = [vocab['object_idx_to_name'][i.item()] for i in objs]
    
    # Filter out __image__ for visualization
    valid_indices = [i for i, name in enumerate(obj_names) if name != '__image__']
    valid_obj_names = [obj_names[i] for i in valid_indices]
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
    
    num_nodes = len(valid_obj_names)
    if num_nodes == 0:
        return

    # Circular Layout
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    r = 1.0
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Draw Nodes
    for i in range(num_nodes):
        ax.add_patch(plt.Circle((x[i], y[i]), 0.1, color='lightblue', zorder=2))
        ax.text(x[i], y[i], valid_obj_names[i], ha='center', va='center', zorder=3, fontsize=10, fontweight='bold')
        
    # Draw Edges
    for t in triples:
        s, p, o = t.tolist()
        
        # Skip if connected to __image__ (which was filtered out)
        if s not in mapping or o not in mapping:
            continue
            
        s_new = mapping[s]
        o_new = mapping[o]
        
        p_name = vocab['pred_idx_to_name'][p]
        if p_name == '__in_image__':
            continue
            
        start = (x[s_new], y[s_new])
        end = (x[o_new], y[o_new])
        
        # Draw arrow
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="gray", lw=1.5), zorder=1)
        
        # Draw label (midpoint)
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, p_name, ha='center', va='center', fontsize=8, color='red', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    plt.title("Scene Graph Visualization")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
