import torch
import copy
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'stage1'))
from stage1.utils import create_tensor_by_assign_samples_to_img

def test_relationship_selection():
    print("Testing relationship selection logic...")
    
    # Setup mock data
    # 1 image, max 5 relationships
    max_sample = 5
    batch_size = 1
    
    # Create dummy samples (vectors)
    # Let's say we have 10 relationships total
    num_rels = 10
    samples = torch.randn(num_rels, 128)
    sample_to_img = torch.zeros(num_rels, dtype=torch.long)
    
    # Define indices
    # 0 is dummy (__in_image__ or __image__)
    # Let's define:
    # 0, 1, 2: Real relationships (pred!=0, obj!=0)
    # 3, 4, 5: Dummy relationships (pred=0 or obj=0)
    # 6, 7, 8, 9: More dummies
    
    # Case 1: Prioritize Real
    # Real: indices 0, 1
    # Dummy: indices 2, 3, 4
    # Subject indices:
    # Real 0: Subject 10
    # Real 1: Subject 11
    # Dummy 2: Subject 10 (Covered)
    # Dummy 3: Subject 12 (Not Covered - Priority)
    # Dummy 4: Subject 13 (Not Covered - Priority)
    
    # Expected selection: 0, 1 (Real), then 3, 4 (Priority Dummies), then 2 (Covered Dummy) -> Total 5
    
    s = torch.tensor([10, 11, 10, 12, 13], dtype=torch.long)
    p = torch.tensor([1,  1,  0,  0,  0], dtype=torch.long) # 0 is dummy pred
    o = torch.tensor([2,  2,  0,  0,  0], dtype=torch.long) # 0 is dummy obj
    
    # We need to pass these to the function
    # But wait, the function takes samples corresponding to these.
    # So we need samples of length 5.
    
    samples_test = samples[:5]
    sample_to_img_test = sample_to_img[:5]
    
    print("Input:")
    print(f"Subjects: {s}")
    print(f"Predicates: {p}")
    print(f"Objects: {o}")
    print(f"Real Rels: Indices 0, 1")
    print(f"Dummy Rels: Indices 2 (Subj 10), 3 (Subj 12), 4 (Subj 13)")
    print(f"Expected: [0, 1, 3, 4, 2] (Order might vary within groups but 3,4 should be before 2)")
    
    # Mock the function call
    # Note: The function returns a tensor of shape [B, Max, D]
    # We can't easily check indices from output tensor unless we mock samples to be indices.
    
    # Let's make samples equal to their indices for tracking
    samples_indices = torch.arange(5, dtype=torch.float32).unsqueeze(1) # [5, 1]
    
    output = create_tensor_by_assign_samples_to_img(
        samples_indices, 
        sample_to_img_test, 
        max_sample, 
        batch_size, 
        subject_idxs=s, 
        pred_idxs=p, 
        obj_idxs=o
    )
    
    selected_indices = output[0, :, 0].long().tolist()
    print(f"Selected Indices: {selected_indices}")
    
    # Check if 0 and 1 are present (Real)
    assert 0 in selected_indices
    assert 1 in selected_indices
    
    # Check prioritization
    # 3 and 4 should be prioritized over 2 because their subjects (12, 13) are not in real subjects (10, 11)
    # Wait, Real 0 has Subject 10. Dummy 2 has Subject 10. So Dummy 2 is covered.
    # Real 1 has Subject 11. Dummy 3 has Subject 12. Not covered.
    # Dummy 4 has Subject 13. Not covered.
    
    # So 3 and 4 should be preferred.
    # Since max is 5, and we have 5 total, all should be selected.
    # Let's reduce max to 4 to test filtering.
    
    print("\nTest with max=4 (Should drop index 2)")
    output_4 = create_tensor_by_assign_samples_to_img(
        samples_indices, 
        sample_to_img_test, 
        4, 
        batch_size, 
        subject_idxs=s, 
        pred_idxs=p, 
        obj_idxs=o
    )
    selected_indices_4 = output_4[0, :, 0].long().tolist()
    print(f"Selected Indices (max=4): {selected_indices_4}")
    
    assert 0 in selected_indices_4
    assert 1 in selected_indices_4
    assert 3 in selected_indices_4
    assert 4 in selected_indices_4
    assert 2 not in selected_indices_4
    
    print("Verification Successful!")

if __name__ == "__main__":
    test_relationship_selection()
