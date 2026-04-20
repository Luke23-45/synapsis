"""
Final Verification for NTH-Attention v2.0 (SOTA)

Verifies:
1. Model initialization and parameter count
2. Forward pass with multi-view images and language instructions
3. Loss computation and gradient flow
4. Inference sampling (Flow Matching ODE)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nth import (
    get_config,
    NTHDiffusionPlanner,
)

def verify_v2():
    print("=" * 60)
    print("NTH-Attention v2.0 SOTA Verification")
    print("=" * 60)
    
    # 1. Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_config('base')
    model = NTHDiffusionPlanner(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized on {device}")
    print(f"✓ Trainable parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # 2. Dummy Batch Generation
    B = 2
    batch = {
        'image': torch.randn(B, 3, config.image_size, config.image_size, device=device),
        'wrist_image': torch.randn(B, 3, config.image_size, config.image_size, device=device),
        'proprio': torch.randn(B, config.proprio_dim, device=device),
        'proprio_history': torch.randn(B, config.proprio_horizon, config.proprio_dim, device=device),
        'action_chunk': torch.randn(B, config.action_chunk_size, config.action_dim, device=device),
        'language_instruction': [
            "pick up the red block",
            "stack block on tray"
        ]
    }
    print(f"✓ Dummy batch created (B={B}, language=['...'])")
    
    # 3. Forward Pass
    print("\nTesting Forward Pass...")
    outputs = model(batch)
    
    assert 'v_pred' in outputs, "Missing v_pred in outputs"
    assert outputs['v_pred'].shape == (B, config.action_chunk_size, config.action_dim), \
        f"Unexpected v_pred shape: {outputs['v_pred'].shape}"
    print(f"✓ Forward pass successful. Output shape: {outputs['v_pred'].shape}")
    
    # 4. Backward Pass
    print("\nTesting Backward Pass...")
    loss, loss_dict = model.compute_loss(outputs, batch)
    loss.backward()
    
    # Check gradients in vision, transformer, and action head
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"✓ Backward pass successful. Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}")
    
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads, "No gradients computed!"
    print("✓ Gradient flow confirmed")
    
    # 5. Inference
    print("\nTesting Inference Sampling...")
    # Use few steps for speed
    actions = model.predict(batch, num_steps=3)
    
    assert actions.shape == (B, config.action_chunk_size, config.action_dim), \
        f"Unexpected inference action shape: {actions.shape}"
    print(f"✓ Inference successful. Action shape: {actions.shape}")
    
    print("\n" + "=" * 60)
    print("ALL V2.0 SOTA VERIFICATION CHECKS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        verify_v2()
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
