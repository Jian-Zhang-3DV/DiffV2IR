"""
Utilities for loading models with custom paths.
"""

import os
import clip
from model_paths import MODEL_PATHS, check_model_exists

def load_clip_with_custom_path(version='ViT-L/14', device='cpu', jit=False):
    """
    Load CLIP model with custom path support.
    
    Args:
        version: CLIP model version (e.g., 'ViT-L/14')
        device: Device to load model on
        jit: Whether to use JIT compilation
    
    Returns:
        model, preprocess
    """
    # Check if we have a local CLIP model
    if version == 'ViT-L/14' and check_model_exists('clip'):
        local_path = MODEL_PATHS['clip']
        print(f"Loading CLIP model from local path: {local_path}")
        
        # Set CLIP download root to the directory containing our model
        model_dir = os.path.dirname(local_path)
        os.environ['CLIP_MODELS_PATH'] = model_dir
        
        # Try to load from local path
        try:
            import torch
            # Load the state dict directly
            model, preprocess = clip.load(version, device=device, jit=jit, download_root=model_dir)
            print("Successfully loaded CLIP from local path")
            return model, preprocess
        except Exception as e:
            print(f"Failed to load from local path: {e}")
            print("Falling back to default CLIP loading...")
    
    # Fall back to default CLIP loading
    return clip.load(version, device=device, jit=jit)