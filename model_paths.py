"""
Configuration file for model paths.
Users can customize these paths to avoid downloading to cache directories.
"""

import os

# Base directory for all models (can be customized)
MODEL_BASE_DIR = os.environ.get('DIFFV2IR_MODEL_DIR', 'models')

# Ensure base directory exists
os.makedirs(MODEL_BASE_DIR, exist_ok=True)

# Model paths
MODEL_PATHS = {
    # BLIP model for image captioning
    'blip': os.path.join(MODEL_BASE_DIR, 'blip', 'model_base_caption_capfilt_large.pth'),
    
    # CLIP model for vision-language understanding
    'clip': os.path.join(MODEL_BASE_DIR, 'clip', 'ViT-L-14.pt'),
    
    # SAM model for segmentation (optional)
    'sam': os.path.join(MODEL_BASE_DIR, 'sam', 'sam_vit_h_4b8939.pth'),
    
    # DiffV2IR checkpoint
    'diffv2ir': 'pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt',
}

# Download URLs (for reference)
DOWNLOAD_URLS = {
    'blip': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth',
    'clip': 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt',
    'sam': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
}

def get_model_path(model_name):
    """Get the local path for a model."""
    return MODEL_PATHS.get(model_name)

def get_download_url(model_name):
    """Get the download URL for a model."""
    return DOWNLOAD_URLS.get(model_name)

def check_model_exists(model_name):
    """Check if a model file exists locally."""
    path = get_model_path(model_name)
    if path and os.path.exists(path):
        return True
    return False

def download_model_if_needed(model_name):
    """Download a model if it doesn't exist locally."""
    import urllib.request
    
    if check_model_exists(model_name):
        print(f"Model {model_name} already exists at {get_model_path(model_name)}")
        return get_model_path(model_name)
    
    url = get_download_url(model_name)
    path = get_model_path(model_name)
    
    if not url or not path:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    print(f"Downloading {model_name} from {url} to {path}...")
    urllib.request.urlretrieve(url, path)
    print(f"Downloaded {model_name} successfully")
    
    return path