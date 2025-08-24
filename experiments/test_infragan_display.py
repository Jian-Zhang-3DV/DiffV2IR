#!/usr/bin/env python3
"""
Test InfraGAN display to debug the issue
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def test_infragan_display():
    """Test InfraGAN image loading and display"""
    
    sample_images = ['DJI_0061', 'DJI_0082', 'DJI_0271']
    infragan_dir = "/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/InfraGAN/results/infragan_vedai/inference_results"
    
    fig, axes = plt.subplots(2, len(sample_images), figsize=(18, 12))
    
    for i, img_name in enumerate(sample_images):
        # Test path construction
        infragan_filename = f"{img_name}_infrared.png"
        infragan_path = os.path.join(infragan_dir, infragan_filename)
        
        print(f"Testing {img_name}:")
        print(f"  Filename: {infragan_filename}")
        print(f"  Full path: {infragan_path}")
        print(f"  Exists: {os.path.exists(infragan_path)}")
        
        if os.path.exists(infragan_path):
            # Load and display original
            infragan_img = cv2.imread(infragan_path, cv2.IMREAD_GRAYSCALE)
            print(f"  Image loaded: {infragan_img is not None}")
            print(f"  Image shape: {infragan_img.shape}")
            print(f"  Image range: [{infragan_img.min()}, {infragan_img.max()}]")
            print(f"  Image mean: {infragan_img.mean():.1f}")
            
            axes[0, i].imshow(infragan_img, cmap='gray', vmin=0, vmax=255)
            axes[0, i].set_title(f'InfraGAN - {img_name}\nOriginal', fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
            
            # Apply CLAHE enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            infragan_enhanced = clahe.apply(infragan_img)
            
            axes[1, i].imshow(infragan_enhanced, cmap='gray', vmin=0, vmax=255)
            axes[1, i].set_title(f'InfraGAN - {img_name}\nCLAHE Enhanced', fontsize=12, fontweight='bold')
            axes[1, i].axis('off')
            
        else:
            print(f"  ERROR: File not found!")
            axes[0, i].text(0.5, 0.5, f'File not found:\n{infragan_filename}', ha='center', va='center')
            axes[0, i].set_title(f'InfraGAN - {img_name}\nERROR', fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
            
            axes[1, i].text(0.5, 0.5, 'Cannot enhance\nFile missing', ha='center', va='center')
            axes[1, i].set_title(f'InfraGAN Enhanced - {img_name}\nERROR', fontsize=12, fontweight='bold')
            axes[1, i].axis('off')
        
        print()
    
    plt.tight_layout()
    plt.savefig('test_infragan_display.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Test display saved to: test_infragan_display.png")

if __name__ == '__main__':
    test_infragan_display()