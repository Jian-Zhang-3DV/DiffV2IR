#!/usr/bin/env python3
"""
Analyze InfraGAN image dynamic range
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def analyze_infragan_range():
    """Analyze the pixel value distribution of InfraGAN images"""
    
    infragan_dir = "/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/InfraGAN/results/infragan_vedai/inference_results"
    sample_files = ['DJI_0061_infrared.png', 'DJI_0082_infrared.png', 'DJI_0271_infrared.png']
    
    fig, axes = plt.subplots(2, len(sample_files), figsize=(15, 10))
    
    for i, filename in enumerate(sample_files):
        filepath = os.path.join(infragan_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        # Original image
        axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(f'Original {filename}')
        axes[0, i].axis('off')
        
        # Histogram
        axes[1, i].hist(img.flatten(), bins=50, alpha=0.7, color='blue')
        axes[1, i].set_title(f'Pixel Distribution\nMin: {img.min()}, Max: {img.max()}, Mean: {img.mean():.1f}')
        axes[1, i].set_xlabel('Pixel Value')
        axes[1, i].set_ylabel('Frequency')
        
        print(f"{filename}:")
        print(f"  Min: {img.min()}, Max: {img.max()}")
        print(f"  Mean: {img.mean():.2f}, Std: {img.std():.2f}")
        print(f"  Percentiles: 1%={np.percentile(img, 1):.1f}, 99%={np.percentile(img, 99):.1f}")
        print()
    
    plt.tight_layout()
    plt.savefig('infragan_range_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("InfraGAN range analysis saved to: infragan_range_analysis.png")

if __name__ == '__main__':
    analyze_infragan_range()