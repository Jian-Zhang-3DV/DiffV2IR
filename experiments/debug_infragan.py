#!/usr/bin/env python3
"""
Debug InfraGAN visualization issues
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def debug_infragan_image():
    """Debug InfraGAN image display issues"""
    
    infragan_path = "/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/InfraGAN/results/infragan_vedai/inference_results/DJI_0061_infrared.png"
    
    # Load image
    img = cv2.imread(infragan_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"Image shape: {img.shape}")
    print(f"Data type: {img.dtype}")
    print(f"Min value: {img.min()}")
    print(f"Max value: {img.max()}")
    print(f"Mean value: {img.mean():.2f}")
    print(f"Std value: {img.std():.2f}")
    
    # Check unique values
    unique_vals = np.unique(img)
    print(f"Number of unique values: {len(unique_vals)}")
    print(f"First 10 unique values: {unique_vals[:10]}")
    print(f"Last 10 unique values: {unique_vals[-10:]}")
    
    # Check histogram
    hist, bins = np.histogram(img.flatten(), bins=50)
    print(f"Histogram peaks at bins: {bins[np.argmax(hist)]:.1f}")
    
    # Create different visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original
    axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title(f'Original (0-255)\nMean: {img.mean():.1f}')
    axes[0, 0].axis('off')
    
    # Auto-scaled
    axes[0, 1].imshow(img, cmap='gray')
    axes[0, 1].set_title('Auto-scaled')
    axes[0, 1].axis('off')
    
    # Percentile scaling
    p1, p99 = np.percentile(img, [1, 99])
    axes[0, 2].imshow(img, cmap='gray', vmin=p1, vmax=p99)
    axes[0, 2].set_title(f'1-99% percentile\n({p1:.1f}-{p99:.1f})')
    axes[0, 2].axis('off')
    
    # Histogram equalized
    img_eq = cv2.equalizeHist(img)
    axes[1, 0].imshow(img_eq, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Histogram Equalized')
    axes[1, 0].axis('off')
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    axes[1, 1].imshow(img_clahe, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('CLAHE Enhanced')
    axes[1, 1].axis('off')
    
    # Histogram
    axes[1, 2].hist(img.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 2].set_title('Pixel Value Distribution')
    axes[1, 2].set_xlabel('Pixel Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('debug_infragan_display.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Debug visualization saved to: debug_infragan_display.png")

if __name__ == '__main__':
    debug_infragan_image()