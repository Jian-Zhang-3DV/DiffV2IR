#!/usr/bin/env python3
"""
Create visual comparison between original white masks and SAM masks
Also show the generated results to understand why SAM masks performed worse
"""
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def create_mask_comparison_visualization():
    """Create visual comparison between white masks and SAM masks"""
    # Select a few representative images
    sample_images = ['DJI_0061', 'DJI_0068', 'DJI_0082']
    
    fig, axes = plt.subplots(4, len(sample_images), figsize=(15, 16))
    
    for i, img_name in enumerate(sample_images):
        # Load RGB input
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f'RGB Input - {img_name}')
        axes[0, i].axis('off')
        
        # Load and show white mask (generate one since it doesn't exist)
        white_mask = np.ones((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8) * 255
        axes[1, i].imshow(white_mask, cmap='gray', vmin=0, vmax=255)
        axes[1, i].set_title(f'Original White Mask')
        axes[1, i].axis('off')
        
        # Load and show SAM mask
        sam_mask_path = f"data_for_diffv2ir/sam_masks/{img_name}_mask.png"
        if os.path.exists(sam_mask_path):
            sam_mask = cv2.imread(sam_mask_path, cv2.IMREAD_GRAYSCALE)
            axes[2, i].imshow(sam_mask, cmap='gray', vmin=0, vmax=255)
            coverage = np.mean(sam_mask > 0) * 100
            axes[2, i].set_title(f'SAM Mask\n({coverage:.1f}% coverage)')
            axes[2, i].axis('off')
        else:
            axes[2, i].text(0.5, 0.5, 'SAM Mask\nNot Found', ha='center', va='center')
            axes[2, i].axis('off')
        
        # Show difference between results
        orig_result_path = f"data_for_diffv2ir/output/{img_name}.png"
        sam_result_path = f"data_for_diffv2ir/output_sam_20steps/{img_name}.png"
        
        if os.path.exists(orig_result_path) and os.path.exists(sam_result_path):
            orig_result = cv2.imread(orig_result_path, cv2.IMREAD_GRAYSCALE)
            sam_result = cv2.imread(sam_result_path, cv2.IMREAD_GRAYSCALE)
            
            # Calculate difference
            if orig_result.shape == sam_result.shape:
                diff = np.abs(orig_result.astype(float) - sam_result.astype(float))
                axes[3, i].imshow(diff, cmap='hot', vmin=0, vmax=255)
                axes[3, i].set_title(f'|Original - SAM| Difference\nMax diff: {np.max(diff):.0f}')
            else:
                # Resize to match
                sam_result_resized = cv2.resize(sam_result, (orig_result.shape[1], orig_result.shape[0]))
                diff = np.abs(orig_result.astype(float) - sam_result_resized.astype(float))
                axes[3, i].imshow(diff, cmap='hot', vmin=0, vmax=255)
                axes[3, i].set_title(f'|Original - SAM| Difference\nMax diff: {np.max(diff):.0f}')
        else:
            axes[3, i].text(0.5, 0.5, 'Results\nNot Found', ha='center', va='center')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/mask_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Mask comparison visualization saved to: figures/mask_comparison_analysis.png")

def create_results_comparison_visualization():
    """Create side-by-side comparison of results"""
    sample_images = ['DJI_0061', 'DJI_0068', 'DJI_0082']
    
    fig, axes = plt.subplots(4, len(sample_images), figsize=(15, 16))
    
    for i, img_name in enumerate(sample_images):
        # Load RGB input
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f'RGB Input - {img_name}')
        axes[0, i].axis('off')
        
        # Load ground truth
        gt_path = f"data_for_diffv2ir/ground_truth/{img_name}.png"
        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            axes[1, i].imshow(gt_img, cmap='gray', vmin=0, vmax=255)
            axes[1, i].set_title(f'Ground Truth IR')
        else:
            axes[1, i].text(0.5, 0.5, 'GT Not Found', ha='center', va='center')
        axes[1, i].axis('off')
        
        # Load original result (white mask)
        orig_path = f"data_for_diffv2ir/output/{img_name}.png"
        if os.path.exists(orig_path):
            orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
            axes[2, i].imshow(orig_img, cmap='gray', vmin=0, vmax=255)
            axes[2, i].set_title(f'Original (White Mask)')
        else:
            axes[2, i].text(0.5, 0.5, 'Original\nNot Found', ha='center', va='center')
        axes[2, i].axis('off')
        
        # Load SAM result
        sam_path = f"data_for_diffv2ir/output_sam_20steps/{img_name}.png"
        if os.path.exists(sam_path):
            sam_img = cv2.imread(sam_path, cv2.IMREAD_GRAYSCALE)
            axes[3, i].imshow(sam_img, cmap='gray', vmin=0, vmax=255)
            axes[3, i].set_title(f'SAM-Enhanced')
        else:
            axes[3, i].text(0.5, 0.5, 'SAM Result\nNot Found', ha='center', va='center')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/results_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Results comparison visualization saved to: figures/results_comparison_analysis.png")

def main():
    print("Creating mask and results comparison visualizations...")
    create_mask_comparison_visualization()
    create_results_comparison_visualization()
    print("Analysis complete!")

if __name__ == '__main__':
    main()