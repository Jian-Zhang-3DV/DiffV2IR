#!/usr/bin/env python3
"""
Create comprehensive SAM masks analysis with multiple cases
Compare: GT visible, GT infrared, Original (no SAM), SAM-enhanced
"""
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def create_comprehensive_comparison():
    """Create comprehensive comparison with multiple cases"""
    # Select more representative images
    sample_images = ['DJI_0061', 'DJI_0068', 'DJI_0082', 'DJI_0271', 'DJI_0128', 'DJI_0281']
    
    fig, axes = plt.subplots(5, len(sample_images), figsize=(24, 20))
    
    for i, img_name in enumerate(sample_images):
        # Row 1: RGB Input (GT Visible)
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        if os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(rgb_img)
            axes[0, i].set_title(f'GT Visible - {img_name}', fontsize=10)
        else:
            axes[0, i].text(0.5, 0.5, 'Visible\nNot Found', ha='center', va='center')
        axes[0, i].axis('off')
        
        # Row 2: Ground Truth IR
        gt_path = f"data_for_diffv2ir/ground_truth/{img_name}.png"
        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            axes[1, i].imshow(gt_img, cmap='gray', vmin=0, vmax=255)
            axes[1, i].set_title(f'GT Infrared', fontsize=10)
        else:
            axes[1, i].text(0.5, 0.5, 'GT Infrared\nNot Found', ha='center', va='center')
        axes[1, i].axis('off')
        
        # Row 3: SAM Mask
        sam_mask_path = f"data_for_diffv2ir/sam_masks/{img_name}_mask.png"
        if os.path.exists(sam_mask_path):
            sam_mask = cv2.imread(sam_mask_path, cv2.IMREAD_GRAYSCALE)
            axes[2, i].imshow(sam_mask, cmap='gray', vmin=0, vmax=255)
            coverage = np.mean(sam_mask > 0) * 100
            axes[2, i].set_title(f'SAM Mask\nCoverage {coverage:.1f}%', fontsize=10)
        else:
            axes[2, i].text(0.5, 0.5, 'SAM Mask\nNot Found', ha='center', va='center')
        axes[2, i].axis('off')
        
        # Row 4: Original result (without SAM)
        orig_path = f"data_for_diffv2ir/output/{img_name}.png"
        if os.path.exists(orig_path):
            orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
            axes[3, i].imshow(orig_img, cmap='gray', vmin=0, vmax=255)
            axes[3, i].set_title(f'Original DiffV2IR\n(White Mask)', fontsize=10)
        else:
            axes[3, i].text(0.5, 0.5, 'Original Result\nNot Found', ha='center', va='center')
        axes[3, i].axis('off')
        
        # Row 5: SAM-enhanced result
        sam_path = f"data_for_diffv2ir/output_sam_20steps/{img_name}.png"
        if os.path.exists(sam_path):
            sam_img = cv2.imread(sam_path, cv2.IMREAD_GRAYSCALE)
            axes[4, i].imshow(sam_img, cmap='gray', vmin=0, vmax=255)
            axes[4, i].set_title(f'SAM-Enhanced', fontsize=10)
        else:
            axes[4, i].text(0.5, 0.5, 'SAM Result\nNot Found', ha='center', va='center')
        axes[4, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/comprehensive_sam_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Comprehensive SAM comparison saved to: figures/comprehensive_sam_comparison.png")

def create_sam_mask_analysis():
    """Create detailed SAM mask analysis visualization"""
    sample_images = ['DJI_0061', 'DJI_0068', 'DJI_0082']
    
    fig, axes = plt.subplots(3, len(sample_images), figsize=(15, 12))
    
    for i, img_name in enumerate(sample_images):
        # Load RGB input
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f'Original RGB Input - {img_name}')
        axes[0, i].axis('off')
        
        # Show SAM mask with details
        sam_mask_path = f"data_for_diffv2ir/sam_masks/{img_name}_mask.png"
        if os.path.exists(sam_mask_path):
            sam_mask = cv2.imread(sam_mask_path, cv2.IMREAD_GRAYSCALE)
            axes[1, i].imshow(sam_mask, cmap='gray', vmin=0, vmax=255)
            
            # Calculate statistics
            total_pixels = sam_mask.size
            white_pixels = np.sum(sam_mask > 127)  # Consider > 127 as white
            coverage = (white_pixels / total_pixels) * 100
            unique_values = len(np.unique(sam_mask))
            
            axes[1, i].set_title(f'SAM Mask Details\nCoverage: {coverage:.1f}%\nUnique Values: {unique_values}')
        else:
            axes[1, i].text(0.5, 0.5, 'SAM Mask Not Found', ha='center', va='center')
        axes[1, i].axis('off')
        
        # Show mask overlay on RGB
        if os.path.exists(sam_mask_path):
            # Create colored overlay
            overlay = rgb_img.copy()
            sam_mask_3d = np.stack([sam_mask, sam_mask, sam_mask], axis=2)
            
            # Highlight segmented regions
            mask_binary = sam_mask > 127
            overlay[mask_binary] = overlay[mask_binary] * 0.7 + np.array([255, 0, 0]) * 0.3
            
            axes[2, i].imshow(overlay.astype(np.uint8))
            axes[2, i].set_title(f'SAM Overlay\nRed=Segmented Areas')
        else:
            axes[2, i].text(0.5, 0.5, 'Overlay Not Generated', ha='center', va='center')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/sam_mask_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("SAM mask detailed analysis saved to: figures/sam_mask_detailed_analysis.png")

def create_performance_degradation_analysis():
    """Create analysis showing why SAM masks caused performance degradation"""
    sample_images = ['DJI_0061', 'DJI_0082', 'DJI_0271']
    
    fig, axes = plt.subplots(4, len(sample_images), figsize=(18, 16))
    
    for i, img_name in enumerate(sample_images):
        # Load ground truth for comparison
        gt_path = f"data_for_diffv2ir/ground_truth/{img_name}.png"
        orig_path = f"data_for_diffv2ir/output/{img_name}.png"
        sam_path = f"data_for_diffv2ir/output_sam_20steps/{img_name}.png"
        
        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            axes[0, i].imshow(gt_img, cmap='gray', vmin=0, vmax=255)
            axes[0, i].set_title(f'Ground Truth IR - {img_name}')
        axes[0, i].axis('off')
        
        if os.path.exists(orig_path):
            orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
            axes[1, i].imshow(orig_img, cmap='gray', vmin=0, vmax=255)
            
            # Calculate PSNR with GT if available
            if os.path.exists(gt_path):
                # Resize if needed
                if orig_img.shape != gt_img.shape:
                    orig_img_resized = cv2.resize(orig_img, (gt_img.shape[1], gt_img.shape[0]))
                else:
                    orig_img_resized = orig_img
                
                from skimage.metrics import peak_signal_noise_ratio as psnr
                psnr_val = psnr(gt_img, orig_img_resized, data_range=255)
                axes[1, i].set_title(f'Original Version\nPSNR: {psnr_val:.2f} dB')
            else:
                axes[1, i].set_title(f'Original Version (White Mask)')
        axes[1, i].axis('off')
        
        if os.path.exists(sam_path):
            sam_img = cv2.imread(sam_path, cv2.IMREAD_GRAYSCALE)
            axes[2, i].imshow(sam_img, cmap='gray', vmin=0, vmax=255)
            
            # Calculate PSNR with GT if available
            if os.path.exists(gt_path):
                # Resize if needed
                if sam_img.shape != gt_img.shape:
                    sam_img_resized = cv2.resize(sam_img, (gt_img.shape[1], gt_img.shape[0]))
                else:
                    sam_img_resized = sam_img
                
                from skimage.metrics import peak_signal_noise_ratio as psnr
                psnr_val = psnr(gt_img, sam_img_resized, data_range=255)
                axes[2, i].set_title(f'SAM-Enhanced Version\nPSNR: {psnr_val:.2f} dB')
            else:
                axes[2, i].set_title(f'SAM-Enhanced Version')
        axes[2, i].axis('off')
        
        # Show difference between original and SAM
        if os.path.exists(orig_path) and os.path.exists(sam_path):
            orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
            sam_img = cv2.imread(sam_path, cv2.IMREAD_GRAYSCALE)
            
            # Ensure same size
            if orig_img.shape != sam_img.shape:
                sam_img = cv2.resize(sam_img, (orig_img.shape[1], orig_img.shape[0]))
            
            diff = np.abs(orig_img.astype(float) - sam_img.astype(float))
            axes[3, i].imshow(diff, cmap='hot', vmin=0, vmax=255)
            axes[3, i].set_title(f'Difference Map\nMax Diff: {np.max(diff):.0f}\nMean Diff: {np.mean(diff):.1f}')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/performance_degradation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Performance degradation analysis saved to: figures/performance_degradation_analysis.png")

def main():
    print("Creating comprehensive SAM analysis visualizations...")
    create_comprehensive_comparison()
    create_sam_mask_analysis()
    create_performance_degradation_analysis()
    print("All SAM analysis visualizations completed!")

if __name__ == '__main__':
    main()