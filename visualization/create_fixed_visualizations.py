#!/usr/bin/env python3
"""
Create fixed visualizations with properly working InfraGAN display
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def create_fixed_comprehensive_comparison():
    """Create comprehensive comparison with properly working InfraGAN"""
    sample_images = ['DJI_0061', 'DJI_0082', 'DJI_0271']
    infragan_dir = "/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/InfraGAN/results/infragan_vedai/inference_results"
    
    # Large figure
    fig, axes = plt.subplots(5, len(sample_images), figsize=(24, 30))
    
    for i, img_name in enumerate(sample_images):
        # Row 1: RGB Input
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        if os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f'RGB Input - {img_name}', fontsize=16, fontweight='bold', pad=20)
        axes[0, i].axis('off')
        
        # Row 2: Ground Truth IR
        gt_path = f"data_for_diffv2ir/ground_truth/{img_name}.png"
        gt_img = None
        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            axes[1, i].imshow(gt_img, cmap='gray', vmin=0, vmax=255)
        axes[1, i].set_title('Ground Truth IR', fontsize=16, fontweight='bold', pad=20)
        axes[1, i].axis('off')
        
        # Row 3: DiffV2IR
        diffv2ir_path = f"data_for_diffv2ir/output/{img_name}.png"
        if os.path.exists(diffv2ir_path):
            diffv2ir_img = cv2.imread(diffv2ir_path, cv2.IMREAD_GRAYSCALE)
            axes[2, i].imshow(diffv2ir_img, cmap='gray', vmin=0, vmax=255)
            
            # Calculate metrics if GT available
            if gt_img is not None:
                if diffv2ir_img.shape != gt_img.shape:
                    diffv2ir_resized = cv2.resize(diffv2ir_img, (gt_img.shape[1], gt_img.shape[0]))
                else:
                    diffv2ir_resized = diffv2ir_img
                
                from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
                psnr_val = psnr(gt_img, diffv2ir_resized, data_range=255)
                ssim_val = ssim(gt_img, diffv2ir_resized, data_range=255)
                axes[2, i].set_title(f'DiffV2IR\nPSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.3f}', 
                                   fontsize=14, fontweight='bold', pad=20)
            else:
                axes[2, i].set_title('DiffV2IR', fontsize=16, fontweight='bold', pad=20)
        axes[2, i].axis('off')
        
        # Row 4: InfraGAN - FIXED PATH LOGIC
        infragan_filename = f"{img_name}_infrared.png"
        infragan_path = os.path.join(infragan_dir, infragan_filename)
        
        print(f"Processing InfraGAN for {img_name}:")
        print(f"  File: {infragan_filename}")
        print(f"  Path: {infragan_path}")
        print(f"  Exists: {os.path.exists(infragan_path)}")
        
        if os.path.exists(infragan_path):
            infragan_img = cv2.imread(infragan_path, cv2.IMREAD_GRAYSCALE)
            if infragan_img is not None:
                axes[3, i].imshow(infragan_img, cmap='gray', vmin=0, vmax=255)
                
                # Calculate metrics if GT available
                if gt_img is not None:
                    if infragan_img.shape != gt_img.shape:
                        infragan_resized = cv2.resize(infragan_img, (gt_img.shape[1], gt_img.shape[0]))
                    else:
                        infragan_resized = infragan_img
                    
                    psnr_val = psnr(gt_img, infragan_resized, data_range=255)
                    ssim_val = ssim(gt_img, infragan_resized, data_range=255)
                    axes[3, i].set_title(f'InfraGAN\nPSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.3f}', 
                                       fontsize=14, fontweight='bold', pad=20)
                    
                    print(f"  PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.3f}")
                    print(f"  Range: [{infragan_img.min()}, {infragan_img.max()}], Mean: {infragan_img.mean():.1f}")
                else:
                    axes[3, i].set_title('InfraGAN', fontsize=16, fontweight='bold', pad=20)
            else:
                print(f"  ERROR: Could not load image")
                axes[3, i].text(0.5, 0.5, 'Failed to load\nInfraGAN image', ha='center', va='center', fontsize=14)
                axes[3, i].set_title('InfraGAN - Error', fontsize=16, fontweight='bold', pad=20)
        else:
            print(f"  ERROR: File not found")
            axes[3, i].text(0.5, 0.5, 'InfraGAN\nNot Available', ha='center', va='center', fontsize=14)
            axes[3, i].set_title('InfraGAN - Missing', fontsize=16, fontweight='bold', pad=20)
        axes[3, i].axis('off')
        
        # Row 5: InfraGAN Enhanced - FIXED PATH LOGIC
        if os.path.exists(infragan_path):
            infragan_img = cv2.imread(infragan_path, cv2.IMREAD_GRAYSCALE)
            if infragan_img is not None:
                # Apply CLAHE enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                infragan_enhanced = clahe.apply(infragan_img)
                
                axes[4, i].imshow(infragan_enhanced, cmap='gray', vmin=0, vmax=255)
                axes[4, i].set_title(f'InfraGAN Enhanced\n(CLAHE Applied)', 
                                   fontsize=14, fontweight='bold', pad=20)
                print(f"  Enhanced version created successfully")
            else:
                axes[4, i].text(0.5, 0.5, 'Enhancement Failed\nCould not load image', ha='center', va='center', fontsize=12)
                axes[4, i].set_title('InfraGAN Enhanced - Error', fontsize=14, fontweight='bold', pad=20)
        else:
            axes[4, i].text(0.5, 0.5, 'InfraGAN Enhanced\nNot Available', ha='center', va='center', fontsize=12)
            axes[4, i].set_title('InfraGAN Enhanced - Missing', fontsize=14, fontweight='bold', pad=20)
        axes[4, i].axis('off')
        
        print()  # Empty line for readability
    
    plt.tight_layout()
    plt.savefig('figures/fixed_comprehensive_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Fixed comprehensive comparison saved to: figures/fixed_comprehensive_comparison.png")

def create_infragan_focus_comparison():
    """Create InfraGAN-focused comparison with detailed analysis"""
    sample_images = ['DJI_0061', 'DJI_0082', 'DJI_0271']
    infragan_dir = "/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/InfraGAN/results/infragan_vedai/inference_results"
    
    fig, axes = plt.subplots(3, len(sample_images), figsize=(24, 18))
    
    for i, img_name in enumerate(sample_images):
        # Row 1: RGB Input
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        if os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f'RGB Input - {img_name}', fontsize=18, fontweight='bold', pad=20)
        axes[0, i].axis('off')
        
        # Row 2: InfraGAN Original
        infragan_filename = f"{img_name}_infrared.png"
        infragan_path = os.path.join(infragan_dir, infragan_filename)
        
        if os.path.exists(infragan_path):
            infragan_img = cv2.imread(infragan_path, cv2.IMREAD_GRAYSCALE)
            if infragan_img is not None:
                axes[1, i].imshow(infragan_img, cmap='gray', vmin=0, vmax=255)
                
                # Add detailed statistics
                stats_text = f'InfraGAN Generated\n'
                stats_text += f'Range: [{infragan_img.min()}, {infragan_img.max()}]\n'
                stats_text += f'Mean: {infragan_img.mean():.1f} ± {infragan_img.std():.1f}\n'
                stats_text += f'Texture Quality: {"High" if infragan_img.std() > 60 else "Moderate"}'
                
                axes[1, i].set_title(stats_text, fontsize=12, fontweight='bold', pad=20)
        axes[1, i].axis('off')
        
        # Row 3: InfraGAN Enhanced with CLAHE
        if os.path.exists(infragan_path):
            infragan_img = cv2.imread(infragan_path, cv2.IMREAD_GRAYSCALE)
            if infragan_img is not None:
                # Apply CLAHE for better contrast
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                infragan_enhanced = clahe.apply(infragan_img)
                
                axes[2, i].imshow(infragan_enhanced, cmap='gray', vmin=0, vmax=255)
                axes[2, i].set_title('InfraGAN + CLAHE Enhancement\n(Improved Contrast)', 
                                   fontsize=14, fontweight='bold', pad=20)
                
                # Add enhancement info
                enhancement_info = f'CLAHE Applied\nClip Limit: 3.0\nGrid: 8×8'
                axes[2, i].text(0.02, 0.98, enhancement_info, transform=axes[2, i].transAxes,
                               fontsize=10, verticalalignment='top',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        axes[2, i].axis('off')
    
    plt.suptitle('InfraGAN Detailed Analysis - Thermal Image Generation from RGB', 
                 fontsize=20, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('figures/infragan_focus_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("InfraGAN focus comparison saved to: figures/infragan_focus_comparison.png")

def main():
    print("Creating fixed visualizations with working InfraGAN display...")
    
    os.makedirs('figures', exist_ok=True)
    
    print("1. Creating fixed comprehensive comparison...")
    create_fixed_comprehensive_comparison()
    
    print("2. Creating InfraGAN focus comparison...")
    create_infragan_focus_comparison()
    
    print("\nAll fixed visualizations completed!")

if __name__ == '__main__':
    main()