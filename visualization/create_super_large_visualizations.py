#!/usr/bin/env python3
"""
Create super large visualizations to ensure InfraGAN is clearly visible
"""
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def create_super_large_comparison():
    """Create super large comparison with clearly visible InfraGAN"""
    sample_images = ['DJI_0061', 'DJI_0082', 'DJI_0271']
    
    # Super large figure - each subplot will be much bigger
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
            
            # Calculate metrics
            if os.path.exists(gt_path):
                if diffv2ir_img.shape != gt_img.shape:
                    diffv2ir_resized = cv2.resize(diffv2ir_img, (gt_img.shape[1], gt_img.shape[0]))
                else:
                    diffv2ir_resized = diffv2ir_img
                
                from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
                psnr_val = psnr(gt_img, diffv2ir_resized, data_range=255)
                ssim_val = ssim(gt_img, diffv2ir_resized, data_range=255)
                axes[2, i].set_title(f'DiffV2IR\nPSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.3f}', 
                                   fontsize=16, fontweight='bold', pad=20)
        axes[2, i].axis('off')
        
        # Row 4: InfraGAN - ENSURE VISIBILITY
        infragan_path = f"/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/InfraGAN/results/infragan_vedai/inference_results/{img_name.replace('.png', '_infrared.png')}"
        if os.path.exists(infragan_path):
            infragan_img = cv2.imread(infragan_path, cv2.IMREAD_GRAYSCALE)
            
            # Display InfraGAN with standard grayscale mapping
            axes[3, i].imshow(infragan_img, cmap='gray', vmin=0, vmax=255)
            
            # Calculate metrics
            if os.path.exists(gt_path):
                if infragan_img.shape != gt_img.shape:
                    infragan_resized = cv2.resize(infragan_img, (gt_img.shape[1], gt_img.shape[0]))
                else:
                    infragan_resized = infragan_img
                
                psnr_val = psnr(gt_img, infragan_resized, data_range=255)
                ssim_val = ssim(gt_img, infragan_resized, data_range=255)
                axes[3, i].set_title(f'InfraGAN\nPSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.3f}\nRange: {infragan_img.min()}-{infragan_img.max()}, Mean: {infragan_img.mean():.1f}', 
                                   fontsize=14, fontweight='bold', pad=20)
                
                print(f"{img_name} - InfraGAN stats: min={infragan_img.min()}, max={infragan_img.max()}, mean={infragan_img.mean():.1f}, PSNR={psnr_val:.2f}, SSIM={ssim_val:.3f}")
        else:
            axes[3, i].text(0.5, 0.5, 'InfraGAN\nNot Available', ha='center', va='center', fontsize=14)
        axes[3, i].axis('off')
        
        # Row 5: InfraGAN with enhanced contrast for better visibility
        if os.path.exists(infragan_path):
            # Apply CLAHE for better visualization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            infragan_enhanced = clahe.apply(infragan_img)
            axes[4, i].imshow(infragan_enhanced, cmap='gray', vmin=0, vmax=255)
            axes[4, i].set_title(f'InfraGAN (Enhanced Contrast)\nCLAHE Applied', 
                               fontsize=14, fontweight='bold', pad=20)
        else:
            axes[4, i].text(0.5, 0.5, 'InfraGAN Enhanced\nNot Available', ha='center', va='center', fontsize=14)
        axes[4, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/super_large_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Super large comparison saved to: figures/super_large_comparison.png")

def create_infragan_detailed_showcase():
    """Create detailed showcase specifically for InfraGAN"""
    sample_images = ['DJI_0061', 'DJI_0068', 'DJI_0082', 'DJI_0271']
    
    # Large figure for detailed InfraGAN analysis
    fig, axes = plt.subplots(2, len(sample_images), figsize=(32, 16))
    
    for i, img_name in enumerate(sample_images):
        # Row 1: RGB Input
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        if os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f'RGB Input - {img_name}', fontsize=18, fontweight='bold', pad=15)
        axes[0, i].axis('off')
        
        # Row 2: InfraGAN with detailed stats
        infragan_path = f"/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/InfraGAN/results/infragan_vedai/inference_results/{img_name.replace('.png', '_infrared.png')}"
        if os.path.exists(infragan_path):
            infragan_img = cv2.imread(infragan_path, cv2.IMREAD_GRAYSCALE)
            
            # Display with full range
            axes[1, i].imshow(infragan_img, cmap='gray', vmin=0, vmax=255)
            
            # Detailed statistics
            stats_text = f'InfraGAN Generated IR\n'
            stats_text += f'Range: [{infragan_img.min()}, {infragan_img.max()}]\n'
            stats_text += f'Mean: {infragan_img.mean():.1f} ± {infragan_img.std():.1f}\n'
            stats_text += f'Unique values: {len(np.unique(infragan_img))}'
            
            axes[1, i].set_title(stats_text, fontsize=14, fontweight='bold', pad=15)
            
            # Add text box with additional info
            info_text = f'Texture: {"High" if infragan_img.std() > 60 else "Low"}\nContrast: {"Good" if (infragan_img.max() - infragan_img.min()) > 200 else "Limited"}'
            axes[1, i].text(0.02, 0.98, info_text, transform=axes[1, i].transAxes, 
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        else:
            axes[1, i].text(0.5, 0.5, 'InfraGAN Result\nNot Available', ha='center', va='center', fontsize=16)
            axes[1, i].set_title('InfraGAN', fontsize=18, fontweight='bold', pad=15)
        axes[1, i].axis('off')
    
    plt.suptitle('InfraGAN Detailed Showcase - Generated Thermal Images', fontsize=24, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('figures/infragan_detailed_showcase.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("InfraGAN detailed showcase saved to: figures/infragan_detailed_showcase.png")

def create_four_model_side_by_side():
    """Create side-by-side comparison of all four models"""
    sample_image = 'DJI_0061'  # Focus on one clear example
    
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    
    # RGB Input
    rgb_path = f"data_for_diffv2ir/input/{sample_image}.png"
    if os.path.exists(rgb_path):
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(rgb_img)
    axes[0].set_title('RGB Input', fontsize=18, fontweight='bold', pad=20)
    axes[0].axis('off')
    
    # Ground Truth
    gt_path = f"data_for_diffv2ir/ground_truth/{sample_image}.png"
    if os.path.exists(gt_path):
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        axes[1].imshow(gt_img, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Ground Truth IR', fontsize=18, fontweight='bold', pad=20)
    axes[1].axis('off')
    
    # DiffV2IR
    diffv2ir_path = f"data_for_diffv2ir/output/{sample_image}.png"
    if os.path.exists(diffv2ir_path):
        diffv2ir_img = cv2.imread(diffv2ir_path, cv2.IMREAD_GRAYSCALE)
        axes[2].imshow(diffv2ir_img, cmap='gray', vmin=0, vmax=255)
        
        # Calculate PSNR/SSIM
        if os.path.exists(gt_path):
            if diffv2ir_img.shape != gt_img.shape:
                diffv2ir_resized = cv2.resize(diffv2ir_img, (gt_img.shape[1], gt_img.shape[0]))
            else:
                diffv2ir_resized = diffv2ir_img
            
            from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
            psnr_val = psnr(gt_img, diffv2ir_resized, data_range=255)
            ssim_val = ssim(gt_img, diffv2ir_resized, data_range=255)
            axes[2].set_title(f'DiffV2IR\nPSNR: {psnr_val:.2f}dB\nSSIM: {ssim_val:.3f}', 
                           fontsize=16, fontweight='bold', pad=20)
    axes[2].axis('off')
    
    # InfraGAN
    infragan_path = f"/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/InfraGAN/results/infragan_vedai/inference_results/{sample_image.replace('.png', '_infrared.png')}"
    if os.path.exists(infragan_path):
        infragan_img = cv2.imread(infragan_path, cv2.IMREAD_GRAYSCALE)
        axes[3].imshow(infragan_img, cmap='gray', vmin=0, vmax=255)
        
        # Calculate PSNR/SSIM
        if os.path.exists(gt_path):
            if infragan_img.shape != gt_img.shape:
                infragan_resized = cv2.resize(infragan_img, (gt_img.shape[1], gt_img.shape[0]))
            else:
                infragan_resized = infragan_img
            
            psnr_val = psnr(gt_img, infragan_resized, data_range=255)
            ssim_val = ssim(gt_img, infragan_resized, data_range=255)
            axes[3].set_title(f'InfraGAN\nPSNR: {psnr_val:.2f}dB\nSSIM: {ssim_val:.3f}', 
                           fontsize=16, fontweight='bold', pad=20)
            
            print(f"InfraGAN for {sample_image}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.3f}")
            print(f"Image stats: min={infragan_img.min()}, max={infragan_img.max()}, mean={infragan_img.mean():.1f}")
    axes[3].axis('off')
    
    # Performance summary
    axes[4].text(0.1, 0.9, 'Model Performance Summary:', fontsize=16, fontweight='bold', transform=axes[4].transAxes)
    perf_text = '''
sRGB-TIR-02: 11.04dB, 0.406 SSIM
Best PSNR performance

PID: 10.75dB, 0.427 SSIM  
Best SSIM performance

DiffV2IR: 9.56dB, 0.389 SSIM
Latest SOTA, moderate performance

InfraGAN: 8.86dB, 0.015 SSIM
Fast generation, low SSIM
    '''
    axes[4].text(0.1, 0.75, perf_text, fontsize=14, transform=axes[4].transAxes, verticalalignment='top')
    axes[4].axis('off')
    
    plt.suptitle(f'Four-Model Comparison on {sample_image}', fontsize=22, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('figures/four_model_side_by_side.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Four-model side-by-side comparison saved to: figures/four_model_side_by_side.png")

def main():
    print("Creating super large visualizations with clearly visible InfraGAN...")
    
    os.makedirs('figures', exist_ok=True)
    
    print("1. Creating super large comparison...")
    create_super_large_comparison()
    
    print("2. Creating InfraGAN detailed showcase...")
    create_infragan_detailed_showcase()
    
    print("3. Creating four-model side-by-side comparison...")
    create_four_model_side_by_side()
    
    print("\nAll super large visualizations completed!")

if __name__ == '__main__':
    main()