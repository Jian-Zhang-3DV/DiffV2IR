#!/usr/bin/env python3
"""
Create final visualizations with proper InfraGAN display and much larger sizes
"""
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd

def create_unified_model_comparison_final():
    """Create unified visualization with proper InfraGAN display and larger sizes"""
    # Model performance data
    models = ['sRGB-TIR-02', 'PID', 'DiffV2IR', 'InfraGAN']
    psnr_values = [11.04, 10.75, 9.56, 8.86]
    ssim_values = [0.4056, 0.4272, 0.3886, 0.0154]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Create figure with much larger size
    fig = plt.figure(figsize=(32, 18))
    gs = GridSpec(2, 4, height_ratios=[1.2, 2.5], width_ratios=[1.5, 1, 1, 1])
    
    # Left side: Metrics comparison
    ax_metrics = fig.add_subplot(gs[:, 0])
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax_metrics.bar(x - width/2, psnr_values, width, label='PSNR (dB)', 
                          color=colors, alpha=0.8)
    ax2 = ax_metrics.twinx()
    bars2 = ax2.bar(x + width/2, ssim_values, width, label='SSIM', 
                   color=colors, alpha=0.6)
    
    ax_metrics.set_xlabel('Models', fontsize=16, fontweight='bold')
    ax_metrics.set_ylabel('PSNR (dB)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('SSIM', fontsize=16, fontweight='bold')
    ax_metrics.set_title('Model Performance Comparison\n(4 Models on xmu Aerial Dataset)', fontsize=18, fontweight='bold')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(models, rotation=15, ha='right', fontsize=14)
    
    # Add value labels on bars
    for bar, val in zip(bars1, psnr_values):
        ax_metrics.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    for bar, val in zip(bars2, ssim_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax_metrics.legend(loc='upper left', fontsize=14)
    ax2.legend(loc='upper right', fontsize=14)
    ax_metrics.grid(True, alpha=0.3)
    ax_metrics.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    
    # Right side: Visual comparisons for 3 representative samples
    sample_images = ['DJI_0061', 'DJI_0082', 'DJI_0271']
    
    for idx, img_name in enumerate(sample_images):
        # RGB Input (top row)
        ax_rgb = fig.add_subplot(gs[0, idx+1])
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        if os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            ax_rgb.imshow(rgb_img)
        ax_rgb.set_title(f'RGB Input\n{img_name}', fontsize=14, fontweight='bold')
        ax_rgb.axis('off')
        
        # Model outputs comparison (bottom row) - MUCH LARGER SIZE
        ax_models = fig.add_subplot(gs[1, idx+1])
        
        # Create 2x3 grid for all models including GT
        n_rows, n_cols = 2, 3
        fig_height = 10  # Increased from 6
        fig_width = 15   # Increased from 9
        
        sub_fig, sub_axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        # GT IR (top left)
        gt_path = f"data_for_diffv2ir/ground_truth/{img_name}.png"
        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            sub_axes[0, 0].imshow(gt_img, cmap='gray', vmin=0, vmax=255)
        sub_axes[0, 0].set_title('Ground Truth IR', fontsize=14, fontweight='bold')
        sub_axes[0, 0].axis('off')
        
        # DiffV2IR (top middle)
        diffv2ir_path = f"data_for_diffv2ir/output/{img_name}.png"
        if os.path.exists(diffv2ir_path):
            diffv2ir_img = cv2.imread(diffv2ir_path, cv2.IMREAD_GRAYSCALE)
            sub_axes[0, 1].imshow(diffv2ir_img, cmap='gray', vmin=0, vmax=255)
        sub_axes[0, 1].set_title('DiffV2IR', fontsize=14, fontweight='bold')
        sub_axes[0, 1].axis('off')
        
        # InfraGAN (top right) - FIX VISUALIZATION
        infragan_path = f"/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/InfraGAN/results/infragan_vedai/inference_results/{img_name.replace('.png', '_infrared.png')}"
        if os.path.exists(infragan_path):
            infragan_img = cv2.imread(infragan_path, cv2.IMREAD_GRAYSCALE)
            # Use full dynamic range - do NOT clip
            sub_axes[0, 2].imshow(infragan_img, cmap='gray', vmin=0, vmax=255)
            # Add statistics for debugging
            print(f"InfraGAN {img_name}: min={infragan_img.min()}, max={infragan_img.max()}, mean={infragan_img.mean():.1f}")
        else:
            sub_axes[0, 2].text(0.5, 0.5, 'InfraGAN\nNot Found', ha='center', va='center', fontsize=12)
        sub_axes[0, 2].set_title('InfraGAN', fontsize=14, fontweight='bold')
        sub_axes[0, 2].axis('off')
        
        # PID (bottom left) - placeholder
        sub_axes[1, 0].text(0.5, 0.5, 'PID\n(Previous Results)', ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))
        sub_axes[1, 0].set_title('PID', fontsize=14, fontweight='bold')
        sub_axes[1, 0].axis('off')
        
        # sRGB-TIR (bottom middle) - placeholder
        sub_axes[1, 1].text(0.5, 0.5, 'sRGB-TIR\n(Previous Results)', ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.3))
        sub_axes[1, 1].set_title('sRGB-TIR', fontsize=14, fontweight='bold')
        sub_axes[1, 1].axis('off')
        
        # Performance summary (bottom right)
        perf_text = f"Performance Summary:\n\n"
        perf_text += f"sRGB-TIR: 11.04dB, 0.406 SSIM\n"
        perf_text += f"PID: 10.75dB, 0.427 SSIM\n"
        perf_text += f"DiffV2IR: 9.56dB, 0.389 SSIM\n"
        perf_text += f"InfraGAN: 8.86dB, 0.015 SSIM\n\n"
        perf_text += f"Best PSNR: sRGB-TIR\n"
        perf_text += f"Best SSIM: PID"
        
        sub_axes[1, 2].text(0.05, 0.95, perf_text, ha='left', va='top', fontsize=11, 
                           bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.8),
                           transform=sub_axes[1, 2].transAxes)
        sub_axes[1, 2].set_title('Performance Summary', fontsize=14, fontweight='bold')
        sub_axes[1, 2].axis('off')
        
        # Save and embed
        sub_fig.tight_layout()
        temp_file = f'temp_{img_name}_models_final.png'
        sub_fig.savefig(temp_file, dpi=200, bbox_inches='tight')
        plt.close(sub_fig)
        
        # Load and display
        temp_img = cv2.imread(temp_file)
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        ax_models.imshow(temp_img)
        ax_models.set_title(f'Model Outputs Comparison', fontsize=14, fontweight='bold')
        ax_models.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/unified_model_comparison_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Clean up temp files
    for img_name in sample_images:
        temp_file = f'temp_{img_name}_models_final.png'
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("Final unified model comparison saved to: figures/unified_model_comparison_final.png")

def create_comprehensive_visual_final():
    """Create comprehensive visual with much larger sizes and proper InfraGAN display"""
    sample_images = ['DJI_0061', 'DJI_0068', 'DJI_0082', 'DJI_0128', 'DJI_0271', 'DJI_0281']
    
    # Much larger figure size
    fig, axes = plt.subplots(4, len(sample_images), figsize=(36, 24))
    
    for i, img_name in enumerate(sample_images):
        # Row 1: RGB Input
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        if os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f'RGB Input\n{img_name}', fontsize=14, fontweight='bold')
        axes[0, i].axis('off')
        
        # Row 2: Ground Truth IR
        gt_path = f"data_for_diffv2ir/ground_truth/{img_name}.png"
        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            axes[1, i].imshow(gt_img, cmap='gray', vmin=0, vmax=255)
        axes[1, i].set_title('Ground Truth IR', fontsize=14, fontweight='bold')
        axes[1, i].axis('off')
        
        # Row 3: DiffV2IR Predicted IR
        pred_path = f"data_for_diffv2ir/output/{img_name}.png"
        if os.path.exists(pred_path):
            pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            axes[2, i].imshow(pred_img, cmap='gray', vmin=0, vmax=255)
            
            # Calculate PSNR if GT available
            if os.path.exists(gt_path):
                if pred_img.shape != gt_img.shape:
                    pred_resized = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
                else:
                    pred_resized = pred_img
                
                from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
                psnr_val = psnr(gt_img, pred_resized, data_range=255)
                ssim_val = ssim(gt_img, pred_resized, data_range=255)
                axes[2, i].set_title(f'DiffV2IR Predicted\nPSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.3f}', 
                                   fontsize=12, fontweight='bold')
            else:
                axes[2, i].set_title('DiffV2IR Predicted IR', fontsize=14, fontweight='bold')
        axes[2, i].axis('off')
        
        # Row 4: InfraGAN Result - PROPER DISPLAY
        infragan_path = f"/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/InfraGAN/results/infragan_vedai/inference_results/{img_name.replace('.png', '_infrared.png')}"
        if os.path.exists(infragan_path):
            infragan_img = cv2.imread(infragan_path, cv2.IMREAD_GRAYSCALE)
            
            # Display with full range - this should show the texture properly
            axes[3, i].imshow(infragan_img, cmap='gray', vmin=0, vmax=255)
            
            # Calculate PSNR if GT available
            if os.path.exists(gt_path):
                if infragan_img.shape != gt_img.shape:
                    infragan_resized = cv2.resize(infragan_img, (gt_img.shape[1], gt_img.shape[0]))
                else:
                    infragan_resized = infragan_img
                
                psnr_val = psnr(gt_img, infragan_resized, data_range=255)
                ssim_val = ssim(gt_img, infragan_resized, data_range=255)
                axes[3, i].set_title(f'InfraGAN\nPSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.3f}', 
                                   fontsize=12, fontweight='bold')
                
                # Debug info
                print(f"InfraGAN {img_name}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.3f}, "
                      f"range=[{infragan_img.min()}-{infragan_img.max()}], mean={infragan_img.mean():.1f}")
            else:
                axes[3, i].set_title('InfraGAN', fontsize=14, fontweight='bold')
        else:
            axes[3, i].text(0.5, 0.5, 'InfraGAN\nNot Available', ha='center', va='center', fontsize=12)
            axes[3, i].set_title('InfraGAN', fontsize=14, fontweight='bold')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/comprehensive_visual_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Final comprehensive visual saved to: figures/comprehensive_visual_final.png")

def create_infragan_specific_analysis():
    """Create specific analysis showing InfraGAN characteristics"""
    sample_images = ['DJI_0061', 'DJI_0082', 'DJI_0271']
    
    fig, axes = plt.subplots(3, len(sample_images), figsize=(18, 15))
    
    for i, img_name in enumerate(sample_images):
        # Row 1: RGB Input
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        if os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f'RGB Input - {img_name}', fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
        
        # Row 2: InfraGAN with enhanced contrast
        infragan_path = f"/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/InfraGAN/results/infragan_vedai/inference_results/{img_name.replace('.png', '_infrared.png')}"
        if os.path.exists(infragan_path):
            infragan_img = cv2.imread(infragan_path, cv2.IMREAD_GRAYSCALE)
            
            # Show with different contrast settings
            axes[1, i].imshow(infragan_img, cmap='gray', vmin=0, vmax=255)
            axes[1, i].set_title(f'InfraGAN (Standard Range)\nMin: {infragan_img.min()}, Max: {infragan_img.max()}', 
                               fontsize=10, fontweight='bold')
        axes[1, i].axis('off')
        
        # Row 3: InfraGAN with histogram equalization
        if os.path.exists(infragan_path):
            # Apply histogram equalization to enhance visibility
            infragan_eq = cv2.equalizeHist(infragan_img)
            axes[2, i].imshow(infragan_eq, cmap='gray', vmin=0, vmax=255)
            axes[2, i].set_title(f'InfraGAN (Histogram Equalized)\nEnhanced Contrast', 
                               fontsize=10, fontweight='bold')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/infragan_specific_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("InfraGAN specific analysis saved to: figures/infragan_specific_analysis.png")

def main():
    print("Creating final visualizations with proper InfraGAN display and larger sizes...")
    
    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)
    
    print("1. Creating final unified model comparison...")
    create_unified_model_comparison_final()
    
    print("2. Creating final comprehensive visual comparison...")
    create_comprehensive_visual_final()
    
    print("3. Creating InfraGAN specific analysis...")
    create_infragan_specific_analysis()
    
    print("\nAll final visualizations completed!")

if __name__ == '__main__':
    main()