#!/usr/bin/env python3
"""
Create improved visualizations with proper sizing and grayscale IR display
Include InfraGAN in the comparison
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

def create_unified_model_comparison_4models():
    """Create unified visualization with 4 models including InfraGAN"""
    # Model performance data (updated with InfraGAN)
    models = ['sRGB-TIR-02', 'PID', 'DiffV2IR', 'InfraGAN']
    psnr_values = [11.04, 10.75, 9.56, 8.86]
    ssim_values = [0.4056, 0.4272, 0.3886, 0.0154]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Create figure with better proportions
    fig = plt.figure(figsize=(24, 14))
    gs = GridSpec(2, 4, height_ratios=[1.2, 1.8], width_ratios=[1.2, 1, 1, 1])
    
    # Left side: Metrics comparison
    ax_metrics = fig.add_subplot(gs[:, 0])
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax_metrics.bar(x - width/2, psnr_values, width, label='PSNR (dB)', 
                          color=colors, alpha=0.8)
    ax2 = ax_metrics.twinx()
    bars2 = ax2.bar(x + width/2, ssim_values, width, label='SSIM', 
                   color=colors, alpha=0.6)
    
    ax_metrics.set_xlabel('Models', fontsize=14, fontweight='bold')
    ax_metrics.set_ylabel('PSNR (dB)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('SSIM', fontsize=14, fontweight='bold')
    ax_metrics.set_title('Model Performance Comparison\n(4 Models on xmu Aerial Dataset)', fontsize=16, fontweight='bold')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(models, rotation=15, ha='right')
    
    # Add value labels on bars
    for bar, val in zip(bars1, psnr_values):
        ax_metrics.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    for bar, val in zip(bars2, ssim_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax_metrics.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    ax_metrics.grid(True, alpha=0.3)
    
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
        ax_rgb.set_title(f'RGB Input\n{img_name}', fontsize=12, fontweight='bold')
        ax_rgb.axis('off')
        
        # Model outputs comparison (bottom row) - LARGER SIZE
        ax_models = fig.add_subplot(gs[1, idx+1])
        
        # Create 2x3 grid for all models including GT
        n_rows, n_cols = 2, 3
        fig_height = 6
        fig_width = 9
        
        sub_fig, sub_axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        # GT IR (top left)
        gt_path = f"data_for_diffv2ir/ground_truth/{img_name}.png"
        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            sub_axes[0, 0].imshow(gt_img, cmap='gray', vmin=0, vmax=255)
        sub_axes[0, 0].set_title('GT IR', fontsize=10, fontweight='bold')
        sub_axes[0, 0].axis('off')
        
        # DiffV2IR (top middle)
        diffv2ir_path = f"data_for_diffv2ir/output/{img_name}.png"
        if os.path.exists(diffv2ir_path):
            diffv2ir_img = cv2.imread(diffv2ir_path, cv2.IMREAD_GRAYSCALE)
            sub_axes[0, 1].imshow(diffv2ir_img, cmap='gray', vmin=0, vmax=255)
        sub_axes[0, 1].set_title('DiffV2IR', fontsize=10, fontweight='bold')
        sub_axes[0, 1].axis('off')
        
        # InfraGAN (top right)
        infragan_path = f"/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/InfraGAN/results/infragan_vedai/inference_results/{img_name.replace('.png', '_infrared.png')}"
        if os.path.exists(infragan_path):
            infragan_img = cv2.imread(infragan_path, cv2.IMREAD_GRAYSCALE)
            sub_axes[0, 2].imshow(infragan_img, cmap='gray', vmin=0, vmax=255)
        sub_axes[0, 2].set_title('InfraGAN', fontsize=10, fontweight='bold')
        sub_axes[0, 2].axis('off')
        
        # PID (bottom left) - placeholder
        sub_axes[1, 0].text(0.5, 0.5, 'PID\n(Previous)', ha='center', va='center', fontsize=10)
        sub_axes[1, 0].axis('off')
        
        # sRGB-TIR (bottom middle) - placeholder
        sub_axes[1, 1].text(0.5, 0.5, 'sRGB-TIR\n(Previous)', ha='center', va='center', fontsize=10)
        sub_axes[1, 1].axis('off')
        
        # Performance summary (bottom right)
        perf_text = f"Performance on {img_name}:\n"
        perf_text += f"Best PSNR: sRGB-TIR (11.04dB)\n"
        perf_text += f"Best SSIM: PID (0.4272)\n"
        perf_text += f"DiffV2IR: 9.56dB, 0.389\n"
        perf_text += f"InfraGAN: 8.86dB, 0.015"
        sub_axes[1, 2].text(0.05, 0.95, perf_text, ha='left', va='top', fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3),
                           transform=sub_axes[1, 2].transAxes)
        sub_axes[1, 2].axis('off')
        
        # Save and embed
        sub_fig.tight_layout()
        temp_file = f'temp_{img_name}_models_4.png'
        sub_fig.savefig(temp_file, dpi=150, bbox_inches='tight')
        plt.close(sub_fig)
        
        # Load and display
        temp_img = cv2.imread(temp_file)
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        ax_models.imshow(temp_img)
        ax_models.set_title(f'Model Outputs Comparison', fontsize=12, fontweight='bold')
        ax_models.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/unified_model_comparison_4models.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Clean up temp files
    for img_name in sample_images:
        temp_file = f'temp_{img_name}_models_4.png'
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("4-model unified comparison saved to: figures/unified_model_comparison_4models.png")

def create_comprehensive_visual_improved():
    """Create comprehensive DiffV2IR visual with larger IR images and grayscale display"""
    sample_images = ['DJI_0061', 'DJI_0068', 'DJI_0082', 'DJI_0128', 'DJI_0271', 'DJI_0281']
    
    fig, axes = plt.subplots(4, len(sample_images), figsize=(30, 20))
    
    for i, img_name in enumerate(sample_images):
        # Row 1: RGB Input
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        if os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f'RGB Input\n{img_name}', fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
        
        # Row 2: Ground Truth IR (GRAYSCALE, LARGER)
        gt_path = f"data_for_diffv2ir/ground_truth/{img_name}.png"
        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            axes[1, i].imshow(gt_img, cmap='gray', vmin=0, vmax=255)
        axes[1, i].set_title('Ground Truth IR', fontsize=12, fontweight='bold')
        axes[1, i].axis('off')
        
        # Row 3: DiffV2IR Predicted IR (GRAYSCALE, LARGER)
        pred_path = f"data_for_diffv2ir/output/{img_name}.png"
        if os.path.exists(pred_path):
            pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            axes[2, i].imshow(pred_img, cmap='gray', vmin=0, vmax=255)
            
            # Calculate PSNR if GT available
            if os.path.exists(gt_path):
                # Resize if needed for PSNR calculation
                if pred_img.shape != gt_img.shape:
                    pred_resized = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
                else:
                    pred_resized = pred_img
                
                from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
                psnr_val = psnr(gt_img, pred_resized, data_range=255)
                ssim_val = ssim(gt_img, pred_resized, data_range=255)
                axes[2, i].set_title(f'DiffV2IR Predicted\nPSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.3f}', 
                                   fontsize=11, fontweight='bold')
            else:
                axes[2, i].set_title('DiffV2IR Predicted IR', fontsize=12, fontweight='bold')
        axes[2, i].axis('off')
        
        # Row 4: InfraGAN Result (GRAYSCALE, LARGER)
        infragan_path = f"/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/InfraGAN/results/infragan_vedai/inference_results/{img_name.replace('.png', '_infrared.png')}"
        if os.path.exists(infragan_path):
            infragan_img = cv2.imread(infragan_path, cv2.IMREAD_GRAYSCALE)
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
                                   fontsize=11, fontweight='bold')
            else:
                axes[3, i].set_title('InfraGAN', fontsize=12, fontweight='bold')
        else:
            axes[3, i].text(0.5, 0.5, 'InfraGAN\nNot Available', ha='center', va='center', fontsize=10)
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/comprehensive_visual_improved.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Improved comprehensive visual saved to: figures/comprehensive_visual_improved.png")

def create_denoising_steps_ablation_improved():
    """Improved denoising steps ablation with larger IR images"""
    # Performance data
    steps_data = {
        '20 Steps': {'PSNR': 9.56, 'SSIM': 0.3886, 'Speed': '1x'},
        '50 Steps': {'PSNR': 8.86, 'SSIM': 0.3677, 'Speed': '2.5x'}
    }
    
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(2, 4, height_ratios=[1, 1.5])
    
    # Left: Metrics comparison
    ax_metrics = fig.add_subplot(gs[:, 0])
    
    steps = list(steps_data.keys())
    psnr_vals = [steps_data[s]['PSNR'] for s in steps]
    ssim_vals = [steps_data[s]['SSIM'] for s in steps]
    
    x = np.arange(len(steps))
    width = 0.35
    
    bars1 = ax_metrics.bar(x - width/2, psnr_vals, width, label='PSNR (dB)', 
                          color=['#2E86AB', '#F18F01'], alpha=0.8)
    ax2 = ax_metrics.twinx()
    bars2 = ax2.bar(x + width/2, ssim_vals, width, label='SSIM', 
                   color=['#2E86AB', '#F18F01'], alpha=0.6)
    
    ax_metrics.set_xlabel('Denoising Steps', fontsize=14, fontweight='bold')
    ax_metrics.set_ylabel('PSNR (dB)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('SSIM', fontsize=14, fontweight='bold')
    ax_metrics.set_title('DiffV2IR Denoising Steps\nAblation Study', fontsize=16, fontweight='bold')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(steps)
    
    # Add performance improvement annotations
    psnr_change = psnr_vals[0] - psnr_vals[1]
    ssim_change = ssim_vals[0] - ssim_vals[1]
    ax_metrics.annotate(f'20 Steps Better:\nPSNR: +{psnr_change:.2f}dB (+{psnr_change/psnr_vals[1]*100:.1f}%)\nSSIM: +{ssim_change:.3f} (+{ssim_change/ssim_vals[1]*100:.1f}%)\nSpeed: 2.5x faster', 
                       xy=(0.5, max(psnr_vals) + 0.5), xytext=(0.5, max(psnr_vals) + 2),
                       ha='center', fontsize=11, fontweight='bold', color='green',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.3),
                       arrowprops=dict(arrowstyle='->', color='green'))
    
    # Add value labels
    for bar, val in zip(bars1, psnr_vals):
        ax_metrics.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    for bar, val in zip(bars2, ssim_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax_metrics.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax_metrics.grid(True, alpha=0.3)
    
    # Right: Visual comparisons - LARGER IR IMAGES
    sample_images = ['DJI_0061', 'DJI_0068', 'DJI_0082']
    
    for idx, img_name in enumerate(sample_images):
        # RGB Input (top row)
        ax_rgb = fig.add_subplot(gs[0, idx+1])
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        if os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            ax_rgb.imshow(rgb_img)
        ax_rgb.set_title(f'RGB Input - {img_name}', fontsize=12, fontweight='bold')
        ax_rgb.axis('off')
        
        # Steps comparison (bottom row) - LARGER
        ax_comp = fig.add_subplot(gs[1, idx+1])
        
        # Create side-by-side comparison
        steps20_path = f"data_for_diffv2ir/output/{img_name}.png"  # 20 steps
        steps50_path = f"data_for_diffv2ir/output_50steps/{img_name}.png"  # 50 steps (if exists)
        
        if os.path.exists(steps20_path):
            img_20 = cv2.imread(steps20_path, cv2.IMREAD_GRAYSCALE)
            if os.path.exists(steps50_path):
                img_50 = cv2.imread(steps50_path, cv2.IMREAD_GRAYSCALE)
                # Create side-by-side comparison
                if img_20.shape == img_50.shape:
                    comparison = np.hstack([img_20, img_50])
                    ax_comp.imshow(comparison, cmap='gray', vmin=0, vmax=255)
                    # Add dividing line
                    ax_comp.axvline(x=img_20.shape[1], color='red', linewidth=3)
                    ax_comp.text(img_20.shape[1]//2, 20, '20 Steps (Better)', ha='center', color='white', 
                               fontweight='bold', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.8))
                    ax_comp.text(img_20.shape[1] + img_50.shape[1]//2, 20, '50 Steps', ha='center', color='white', 
                               fontweight='bold', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.8))
                else:
                    ax_comp.imshow(img_20, cmap='gray', vmin=0, vmax=255)
                    ax_comp.text(0.5, 0.95, '20 Steps Only', transform=ax_comp.transAxes, ha='center', 
                               color='white', fontweight='bold')
            else:
                ax_comp.imshow(img_20, cmap='gray', vmin=0, vmax=255)
                ax_comp.text(0.5, 0.95, '20 Steps', transform=ax_comp.transAxes, ha='center', 
                           color='white', fontweight='bold')
        
        ax_comp.set_title('20 Steps vs 50 Steps Comparison', fontsize=12, fontweight='bold')
        ax_comp.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/denoising_steps_ablation_improved.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Improved denoising steps ablation saved to: figures/denoising_steps_ablation_improved.png")

def create_mask_ablation_improved():
    """Improved mask ablation with larger IR images"""
    # Performance data
    mask_data = {
        'White Mask (Original)': {'PSNR': 9.60, 'SSIM': 0.4140},
        'SAM Mask': {'PSNR': 7.73, 'SSIM': 0.3968}
    }
    
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 4, height_ratios=[0.8, 1.2, 1.2])
    
    # Top: Metrics comparison  
    ax_metrics = fig.add_subplot(gs[0, :2])
    
    mask_types = list(mask_data.keys())
    psnr_vals = [mask_data[m]['PSNR'] for m in mask_types]
    ssim_vals = [mask_data[m]['SSIM'] for m in mask_types]
    
    x = np.arange(len(mask_types))
    width = 0.35
    
    bars1 = ax_metrics.bar(x - width/2, psnr_vals, width, label='PSNR (dB)', 
                          color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax2 = ax_metrics.twinx()
    bars2 = ax2.bar(x + width/2, ssim_vals, width, label='SSIM', 
                   color=['#2E86AB', '#A23B72'], alpha=0.6)
    
    ax_metrics.set_xlabel('Mask Type', fontsize=14, fontweight='bold')
    ax_metrics.set_ylabel('PSNR (dB)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('SSIM', fontsize=14, fontweight='bold')
    ax_metrics.set_title('SAM Mask vs White Mask\nPerformance Comparison', fontsize=16, fontweight='bold')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(['White Mask', 'SAM Mask'])
    
    # Add performance degradation annotation
    psnr_degradation = psnr_vals[0] - psnr_vals[1]
    ssim_degradation = ssim_vals[0] - ssim_vals[1]
    ax_metrics.annotate(f'SAM Mask Performance Drop:\nPSNR: -{psnr_degradation:.2f}dB (-{psnr_degradation/psnr_vals[0]*100:.1f}%)\nSSIM: -{ssim_degradation:.3f} (-{ssim_degradation/ssim_vals[0]*100:.1f}%)', 
                       xy=(1, psnr_vals[1]), xytext=(1.5, psnr_vals[1] + 0.5),
                       ha='left', fontsize=12, fontweight='bold', color='red',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.3),
                       arrowprops=dict(arrowstyle='->', color='red'))
    
    # Add value labels
    for bar, val in zip(bars1, psnr_vals):
        ax_metrics.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    for bar, val in zip(bars2, ssim_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax_metrics.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax_metrics.grid(True, alpha=0.3)
    
    # Middle & Bottom: Visual comparisons - LARGER IMAGES
    sample_images = ['DJI_0061', 'DJI_0082', 'DJI_0271']
    
    for idx, img_name in enumerate(sample_images):
        # RGB Input and SAM mask (middle row)
        ax_input = fig.add_subplot(gs[1, idx+1])
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        sam_mask_path = f"data_for_diffv2ir/sam_masks/{img_name}_mask.png"
        
        # Create side-by-side: RGB + SAM mask
        if os.path.exists(rgb_path) and os.path.exists(sam_mask_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            sam_mask = cv2.imread(sam_mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Resize to match if needed
            if rgb_img.shape[:2] != sam_mask.shape:
                sam_mask = cv2.resize(sam_mask, (rgb_img.shape[1], rgb_img.shape[0]))
            
            # Create overlay visualization
            overlay = rgb_img.copy().astype(float)
            mask_binary = sam_mask > 127
            overlay[mask_binary] = overlay[mask_binary] * 0.7 + np.array([255, 0, 0]) * 0.3
            
            ax_input.imshow(overlay.astype(np.uint8))
            coverage = np.mean(mask_binary) * 100
            ax_input.set_title(f'{img_name}\nSAM Coverage: {coverage:.1f}%', fontsize=12, fontweight='bold')
        ax_input.axis('off')
        
        # Results comparison (bottom row) - LARGER GRAYSCALE
        ax_results = fig.add_subplot(gs[2, idx+1])
        
        white_mask_path = f"data_for_diffv2ir/output/{img_name}.png"
        sam_mask_result_path = f"data_for_diffv2ir/output_sam_20steps/{img_name}.png"
        
        if os.path.exists(white_mask_path) and os.path.exists(sam_mask_result_path):
            white_result = cv2.imread(white_mask_path, cv2.IMREAD_GRAYSCALE)
            sam_result = cv2.imread(sam_mask_result_path, cv2.IMREAD_GRAYSCALE)
            
            # Create side-by-side comparison
            if white_result.shape == sam_result.shape:
                comparison = np.hstack([white_result, sam_result])
                ax_results.imshow(comparison, cmap='gray', vmin=0, vmax=255)
                # Add dividing line
                ax_results.axvline(x=white_result.shape[1], color='red', linewidth=3)
                ax_results.text(white_result.shape[1]//2, 30, 'White Mask (Better)', ha='center', color='white', 
                              fontweight='bold', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.8))
                ax_results.text(white_result.shape[1] + sam_result.shape[1]//2, 30, 'SAM Mask', ha='center', color='white', 
                              fontweight='bold', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8))
        
        ax_results.set_title('White Mask vs SAM Mask Results', fontsize=12, fontweight='bold')
        ax_results.axis('off')
    
    # Key finding box (top right)
    ax_finding = fig.add_subplot(gs[0, 2:])
    ax_finding.text(0.5, 0.5, 'Key Finding: Simple Masks > Complex Masks\n\n'
                              '• SAM provides detailed object segmentation\n'
                              '• However, performance significantly degrades:\n'
                              '  - PSNR drops by 19.4%\n'
                              '  - SSIM drops by 4.2%\n'
                              '• Simple white masks work better for aerial IR\n'
                              '• Lesson: Model design assumptions > mask complexity', 
                   ha='center', va='center', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    ax_finding.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/mask_ablation_improved.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Improved mask ablation saved to: figures/mask_ablation_improved.png")

def main():
    print("Creating improved visualizations with InfraGAN and better sizing...")
    
    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)
    
    print("1. Creating unified 4-model comparison...")
    create_unified_model_comparison_4models()
    
    print("2. Creating improved comprehensive visual comparison...")
    create_comprehensive_visual_improved()
    
    print("3. Creating improved denoising steps ablation...")
    create_denoising_steps_ablation_improved()
    
    print("4. Creating improved mask ablation...")
    create_mask_ablation_improved()
    
    print("\nAll improved visualizations completed!")

if __name__ == '__main__':
    main()