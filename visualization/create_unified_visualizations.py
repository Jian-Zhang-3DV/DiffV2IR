#!/usr/bin/env python3
"""
Create unified and organized visualizations for DiffV2IR evaluation report
1. Model comparison with metrics and visual examples
2. DiffV2IR comprehensive visual comparisons 
3. Ablation studies (denoising steps, SAM masks)
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

def create_unified_model_comparison():
    """Create unified visualization showing metrics and visual comparisons of all models"""
    # Model performance data
    models = ['sRGB-TIR-02', 'PID', 'DiffV2IR']
    psnr_values = [11.04, 10.75, 9.56]
    ssim_values = [0.4056, 0.4272, 0.3886]
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])
    
    # Left side: Metrics comparison
    ax_metrics = fig.add_subplot(gs[:, 0])
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax_metrics.bar(x - width/2, psnr_values, width, label='PSNR (dB)', 
                          color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
    ax2 = ax_metrics.twinx()
    bars2 = ax2.bar(x + width/2, ssim_values, width, label='SSIM', 
                   color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.6)
    
    ax_metrics.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax_metrics.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
    ax_metrics.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(models, rotation=0)
    
    # Add value labels on bars
    for bar, val in zip(bars1, psnr_values):
        ax_metrics.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    for bar, val in zip(bars2, ssim_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax_metrics.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax_metrics.grid(True, alpha=0.3)
    
    # Right side: Visual comparisons for 3 representative samples
    sample_images = ['DJI_0061', 'DJI_0082', 'DJI_0271']
    
    for idx, img_name in enumerate(sample_images):
        # RGB Input
        ax_rgb = fig.add_subplot(gs[0, idx+1])
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        if os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            ax_rgb.imshow(rgb_img)
        ax_rgb.set_title(f'RGB Input\n{img_name}', fontsize=10, fontweight='bold')
        ax_rgb.axis('off')
        
        # Model comparisons (GT + 3 models)
        ax_models = fig.add_subplot(gs[1, idx+1])
        
        # Load images
        gt_path = f"data_for_diffv2ir/ground_truth/{img_name}.png"
        diffv2ir_path = f"data_for_diffv2ir/output/{img_name}.png"
        
        # Create 2x2 subgrid for this sample
        sub_fig, sub_axes = plt.subplots(2, 2, figsize=(4, 4))
        
        # GT IR
        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            sub_axes[0, 0].imshow(gt_img, cmap='gray')
        sub_axes[0, 0].set_title('GT IR', fontsize=8)
        sub_axes[0, 0].axis('off')
        
        # DiffV2IR
        if os.path.exists(diffv2ir_path):
            diffv2ir_img = cv2.imread(diffv2ir_path, cv2.IMREAD_GRAYSCALE)
            sub_axes[0, 1].imshow(diffv2ir_img, cmap='gray')
        sub_axes[0, 1].set_title('DiffV2IR', fontsize=8)
        sub_axes[0, 1].axis('off')
        
        # PID (if available)
        sub_axes[1, 0].text(0.5, 0.5, 'PID\n(Previous)', ha='center', va='center', fontsize=8)
        sub_axes[1, 0].axis('off')
        
        # sRGB-TIR (if available)
        sub_axes[1, 1].text(0.5, 0.5, 'sRGB-TIR\n(Previous)', ha='center', va='center', fontsize=8)
        sub_axes[1, 1].axis('off')
        
        # Save sub-figure and embed
        sub_fig.tight_layout()
        sub_fig.savefig(f'temp_{img_name}_models.png', dpi=100, bbox_inches='tight')
        plt.close(sub_fig)
        
        # Load and display the saved image
        temp_img = cv2.imread(f'temp_{img_name}_models.png')
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        ax_models.imshow(temp_img)
        ax_models.set_title(f'Model Outputs', fontsize=10)
        ax_models.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/unified_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Clean up temp files
    for img_name in sample_images:
        temp_file = f'temp_{img_name}_models.png'
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("Unified model comparison saved to: figures/unified_model_comparison.png")

def create_diffv2ir_comprehensive_visual():
    """Create comprehensive DiffV2IR visual comparisons (RGB + GT IR + Predicted IR)"""
    sample_images = ['DJI_0061', 'DJI_0068', 'DJI_0082', 'DJI_0128', 'DJI_0271', 'DJI_0281']
    
    fig, axes = plt.subplots(3, len(sample_images), figsize=(24, 12))
    
    for i, img_name in enumerate(sample_images):
        # Row 1: RGB Input
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        if os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f'RGB Input\n{img_name}', fontsize=11, fontweight='bold')
        axes[0, i].axis('off')
        
        # Row 2: Ground Truth IR
        gt_path = f"data_for_diffv2ir/ground_truth/{img_name}.png"
        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            im2 = axes[1, i].imshow(gt_img, cmap='hot')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046)
        axes[1, i].set_title('Ground Truth IR', fontsize=11, fontweight='bold')
        axes[1, i].axis('off')
        
        # Row 3: DiffV2IR Predicted IR
        pred_path = f"data_for_diffv2ir/output/{img_name}.png"
        if os.path.exists(pred_path):
            pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            im3 = axes[2, i].imshow(pred_img, cmap='hot')
            plt.colorbar(im3, ax=axes[2, i], fraction=0.046)
            
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
                                   fontsize=10, fontweight='bold')
            else:
                axes[2, i].set_title('DiffV2IR Predicted IR', fontsize=11, fontweight='bold')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/diffv2ir_comprehensive_visual.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("DiffV2IR comprehensive visual comparison saved to: figures/diffv2ir_comprehensive_visual.png")

def create_denoising_steps_ablation():
    """Create denoising steps ablation study visualization"""
    # Performance data
    steps_data = {
        '20 Steps': {'PSNR': 9.56, 'SSIM': 0.3886, 'Speed': '1x'},
        '50 Steps': {'PSNR': 8.86, 'SSIM': 0.3677, 'Speed': '2.5x'}
    }
    
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 4, height_ratios=[1, 1])
    
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
    
    ax_metrics.set_xlabel('Denoising Steps', fontsize=12, fontweight='bold')
    ax_metrics.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
    ax_metrics.set_title('Denoising Steps Ablation Study', fontsize=14, fontweight='bold')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(steps)
    
    # Add performance change annotations
    psnr_change = psnr_vals[0] - psnr_vals[1]
    ssim_change = ssim_vals[0] - ssim_vals[1]
    ax_metrics.annotate(f'PSNR: +{psnr_change:.2f}dB (+{psnr_change/psnr_vals[1]*100:.1f}%)', 
                       xy=(0.5, max(psnr_vals) + 0.5), xytext=(0.5, max(psnr_vals) + 1),
                       ha='center', fontsize=10, fontweight='bold', color='green',
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
    
    # Right: Visual comparisons for representative samples
    sample_images = ['DJI_0061', 'DJI_0068', 'DJI_0082']
    
    for idx, img_name in enumerate(sample_images):
        # RGB Input (top row)
        ax_rgb = fig.add_subplot(gs[0, idx+1])
        rgb_path = f"data_for_diffv2ir/input/{img_name}.png"
        if os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            ax_rgb.imshow(rgb_img)
        ax_rgb.set_title(f'RGB Input - {img_name}', fontsize=10, fontweight='bold')
        ax_rgb.axis('off')
        
        # Comparison (bottom row)
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
                    ax_comp.imshow(comparison, cmap='gray')
                    # Add dividing line
                    ax_comp.axvline(x=img_20.shape[1], color='red', linewidth=2)
                    ax_comp.text(img_20.shape[1]//2, 10, '20 Steps', ha='center', color='white', 
                               fontweight='bold', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7))
                    ax_comp.text(img_20.shape[1] + img_50.shape[1]//2, 10, '50 Steps', ha='center', color='white', 
                               fontweight='bold', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
                else:
                    ax_comp.imshow(img_20, cmap='gray')
                    ax_comp.text(0.5, 0.95, '20 Steps Only', transform=ax_comp.transAxes, ha='center', 
                               color='white', fontweight='bold')
            else:
                ax_comp.imshow(img_20, cmap='gray')
                ax_comp.text(0.5, 0.95, '20 Steps', transform=ax_comp.transAxes, ha='center', 
                           color='white', fontweight='bold')
        
        ax_comp.set_title('20 Steps vs 50 Steps', fontsize=10, fontweight='bold')
        ax_comp.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/denoising_steps_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Denoising steps ablation study saved to: figures/denoising_steps_ablation.png")

def create_mask_ablation_study():
    """Create SAM mask ablation study visualization"""
    # Performance data
    mask_data = {
        'White Mask (Original)': {'PSNR': 9.60, 'SSIM': 0.4140},
        'SAM Mask': {'PSNR': 7.73, 'SSIM': 0.3968}
    }
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, height_ratios=[0.8, 1, 1])
    
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
    
    ax_metrics.set_xlabel('Mask Type', fontsize=12, fontweight='bold')
    ax_metrics.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
    ax_metrics.set_title('Mask Ablation Study: Performance Impact', fontsize=14, fontweight='bold')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(['White Mask', 'SAM Mask'])
    
    # Add performance degradation annotation
    psnr_degradation = psnr_vals[0] - psnr_vals[1]
    ssim_degradation = ssim_vals[0] - ssim_vals[1]
    ax_metrics.annotate(f'Performance Drop:\nPSNR: -{psnr_degradation:.2f}dB (-{psnr_degradation/psnr_vals[0]*100:.1f}%)\nSSIM: -{ssim_degradation:.3f} (-{ssim_degradation/ssim_vals[0]*100:.1f}%)', 
                       xy=(1, psnr_vals[1]), xytext=(1.5, psnr_vals[1] + 0.5),
                       ha='left', fontsize=10, fontweight='bold', color='red',
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
    
    # Middle & Bottom: Visual comparisons
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
            ax_input.set_title(f'{img_name}\nSAM Coverage: {coverage:.1f}%', fontsize=10, fontweight='bold')
        ax_input.axis('off')
        
        # Results comparison (bottom row)
        ax_results = fig.add_subplot(gs[2, idx+1])
        
        white_mask_path = f"data_for_diffv2ir/output/{img_name}.png"
        sam_mask_result_path = f"data_for_diffv2ir/output_sam_20steps/{img_name}.png"
        
        if os.path.exists(white_mask_path) and os.path.exists(sam_mask_result_path):
            white_result = cv2.imread(white_mask_path, cv2.IMREAD_GRAYSCALE)
            sam_result = cv2.imread(sam_mask_result_path, cv2.IMREAD_GRAYSCALE)
            
            # Create side-by-side comparison
            if white_result.shape == sam_result.shape:
                comparison = np.hstack([white_result, sam_result])
                ax_results.imshow(comparison, cmap='gray')
                # Add dividing line
                ax_results.axvline(x=white_result.shape[1], color='red', linewidth=2)
                ax_results.text(white_result.shape[1]//2, 20, 'White Mask', ha='center', color='white', 
                              fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7))
                ax_results.text(white_result.shape[1] + sam_result.shape[1]//2, 20, 'SAM Mask', ha='center', color='white', 
                              fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
        
        ax_results.set_title('White Mask vs SAM Mask Results', fontsize=10, fontweight='bold')
        ax_results.axis('off')
    
    # Key finding box (top right)
    ax_finding = fig.add_subplot(gs[0, 2:])
    ax_finding.text(0.5, 0.5, '🔍 Key Finding: More Detailed Masks ≠ Better Results\n\n'
                              '• SAM masks provide more detailed object segmentation\n'
                              '• However, performance significantly degrades:\n'
                              '  - PSNR drops by 19.4%\n'
                              '  - SSIM drops by 4.2%\n'
                              '• Simple white masks work better for aerial IR generation\n'
                              '• Lesson: Model design assumptions matter more than mask detail', 
                   ha='center', va='center', fontsize=11, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    ax_finding.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/mask_ablation_study.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Mask ablation study saved to: figures/mask_ablation_study.png")

def main():
    print("Creating unified and organized visualizations...")
    
    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)
    
    print("1. Creating unified model comparison...")
    create_unified_model_comparison()
    
    print("2. Creating comprehensive DiffV2IR visual comparisons...")
    create_diffv2ir_comprehensive_visual()
    
    print("3. Creating denoising steps ablation study...")
    create_denoising_steps_ablation()
    
    print("4. Creating mask ablation study...")
    create_mask_ablation_study()
    
    print("\nAll unified visualizations completed!")

if __name__ == '__main__':
    main()