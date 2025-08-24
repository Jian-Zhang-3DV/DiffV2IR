#!/usr/bin/env python3
"""
Evaluate the impact of SAM masks on DiffV2IR performance
Compare original white masks vs SAM-generated masks
"""
import os
import numpy as np
import cv2
from PIL import Image
import json
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_resize_image(image_path, target_size):
    """Load image and resize to target size"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Resize to match target dimensions
    img_resized = cv2.resize(img, target_size)
    return img_resized

def calculate_metrics(pred_path, gt_path):
    """Calculate PSNR and SSIM between prediction and ground truth"""
    # Load ground truth
    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        return None, None
    
    # Load prediction
    pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if pred_img is None:
        return None, None
    
    # Resize prediction to match GT if needed
    if pred_img.shape != gt_img.shape:
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
    
    # Calculate metrics
    psnr_val = psnr(gt_img, pred_img, data_range=255)
    ssim_val = ssim(gt_img, pred_img, data_range=255)
    
    return psnr_val, ssim_val

def evaluate_directory(pred_dir, gt_dir, result_name):
    """Evaluate all images in prediction directory against ground truth"""
    results = {
        'name': result_name,
        'total_images': 0,
        'per_image_results': {},
        'average_psnr': 0,
        'std_psnr': 0,
        'average_ssim': 0,
        'std_ssim': 0,
        'min_psnr': 0,
        'max_psnr': 0,
        'min_ssim': 0,
        'max_ssim': 0
    }
    
    psnr_values = []
    ssim_values = []
    
    # Get list of prediction files
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.png')]
    pred_files.sort()
    
    for pred_file in pred_files:
        pred_path = os.path.join(pred_dir, pred_file)
        
        # Find corresponding ground truth
        # Handle different naming patterns
        gt_candidates = [
            pred_file,  # Same name
            pred_file.replace('.png', '.jpg'),  # Different extension
        ]
        
        gt_path = None
        for gt_candidate in gt_candidates:
            potential_gt_path = os.path.join(gt_dir, gt_candidate)
            if os.path.exists(potential_gt_path):
                gt_path = potential_gt_path
                break
        
        if gt_path is None:
            print(f"No ground truth found for {pred_file}")
            continue
        
        # Calculate metrics
        psnr_val, ssim_val = calculate_metrics(pred_path, gt_path)
        
        if psnr_val is not None and ssim_val is not None:
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            results['per_image_results'][pred_file] = {
                'psnr': psnr_val,
                'ssim': ssim_val
            }
        else:
            print(f"Failed to process {pred_file}")
    
    if len(psnr_values) > 0:
        results['total_images'] = len(psnr_values)
        results['average_psnr'] = np.mean(psnr_values)
        results['std_psnr'] = np.std(psnr_values)
        results['average_ssim'] = np.mean(ssim_values)
        results['std_ssim'] = np.std(ssim_values)
        results['min_psnr'] = np.min(psnr_values)
        results['max_psnr'] = np.max(psnr_values)
        results['min_ssim'] = np.min(ssim_values)
        results['max_ssim'] = np.max(ssim_values)
    
    return results

def create_comparison_visualization(original_results, sam_results, output_path):
    """Create visualization comparing original vs SAM-enhanced results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # PSNR comparison bar chart
    methods = ['Original (White Masks)', 'SAM-Enhanced']
    psnr_means = [original_results['average_psnr'], sam_results['average_psnr']]
    psnr_stds = [original_results['std_psnr'], sam_results['std_psnr']]
    
    colors = ['#ff7f0e', '#2ca02c']
    bars1 = ax1.bar(methods, psnr_means, yerr=psnr_stds, capsize=5, 
                    color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR Comparison: Original vs SAM-Enhanced')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars1, psnr_means, psnr_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{mean:.2f} ± {std:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # SSIM comparison bar chart  
    ssim_means = [original_results['average_ssim'], sam_results['average_ssim']]
    ssim_stds = [original_results['std_ssim'], sam_results['std_ssim']]
    
    bars2 = ax2.bar(methods, ssim_means, yerr=ssim_stds, capsize=5,
                    color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM Comparison: Original vs SAM-Enhanced')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars2, ssim_means, ssim_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.4f} ± {std:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Per-image PSNR scatter plot
    common_images = set(original_results['per_image_results'].keys()) & \
                   set(sam_results['per_image_results'].keys())
    
    orig_psnr_vals = [original_results['per_image_results'][img]['psnr'] for img in common_images]
    sam_psnr_vals = [sam_results['per_image_results'][img]['psnr'] for img in common_images]
    
    ax3.scatter(orig_psnr_vals, sam_psnr_vals, alpha=0.7, s=60, color='#1f77b4')
    min_val = min(orig_psnr_vals + sam_psnr_vals)
    max_val = max(orig_psnr_vals + sam_psnr_vals)
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    ax3.set_xlabel('Original PSNR (dB)')
    ax3.set_ylabel('SAM-Enhanced PSNR (dB)')
    ax3.set_title('Per-Image PSNR Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Per-image SSIM scatter plot
    orig_ssim_vals = [original_results['per_image_results'][img]['ssim'] for img in common_images]
    sam_ssim_vals = [sam_results['per_image_results'][img]['ssim'] for img in common_images]
    
    ax4.scatter(orig_ssim_vals, sam_ssim_vals, alpha=0.7, s=60, color='#ff7f0e')
    min_val = min(orig_ssim_vals + sam_ssim_vals)
    max_val = max(orig_ssim_vals + sam_ssim_vals)
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    ax4.set_xlabel('Original SSIM')
    ax4.set_ylabel('SAM-Enhanced SSIM')
    ax4.set_title('Per-Image SSIM Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison visualization saved to: {output_path}")

def create_detailed_comparison_table(original_results, sam_results, output_path):
    """Create detailed comparison table"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate improvements
    psnr_improvement = sam_results['average_psnr'] - original_results['average_psnr']
    ssim_improvement = sam_results['average_ssim'] - original_results['average_ssim']
    psnr_improvement_pct = (psnr_improvement / original_results['average_psnr']) * 100
    ssim_improvement_pct = (ssim_improvement / original_results['average_ssim']) * 100
    
    table_data = [
        ['Method', 'Images', 'PSNR (dB)', 'SSIM', 'PSNR Range', 'SSIM Range'],
        ['Original (White Masks)', 
         str(original_results['total_images']),
         f"{original_results['average_psnr']:.2f} ± {original_results['std_psnr']:.2f}",
         f"{original_results['average_ssim']:.4f} ± {original_results['std_ssim']:.4f}",
         f"[{original_results['min_psnr']:.2f}, {original_results['max_psnr']:.2f}]",
         f"[{original_results['min_ssim']:.4f}, {original_results['max_ssim']:.4f}]"],
        ['SAM-Enhanced',
         str(sam_results['total_images']),
         f"{sam_results['average_psnr']:.2f} ± {sam_results['std_psnr']:.2f}",
         f"{sam_results['average_ssim']:.4f} ± {sam_results['std_ssim']:.4f}",
         f"[{sam_results['min_psnr']:.2f}, {sam_results['max_psnr']:.2f}]",
         f"[{sam_results['min_ssim']:.4f}, {sam_results['max_ssim']:.4f}]"],
        ['Improvement',
         '',
         f"{psnr_improvement:+.2f} ({psnr_improvement_pct:+.1f}%)",
         f"{ssim_improvement:+.4f} ({ssim_improvement_pct:+.1f}%)",
         '', '']
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.1, 0.2, 0.2, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style improvement row
    for i in range(len(table_data[0])):
        if psnr_improvement > 0 and ssim_improvement > 0:
            table[(3, i)].set_facecolor('#E8F5E8')  # Light green for improvement
        else:
            table[(3, i)].set_facecolor('#FFE8E8')  # Light red for degradation
    
    plt.title('DiffV2IR Performance: SAM Masks vs Original White Masks\nDetailed Comparison Table', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Detailed comparison table saved to: {output_path}")

def main():
    # Paths
    original_output_dir = "data_for_diffv2ir/output"  # 20 steps with white masks
    sam_output_dir = "data_for_diffv2ir/output_sam_20steps"  # 20 steps with SAM masks
    gt_dir = "data_for_diffv2ir/ground_truth"  # Ground truth
    
    print("Evaluating DiffV2IR performance with and without SAM masks...")
    
    # Evaluate both versions
    print("Evaluating original version (white masks)...")
    original_results = evaluate_directory(original_output_dir, gt_dir, "Original_20steps")
    
    print("Evaluating SAM-enhanced version...")
    sam_results = evaluate_directory(sam_output_dir, gt_dir, "SAM_Enhanced_20steps")
    
    # Save results
    all_results = {
        'original': original_results,
        'sam_enhanced': sam_results,
        'evaluation_summary': {
            'psnr_improvement': sam_results['average_psnr'] - original_results['average_psnr'],
            'ssim_improvement': sam_results['average_ssim'] - original_results['average_ssim'],
            'psnr_improvement_percent': ((sam_results['average_psnr'] - original_results['average_psnr']) / original_results['average_psnr']) * 100,
            'ssim_improvement_percent': ((sam_results['average_ssim'] - original_results['average_ssim']) / original_results['average_ssim']) * 100
        }
    }
    
    with open('sam_enhancement_evaluation.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create visualizations
    create_comparison_visualization(original_results, sam_results, 
                                   'figures/sam_enhancement_comparison.png')
    create_detailed_comparison_table(original_results, sam_results,
                                    'figures/sam_enhancement_table.png')
    
    # Print summary
    print("\n" + "="*60)
    print("SAM ENHANCEMENT EVALUATION RESULTS")
    print("="*60)
    print(f"Original (White Masks):")
    print(f"  PSNR: {original_results['average_psnr']:.2f} ± {original_results['std_psnr']:.2f} dB")
    print(f"  SSIM: {original_results['average_ssim']:.4f} ± {original_results['std_ssim']:.4f}")
    print(f"  Images: {original_results['total_images']}")
    
    print(f"\nSAM-Enhanced:")
    print(f"  PSNR: {sam_results['average_psnr']:.2f} ± {sam_results['std_psnr']:.2f} dB")
    print(f"  SSIM: {sam_results['average_ssim']:.4f} ± {sam_results['std_ssim']:.4f}")
    print(f"  Images: {sam_results['total_images']}")
    
    psnr_imp = all_results['evaluation_summary']['psnr_improvement']
    ssim_imp = all_results['evaluation_summary']['ssim_improvement']
    psnr_imp_pct = all_results['evaluation_summary']['psnr_improvement_percent']
    ssim_imp_pct = all_results['evaluation_summary']['ssim_improvement_percent']
    
    print(f"\nIMPROVEMENT ANALYSIS:")
    print(f"  PSNR: {psnr_imp:+.2f} dB ({psnr_imp_pct:+.1f}%)")
    print(f"  SSIM: {ssim_imp:+.4f} ({ssim_imp_pct:+.1f}%)")
    
    if psnr_imp > 0 and ssim_imp > 0:
        print(f"\n✅ SAM masks provide SIGNIFICANT IMPROVEMENT over white masks!")
    elif psnr_imp > 0 or ssim_imp > 0:
        print(f"\n⚠️  SAM masks provide MIXED results compared to white masks.")
    else:
        print(f"\n❌ SAM masks do NOT improve performance over white masks.")
    
    print("\nFiles saved:")
    print("  - sam_enhancement_evaluation.json")
    print("  - figures/sam_enhancement_comparison.png")
    print("  - figures/sam_enhancement_table.png")
    print("="*60)

if __name__ == '__main__':
    main()