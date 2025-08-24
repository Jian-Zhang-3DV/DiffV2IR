#!/usr/bin/env python3
"""
Evaluate InfraGAN performance on xmu aerial dataset
"""
import os
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pandas as pd

def evaluate_infragan():
    """Evaluate InfraGAN results"""
    
    # Paths
    infragan_results_dir = "/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/InfraGAN/results/infragan_vedai/inference_results"
    gt_dir = "data_for_diffv2ir/ground_truth"
    
    results = []
    
    # Get all InfraGAN DJI results
    infragan_files = [f for f in os.listdir(infragan_results_dir) if f.startswith('DJI_') and f.endswith('_infrared.png')]
    
    print(f"Found {len(infragan_files)} InfraGAN results")
    
    for infragan_file in infragan_files:
        # Extract image name
        img_name = infragan_file.replace('_infrared.png', '.png')
        base_name = img_name.replace('.png', '')
        
        # Load InfraGAN result
        infragan_path = os.path.join(infragan_results_dir, infragan_file)
        infragan_img = cv2.imread(infragan_path, cv2.IMREAD_GRAYSCALE)
        
        if infragan_img is None:
            continue
            
        # Load ground truth
        gt_path = os.path.join(gt_dir, img_name)
        if not os.path.exists(gt_path):
            print(f"Warning: GT not found for {img_name}")
            continue
            
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize if needed
        if infragan_img.shape != gt_img.shape:
            infragan_img_resized = cv2.resize(infragan_img, (gt_img.shape[1], gt_img.shape[0]))
        else:
            infragan_img_resized = infragan_img
            
        # Calculate metrics
        try:
            psnr_val = psnr(gt_img, infragan_img_resized, data_range=255)
            ssim_val = ssim(gt_img, infragan_img_resized, data_range=255)
            
            results.append({
                'Image': base_name,
                'PSNR': psnr_val,
                'SSIM': ssim_val
            })
            
            print(f"{base_name}: PSNR={psnr_val:.2f}dB, SSIM={ssim_val:.4f}")
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            continue
    
    if results:
        # Calculate statistics
        df = pd.DataFrame(results)
        mean_psnr = df['PSNR'].mean()
        std_psnr = df['PSNR'].std()
        mean_ssim = df['SSIM'].mean()
        std_ssim = df['SSIM'].std()
        
        print(f"\nInfraGAN Performance Summary:")
        print(f"Images evaluated: {len(results)}")
        print(f"PSNR: {mean_psnr:.2f} ± {std_psnr:.2f} dB")
        print(f"SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")
        print(f"PSNR range: [{df['PSNR'].min():.2f}, {df['PSNR'].max():.2f}]")
        print(f"SSIM range: [{df['SSIM'].min():.4f}, {df['SSIM'].max():.4f}]")
        
        # Save detailed results
        df.to_csv('infragan_evaluation_results.csv', index=False)
        print(f"\nDetailed results saved to: infragan_evaluation_results.csv")
        
        return {
            'mean_psnr': mean_psnr,
            'std_psnr': std_psnr,
            'mean_ssim': mean_ssim,
            'std_ssim': std_ssim,
            'count': len(results),
            'results': results
        }
    else:
        print("No valid results found!")
        return None

if __name__ == '__main__':
    evaluate_infragan()