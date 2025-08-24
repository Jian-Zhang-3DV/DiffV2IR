#!/usr/bin/env python3
"""
评估DiffV2IR生成的红外图像质量
基于PID的评估脚本适配而来
处理：
1. 分辨率不匹配：GT(640×512) vs 预测(512×448)
2. 通道不匹配：GT(灰度) vs 预测(RGB)
"""

import os
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse
from tqdm import tqdm
import json

def prepare_images_for_comparison(pred_path, gt_path, resize_method='resize'):
    """
    准备图像用于比较，处理分辨率和通道差异
    resize_method: 'resize' 或 'crop'
    """
    # 加载预测图像（RGB，512×448）
    pred_img = Image.open(pred_path)
    if pred_img.mode != 'RGB':
        pred_img = pred_img.convert('RGB')
    pred_array = np.array(pred_img)
    
    # 加载GT图像
    gt_img = Image.open(gt_path)
    
    # 处理分辨率差异
    if resize_method == 'resize':
        # 方法1：调整GT到预测图像尺寸
        if gt_img.size != pred_img.size:
            gt_img = gt_img.resize(pred_img.size, Image.LANCZOS)
    elif resize_method == 'crop':
        # 方法2：从GT中心裁剪到预测图像尺寸
        pred_w, pred_h = pred_img.size
        gt_w, gt_h = gt_img.size
        if gt_img.size != pred_img.size:
            # 计算中心裁剪区域
            left = max(0, (gt_w - pred_w) // 2)
            top = max(0, (gt_h - pred_h) // 2)
            right = min(gt_w, left + pred_w)
            bottom = min(gt_h, top + pred_h)
            gt_img = gt_img.crop((left, top, right, bottom))
            # 如果裁剪后尺寸还不匹配，进行resize
            if gt_img.size != pred_img.size:
                gt_img = gt_img.resize(pred_img.size, Image.LANCZOS)
    
    # 处理通道差异
    if gt_img.mode == 'L':
        # GT是灰度图，将预测图像也转为灰度进行比较
        pred_gray = pred_img.convert('L')
        pred_array = np.array(pred_gray)
        gt_array = np.array(gt_img)
    elif gt_img.mode == 'RGB':
        # 都是RGB
        gt_array = np.array(gt_img)
        pred_array = np.array(pred_img)
    else:
        # 其他情况，都转为RGB
        gt_img = gt_img.convert('RGB')
        gt_array = np.array(gt_img)
        if pred_img.mode != 'RGB':
            pred_img = pred_img.convert('RGB')
            pred_array = np.array(pred_img)
    
    return pred_array, gt_array, pred_img.size, Image.open(gt_path).size, gt_img.mode

def compute_metrics(pred_path, gt_path, resize_method='resize'):
    """计算单张图像的PSNR和SSIM"""
    pred_array, gt_array, pred_size, gt_original_size, gt_mode = \
        prepare_images_for_comparison(pred_path, gt_path, resize_method)
    
    # 确保数据类型正确
    pred_array = pred_array.astype(np.uint8)
    gt_array = gt_array.astype(np.uint8)
    
    # 计算PSNR
    psnr_value = psnr(gt_array, pred_array, data_range=255)
    
    # 计算SSIM
    if len(gt_array.shape) == 2:
        # 灰度图像
        ssim_value = ssim(gt_array, pred_array, data_range=255)
    else:
        # 彩色图像
        ssim_value = ssim(gt_array, pred_array, multichannel=True, channel_axis=2, data_range=255)
    
    return psnr_value, ssim_value, pred_size, gt_original_size, gt_mode

def evaluate_dataset(pred_dir, gt_dir, output_file=None, resize_method='resize'):
    """评估整个数据集"""
    results = {}
    psnr_list = []
    ssim_list = []
    size_info = {}
    mode_info = {}
    
    # 获取所有预测图像文件
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    
    print(f"找到 {len(pred_files)} 张预测图像")
    print(f"处理方法: {resize_method}")
    print("注意：")
    print("  - GT灰度图像将与预测图像的灰度版本进行比较")
    if resize_method == 'resize':
        print("  - GT图像将被调整到预测图像的分辨率")
    else:
        print("  - GT图像将被中心裁剪到预测图像的分辨率")
    
    # 处理每张图像
    for filename in tqdm(pred_files, desc="评估中"):
        pred_path = os.path.join(pred_dir, filename)
        gt_path = os.path.join(gt_dir, filename)
        
        # 检查GT图像是否存在
        if not os.path.exists(gt_path):
            print(f"警告: 未找到对应的GT图像 {filename}")
            continue
        
        # 计算指标
        try:
            psnr_val, ssim_val, pred_size, gt_original_size, gt_mode = \
                compute_metrics(pred_path, gt_path, resize_method)
            
            # 保存结果
            results[filename] = {
                'psnr': float(psnr_val),
                'ssim': float(ssim_val),
                'pred_size': pred_size,
                'gt_original_size': gt_original_size,
                'gt_mode': gt_mode
            }
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            
            # 记录尺寸信息
            size_key = f"{pred_size} <- {gt_original_size}"
            if size_key not in size_info:
                size_info[size_key] = 0
            size_info[size_key] += 1
            
            # 记录模式信息
            if gt_mode not in mode_info:
                mode_info[gt_mode] = 0
            mode_info[gt_mode] += 1
            
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
            continue
    
    # 计算平均值
    avg_psnr = np.mean(psnr_list) if psnr_list else 0
    avg_ssim = np.mean(ssim_list) if ssim_list else 0
    std_psnr = np.std(psnr_list) if psnr_list else 0
    std_ssim = np.std(ssim_list) if ssim_list else 0
    
    # 汇总结果
    summary = {
        'total_images': len(psnr_list),
        'average_psnr': float(avg_psnr),
        'std_psnr': float(std_psnr),
        'average_ssim': float(avg_ssim),
        'std_ssim': float(std_ssim),
        'min_psnr': float(np.min(psnr_list)) if psnr_list else 0,
        'max_psnr': float(np.max(psnr_list)) if psnr_list else 0,
        'min_ssim': float(np.min(ssim_list)) if ssim_list else 0,
        'max_ssim': float(np.max(ssim_list)) if ssim_list else 0,
        'resize_method': resize_method,
        'size_conversions': size_info,
        'gt_modes': mode_info,
        'per_image_results': results
    }
    
    # 打印结果
    print("\n" + "="*60)
    print(f"DiffV2IR评估结果汇总（{resize_method}方法）")
    print("="*60)
    print(f"评估图像数量: {summary['total_images']}")
    print(f"平均 PSNR: {summary['average_psnr']:.2f} ± {summary['std_psnr']:.2f} dB")
    print(f"PSNR 范围: [{summary['min_psnr']:.2f}, {summary['max_psnr']:.2f}]")
    print(f"平均 SSIM: {summary['average_ssim']:.4f} ± {summary['std_ssim']:.4f}")
    print(f"SSIM 范围: [{summary['min_ssim']:.4f}, {summary['max_ssim']:.4f}]")
    print("\n图像信息:")
    print("  分辨率转换:")
    for size_conv, count in size_info.items():
        print(f"    {size_conv}: {count} 张")
    print("  GT图像模式:")
    for mode, count in mode_info.items():
        print(f"    {mode}: {count} 张")
    print("="*60)
    
    # 保存结果到文件
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:  # 只有当目录不为空时才创建
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n详细结果已保存到: {output_file}")
    
    return summary

def compare_methods(pred_dir, gt_dir):
    """比较不同处理方法的结果"""
    # 使用resize方法
    print("\n>>> 方法1: Resize (调整GT尺寸)")
    results_resize = evaluate_dataset(
        pred_dir, 
        gt_dir,
        'diffv2ir_evaluation_resize.json',
        'resize'
    )
    
    # 使用crop方法
    print("\n>>> 方法2: Crop (中心裁剪GT)")
    results_crop = evaluate_dataset(
        pred_dir,
        gt_dir, 
        'diffv2ir_evaluation_crop.json',
        'crop'
    )
    
    # 对比结果
    print("\n" + "="*60)
    print("DiffV2IR方法对比")
    print("="*60)
    print(f"{'方法':<10} {'PSNR':<15} {'SSIM':<15}")
    print("-"*40)
    print(f"{'Resize':<10} {results_resize['average_psnr']:<15.2f} {results_resize['average_ssim']:<15.4f}")
    print(f"{'Crop':<10} {results_crop['average_psnr']:<15.2f} {results_crop['average_ssim']:<15.4f}")
    print("="*60)
    
    return results_resize, results_crop

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估DiffV2IR红外图像生成质量')
    parser.add_argument('--pred_dir', type=str, 
                       default='data_for_diffv2ir/output',
                       help='预测图像目录')
    parser.add_argument('--gt_dir', type=str,
                       default='data_for_diffv2ir/ground_truth',
                       help='真实图像目录')
    parser.add_argument('--output', type=str,
                       default='diffv2ir_evaluation.json',
                       help='输出结果文件')
    parser.add_argument('--method', type=str, default='resize',
                       choices=['resize', 'crop', 'both'],
                       help='处理方法: resize(调整尺寸), crop(中心裁剪), both(比较两种)')
    
    args = parser.parse_args()
    
    if args.method == 'both':
        compare_methods(args.pred_dir, args.gt_dir)
    else:
        evaluate_dataset(args.pred_dir, args.gt_dir, args.output, args.method)