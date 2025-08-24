#!/usr/bin/env python3
"""
对比不同去噪步数的DiffV2IR结果
"""

import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from evaluate_diffv2ir import compute_metrics

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def evaluate_steps_directory(pred_dir, gt_dir, steps_name):
    """评估指定步数目录的结果"""
    print(f"\n评估 {steps_name} 结果...")
    
    results = {}
    psnr_list = []
    ssim_list = []
    
    # 获取所有预测图像文件
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    
    for filename in pred_files:
        pred_path = os.path.join(pred_dir, filename)
        gt_path = os.path.join(gt_dir, filename)
        
        # 检查GT图像是否存在
        if not os.path.exists(gt_path):
            print(f"警告: 未找到对应的GT图像 {filename}")
            continue
        
        try:
            psnr_val, ssim_val, pred_size, gt_original_size, gt_mode = \
                compute_metrics(pred_path, gt_path, 'resize')
            
            results[filename] = {
                'psnr': float(psnr_val),
                'ssim': float(ssim_val)
            }
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
            continue
    
    # 计算统计信息
    summary = {
        'steps': steps_name,
        'total_images': len(psnr_list),
        'average_psnr': float(np.mean(psnr_list)) if psnr_list else 0,
        'std_psnr': float(np.std(psnr_list)) if psnr_list else 0,
        'average_ssim': float(np.mean(ssim_list)) if psnr_list else 0,
        'std_ssim': float(np.std(ssim_list)) if psnr_list else 0,
        'min_psnr': float(np.min(psnr_list)) if psnr_list else 0,
        'max_psnr': float(np.max(psnr_list)) if psnr_list else 0,
        'min_ssim': float(np.min(ssim_list)) if psnr_list else 0,
        'max_ssim': float(np.max(ssim_list)) if psnr_list else 0,
        'per_image_results': results
    }
    
    print(f"{steps_name} 结果:")
    print(f"  评估图像数量: {summary['total_images']}")
    print(f"  平均 PSNR: {summary['average_psnr']:.2f} ± {summary['std_psnr']:.2f} dB")
    print(f"  平均 SSIM: {summary['average_ssim']:.4f} ± {summary['std_ssim']:.4f}")
    
    return summary

def compare_steps_results():
    """对比不同步数的结果"""
    base_path = Path("/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/DiffV2IR")
    gt_dir = base_path / "data_for_diffv2ir/ground_truth"
    
    # 定义要评估的目录
    steps_configs = [
        ("20步", base_path / "data_for_diffv2ir/output"),
        ("50步", base_path / "data_for_diffv2ir/output_50steps"),
    ]
    
    # 评估每个配置
    all_results = []
    for steps_name, pred_dir in steps_configs:
        if pred_dir.exists():
            result = evaluate_steps_directory(str(pred_dir), str(gt_dir), steps_name)
            all_results.append(result)
        else:
            print(f"目录不存在: {pred_dir}")
    
    if len(all_results) < 2:
        print("需要至少两个步数的结果进行对比")
        return
    
    # 创建对比图表
    create_steps_comparison_chart(all_results)
    
    # 保存详细结果
    output_file = base_path / "steps_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n详细对比结果已保存到: {output_file}")
    
    return all_results

def create_steps_comparison_chart(results):
    """创建步数对比图表"""
    
    # 提取数据
    steps_names = [r['steps'] for r in results]
    psnr_values = [r['average_psnr'] for r in results]
    psnr_stds = [r['std_psnr'] for r in results]
    ssim_values = [r['average_ssim'] for r in results]
    ssim_stds = [r['std_ssim'] for r in results]
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PSNR对比
    bars1 = ax1.bar(steps_names, psnr_values, yerr=psnr_stds, 
                    color=['#FF6B6B', '#4ECDC4'], alpha=0.8, capsize=5)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('PSNR vs Denoising Steps', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, value, std in zip(bars1, psnr_values, psnr_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{value:.2f}±{std:.2f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # SSIM对比
    bars2 = ax2.bar(steps_names, ssim_values, yerr=ssim_stds, 
                    color=['#FF6B6B', '#4ECDC4'], alpha=0.8, capsize=5)
    ax2.set_ylabel('SSIM', fontsize=12)
    ax2.set_title('SSIM vs Denoising Steps', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, value, std in zip(bars2, ssim_values, ssim_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{value:.4f}±{std:.4f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    plt.suptitle('DiffV2IR Performance vs Denoising Steps', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    output_path = "/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/周报/figures/diffv2ir_steps_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"步数对比图已保存: {output_path}")
    
    # 创建详细对比表
    create_steps_comparison_table(results)
    
    return output_path

def create_steps_comparison_table(results):
    """创建步数对比详细表格"""
    
    # 准备数据
    data = {
        'Steps': [r['steps'] for r in results],
        'Images': [r['total_images'] for r in results],
        'PSNR (dB)': [f"{r['average_psnr']:.2f} ± {r['std_psnr']:.2f}" for r in results],
        'SSIM': [f"{r['average_ssim']:.4f} ± {r['std_ssim']:.4f}" for r in results],
        'PSNR Range': [f"[{r['min_psnr']:.2f}, {r['max_psnr']:.2f}]" for r in results],
        'SSIM Range': [f"[{r['min_ssim']:.4f}, {r['max_ssim']:.4f}]" for r in results]
    }
    
    # 创建表格图
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table_data = []
    headers = list(data.keys())
    for i in range(len(data['Steps'])):
        row = [data[col][i] for col in headers]
        table_data.append(row)
    
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12, 0.12, 0.2, 0.2, 0.18, 0.18])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    # 设置行颜色
    colors = ['#FFE6E6', '#E6F7F7']
    for i in range(1, len(data['Steps']) + 1):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(colors[i-1])
    
    plt.title('DiffV2IR Denoising Steps Comparison - Detailed Metrics', 
              fontsize=14, fontweight='bold', pad=20)
    
    # 保存图片
    output_path = "/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/周报/figures/diffv2ir_steps_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"步数对比表已保存: {output_path}")
    
    return output_path

def create_sample_visual_comparison(sample_name="DJI_0061"):
    """创建特定样本的不同步数可视化对比"""
    base_path = Path("/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/DiffV2IR")
    
    # 定义图像路径
    paths = {
        'rgb_input': base_path / f"data_for_diffv2ir/input/{sample_name}.png",
        'gt_ir': base_path / f"data_for_diffv2ir/ground_truth/{sample_name}.png",
        '20_steps': base_path / f"data_for_diffv2ir/output/{sample_name}.png",
        '50_steps': base_path / f"data_for_diffv2ir/output_50steps/{sample_name}.png"
    }
    
    # 检查文件是否存在
    missing_files = []
    for name, path in paths.items():
        if not path.exists():
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print(f"缺少以下文件，跳过 {sample_name}:")
        for file in missing_files:
            print(f"  - {file}")
        return None
    
    # 加载图像
    target_size = (512, 448)
    
    def load_and_process_image(image_path, is_rgb=True):
        img = Image.open(image_path)
        if is_rgb and img.mode in ['RGB', 'RGBA']:
            img = img.convert('RGB')
        else:
            img = img.convert('L')
        img = img.resize(target_size, Image.LANCZOS)
        return np.array(img)
    
    rgb_img = load_and_process_image(paths['rgb_input'], is_rgb=True)
    gt_img = load_and_process_image(paths['gt_ir'], is_rgb=False)
    img_20 = load_and_process_image(paths['20_steps'], is_rgb=False)
    img_50 = load_and_process_image(paths['50_steps'], is_rgb=False)
    
    # 创建对比图
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    titles = ['RGB Input', 'Ground Truth', '20 Steps', '50 Steps']
    images = [rgb_img, gt_img, img_20, img_50]
    
    for i, (ax, title, img) in enumerate(zip(axes, titles, images)):
        if i == 0:  # RGB图像
            ax.imshow(img)
        else:  # 红外图像
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # 添加彩色边框
        colors = ['green', 'blue', 'red', 'orange']
        from matplotlib.patches import Rectangle
        rect = Rectangle((0, 0), img.shape[1]-1, img.shape[0]-1, 
                        linewidth=3, edgecolor=colors[i], facecolor='none')
        ax.add_patch(rect)
    
    plt.suptitle(f'Denoising Steps Comparison: {sample_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图像
    output_path = base_path.parent / f"周报/figures/diffv2ir_steps_visual_{sample_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"样本步数对比图已保存: {output_path}")
    return output_path

def main():
    """主函数"""
    print("开始评估DiffV2IR不同去噪步数的性能...")
    
    # 对比步数结果
    results = compare_steps_results()
    
    if results:
        # 生成代表性样本的可视化对比
        samples = ["DJI_0061", "DJI_0068", "DJI_0082"]
        for sample in samples:
            create_sample_visual_comparison(sample)
        
        print("\n" + "="*60)
        print("DiffV2IR去噪步数对比分析完成!")
        print("="*60)
        print("生成的文件:")
        print("1. steps_comparison_results.json - 详细对比数据")
        print("2. diffv2ir_steps_comparison.png - 性能对比图")
        print("3. diffv2ir_steps_table.png - 详细对比表")
        print("4. diffv2ir_steps_visual_*.png - 样本可视化对比")
        print("="*60)

if __name__ == "__main__":
    main()