"""
直接测试推理功能（不使用 API 封装）
"""

import os
import sys
import time

def test_direct_inference():
    """直接使用 infer.py 进行测试"""
    
    print("=" * 60)
    print("DiffV2IR 直接推理测试")
    print("=" * 60)
    
    # 准备测试参数
    input_dir = "data_for_diffv2ir/input"
    output_dir = "test_output_direct"
    checkpoint = "pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建命令
    cmd = f"""python infer.py \
        --input {input_dir} \
        --output {output_dir} \
        --ckpt {checkpoint} \
        --steps 10 \
        --resolution 256 \
        --seed 42
    """
    
    print(f"\n执行命令:\n{cmd}")
    print("\n开始推理测试（使用少量步数）...")
    
    start_time = time.time()
    
    # 执行推理
    exit_code = os.system(cmd)
    
    elapsed_time = time.time() - start_time
    
    if exit_code == 0:
        print(f"\n✓ 推理成功！耗时: {elapsed_time:.2f} 秒")
        
        # 检查输出文件
        output_files = os.listdir(output_dir) if os.path.exists(output_dir) else []
        if output_files:
            print(f"✓ 生成了 {len(output_files)} 个输出文件:")
            for f in output_files[:5]:  # 只显示前5个
                size = os.path.getsize(os.path.join(output_dir, f)) / 1024
                print(f"  - {f} ({size:.2f} KB)")
        
        return True
    else:
        print(f"\n✗ 推理失败，退出码: {exit_code}")
        return False

if __name__ == "__main__":
    success = test_direct_inference()
    
    print("\n" + "=" * 60)
    if success:
        print("测试通过！DiffV2IR 基本功能正常")
        print("\n使用方式:")
        print("1. 直接使用 infer.py 脚本")
        print("2. API 封装需要解决依赖版本冲突问题")
    else:
        print("测试失败，请检查错误信息")