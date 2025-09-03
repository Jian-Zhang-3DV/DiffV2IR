"""
测试 DiffV2IR API 功能
"""

import os
import sys
import time
from diffv2ir_api import diffv2ir_convert, DiffV2IR

def test_simple_api():
    """测试简单函数接口"""
    print("=" * 60)
    print("测试简单函数接口 diffv2ir_convert")
    print("=" * 60)
    
    # 测试单张图像转换
    input_path = "data_for_diffv2ir/input/DJI_0061.png"
    
    if not os.path.exists(input_path):
        print(f"错误：测试图像不存在 {input_path}")
        return False
    
    print(f"\n输入图像: {input_path}")
    print("开始转换...")
    
    start_time = time.time()
    
    try:
        # 使用封装的函数接口
        output_path = diffv2ir_convert(
            input_path=input_path,
            output_path="test_output/DJI_0061_api_test.png",
            config_path="configs/generate.yaml",
            checkpoint_path="pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt",
            resolution=512,
            steps=30,  # 使用较少步数加快测试
            use_fp16=True,  # 使用半精度
            seed=42  # 固定种子以便复现
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"转换成功！")
        print(f"输出路径: {output_path}")
        print(f"耗时: {elapsed_time:.2f} 秒")
        
        # 验证输出文件
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024  # KB
            print(f"输出文件大小: {file_size:.2f} KB")
            return True
        else:
            print("错误：输出文件未创建")
            return False
            
    except Exception as e:
        print(f"转换失败: {str(e)}")
        return False


def test_class_api():
    """测试类接口"""
    print("\n" + "=" * 60)
    print("测试类接口 DiffV2IR")
    print("=" * 60)
    
    try:
        # 初始化模型
        print("\n初始化 DiffV2IR 模型...")
        model = DiffV2IR(
            config_path="configs/generate.yaml",
            checkpoint_path="pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt",
            use_fp16=True,
            load_blip=True
        )
        print("模型初始化成功！")
        
        # 测试单张图像
        print("\n测试单张图像转换...")
        input_path = "data_for_diffv2ir/input/DJI_0068.png"
        
        start_time = time.time()
        output_path = model.convert(
            input_path=input_path,
            output_path="test_output/DJI_0068_class_test.png",
            resolution=512,
            steps=20,  # 快速测试
            seed=42
        )
        elapsed_time = time.time() - start_time
        
        print(f"转换完成: {output_path}")
        print(f"耗时: {elapsed_time:.2f} 秒")
        
        # 测试批量转换
        print("\n测试批量转换（前3张图像）...")
        
        # 创建临时输入文件夹，只包含几张图像
        test_input_dir = "test_input"
        test_output_dir = "test_output/batch"
        
        os.makedirs(test_input_dir, exist_ok=True)
        
        # 复制几张测试图像
        import shutil
        test_images = ["DJI_0061.png", "DJI_0068.png", "DJI_0082.png"]
        for img in test_images:
            src = f"data_for_diffv2ir/input/{img}"
            dst = f"{test_input_dir}/{img}"
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # 执行批量转换
        start_time = time.time()
        results = model.batch_convert(
            input_folder=test_input_dir,
            output_folder=test_output_dir,
            resolution=512,
            steps=20,
            seed=42
        )
        elapsed_time = time.time() - start_time
        
        print(f"\n批量转换完成！")
        print(f"处理 {len(results)} 张图像")
        print(f"总耗时: {elapsed_time:.2f} 秒")
        print(f"平均每张: {elapsed_time/len(results):.2f} 秒")
        
        # 显示结果
        for input_name, output_path in results.items():
            if output_path:
                print(f"  ✓ {input_name} -> {output_path}")
            else:
                print(f"  ✗ {input_name} 转换失败")
        
        # 清理临时文件夹
        shutil.rmtree(test_input_dir)
        
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_auto_output_path():
    """测试自动生成输出路径功能"""
    print("\n" + "=" * 60)
    print("测试自动输出路径生成")
    print("=" * 60)
    
    input_path = "data_for_diffv2ir/input/DJI_0082.png"
    
    try:
        print(f"\n输入: {input_path}")
        print("不指定输出路径，测试自动生成...")
        
        output_path = diffv2ir_convert(
            input_path=input_path,
            # output_path 不指定，会自动生成
            config_path="configs/generate.yaml",
            checkpoint_path="pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt",
            resolution=256,  # 使用较小分辨率加快测试
            steps=10,  # 最少步数，只为测试功能
            use_fp16=True,
            seed=42
        )
        
        print(f"自动生成的输出路径: {output_path}")
        
        # 验证文件名格式
        expected_suffix = "_infrared.png"
        if expected_suffix in output_path:
            print(f"✓ 输出文件名格式正确（包含 {expected_suffix}）")
        else:
            print(f"✗ 输出文件名格式可能不正确")
        
        # 清理自动生成的文件
        if os.path.exists(output_path):
            os.remove(output_path)
            print("清理测试文件完成")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("DiffV2IR API 功能测试")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs("test_output", exist_ok=True)
    os.makedirs("test_output/batch", exist_ok=True)
    
    # 运行测试
    results = []
    
    # 测试1: 简单函数接口
    print("\n[测试 1/3]")
    results.append(("简单函数接口", test_simple_api()))
    
    # 测试2: 类接口
    print("\n[测试 2/3]")
    results.append(("类接口", test_class_api()))
    
    # 测试3: 自动输出路径
    print("\n[测试 3/3]")
    results.append(("自动输出路径", test_auto_output_path()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)