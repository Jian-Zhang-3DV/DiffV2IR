"""
简单测试 DiffV2IR API 
"""

import os
import sys
sys.path.append('/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/DiffV2IR')

def test_import():
    """测试导入"""
    try:
        from diffv2ir_api import diffv2ir_convert, DiffV2IR
        print("✓ 成功导入 diffv2ir_api")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_basic_convert():
    """测试基本转换功能"""
    from diffv2ir_api import diffv2ir_convert
    
    input_path = "data_for_diffv2ir/input/DJI_0061.png"
    output_path = "test_output_simple.png"
    
    if not os.path.exists(input_path):
        print(f"测试图像不存在: {input_path}")
        return False
    
    print(f"测试图像: {input_path}")
    
    try:
        # 创建输出目录
        os.makedirs("test_output", exist_ok=True)
        
        print("开始转换（使用最少步数进行快速测试）...")
        result = diffv2ir_convert(
            input_path=input_path,
            output_path=output_path,
            config_path="configs/generate.yaml",
            checkpoint_path="pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt",
            resolution=256,  # 低分辨率
            steps=5,  # 极少步数，只为测试功能
            use_fp16=True,
            seed=42
        )
        
        print(f"✓ 转换成功！输出: {result}")
        
        # 验证文件
        if os.path.exists(result):
            size = os.path.getsize(result) / 1024
            print(f"✓ 输出文件存在，大小: {size:.2f} KB")
            # 不删除文件，保留用于检查
            # os.remove(result)
            return True
        else:
            print("✗ 输出文件不存在")
            return False
            
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DiffV2IR API 简单测试")
    print("=" * 60)
    
    # 测试导入
    print("\n[1] 测试导入...")
    success1 = test_import()
    
    # 测试基本功能
    print("\n[2] 测试基本转换功能...")
    success2 = test_basic_convert()
    
    # 总结
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✓ 测试通过！API 可正常使用")
    else:
        print("✗ 测试失败，请检查错误信息")