# DiffV2IR 前端集成指南

## 项目简介

DiffV2IR 是一个基于扩散模型的可见光到红外图像转换系统，通过视觉-语言理解实现高质量的图像转换。该系统包含两个核心模块：
- **渐进学习模块 (PLM)**：使用多阶段知识学习实现从全波段到目标波长的红外转换
- **视觉-语言理解模块 (VLUM)**：结合语言描述和分割图增强语义感知能力

## 环境配置

### 1. 系统要求
- Python 3.10.15
- CUDA 12.8 支持的 GPU（建议显存 >= 8GB）
- Linux/Unix 系统（推荐 Ubuntu 20.04+）
- Git
- 至少 20GB 可用磁盘空间（用于模型和数据）

### 2. 创建虚拟环境
```bash
# 使用 conda 创建环境（推荐）
conda create -n DiffV2IR python=3.10.15
conda activate DiffV2IR
```

### 3. 克隆项目代码
```bash
git clone https://github.com/Jian-Zhang-3DV/DiffV2IR.git
cd DiffV2IR
```

### 4. 安装依赖

```bash
# 安装 PyTorch 2.8.0 with CUDA 12.8
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# 安装所有其他依赖（requirements.txt 已更新为正确版本）
pip install -r requirements.txt
```

### 5. 验证环境配置

```bash
# 验证 Python 版本
python --version  # 应显示 Python 3.10.15

# 验证 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 验证关键包
python -c "import clip, diffusers, transformers, einops, omegaconf; print('All packages imported successfully')"
```

主要依赖包括：
- torch==2.8.0（使用 CUDA 12.8）
- torchvision==0.23.0
- numpy==1.26.0
- transformers==4.26.1
- diffusers==0.35.1
- pytorch-lightning==1.4.2
- opencv-python==4.10.0.84
- CLIP（OpenAI 视觉-语言模型）
- k-diffusion（采样算法）
- taming-transformers（向量量化模块）

### 5. 下载预训练模型

#### 模型管理系统
DiffV2IR 现在支持自定义模型路径，避免下载到系统缓存目录。所有模型都可以统一管理。

#### 自动下载所需模型
使用提供的下载脚本自动下载 BLIP 和 CLIP 模型：
```bash
# 下载所有必需的模型到 models/ 目录
python download_models.py

# 或指定自定义目录
python download_models.py --model-dir /path/to/your/models

# 只下载特定模型
python download_models.py --models blip clip

# 跳过已存在的模型
python download_models.py --skip-existing
```

#### DiffV2IR 主模型权重
从以下链接下载 DiffV2IR 预训练模型：

**选项 1：夸克网盘**
- 链接：https://pan.quark.cn/s/e2f28304ee90
- 访问码：EWCz

**选项 2：HuggingFace**
- 链接：https://huggingface.co/datasets/Lidong26/IR-500K/tree/main

下载后配置：
```bash
# 创建目录结构
mkdir -p pretrained/DiffV2IR/IR-500k/finetuned_checkpoints

# 移动下载的文件到正确位置
mv ~/Downloads/after_phase_2.ckpt pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/

# 验证文件（应该约 5.7GB）
ls -lh pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt
```

#### 模型文件说明
| 模型 | 用途 | 大小 | 默认路径 |
|------|------|------|----------|
| after_phase_2.ckpt | DiffV2IR 主模型 | ~5.7GB | pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/ |
| BLIP | 图像描述生成 | ~1.4GB | models/blip/model_base_caption_capfilt_large.pth |
| CLIP | 视觉-语言理解 | ~900MB | models/clip/ViT-L-14.pt |
| SAM (可选) | 分割图生成 | ~2.4GB | models/sam/sam_vit_h_4b8939.pth |

#### 自定义模型路径
如果你想使用不同的模型存储位置，可以设置环境变量：
```bash
# 设置模型基础目录
export DIFFV2IR_MODEL_DIR=/your/custom/path/models

# 然后运行推理
python infer.py --input test_input --output test_output ...
```

或修改 `model_paths.py` 中的路径配置：
```python
MODEL_PATHS = {
    'blip': '/your/path/to/blip_model.pth',
    'clip': '/your/path/to/clip_model.pt',
    # ...
}
```

## 快速开始 - 推理测试

### 准备测试数据
```bash
# 创建测试目录
mkdir -p test_input test_input_seg test_output

# 将你的可见光图像放入 test_input 目录
# 例如：cp your_visible_image.jpg test_input/

# 生成分割图（如果没有的话）
# 选项 1：使用 SAM 自动生成（需要先下载 SAM 模型）
python generate_sam_masks.py --input test_input --output test_input_seg

# 选项 2：手动创建简单的分割图（全白图像）
# 对于每个输入图像，创建同名的 PNG 分割图
for img in test_input/*; do
    filename=$(basename "$img" | cut -d. -f1)
    convert -size 512x512 xc:white test_input_seg/${filename}.png
done
```

### 运行推理

#### 基础命令
```bash
# 使用 after_phase_2.ckpt 模型进行推理
python infer.py \
    --input test_input \
    --output test_output \
    --ckpt pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt \
    --steps 50 \
    --config configs/generate.yaml

```

#### 高级参数调整
```bash
python infer.py \
    --input test_input \
    --output test_output \
    --ckpt pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt \
    --steps 100 \              # 增加步数以获得更高质量（默认100）
    --resolution 512 \         # 处理分辨率（默认512）
    --cfg-text 7.5 \          # 文本引导强度（默认7.5）
    --cfg-image 1.5 \         # 图像条件强度（默认1.5）
    --cfg-seg 1.5 \           # 分割图引导强度（默认1.5）
    --seed 42 \               # 固定种子以获得可重现结果
    --config configs/generate.yaml
```

### 验证输出
```bash
# 查看生成的红外图像
ls -la test_output/

# 使用图像查看器查看结果
# eog test_output/*.png  # Ubuntu
# open test_output/*.png # macOS
```

## API 接口设计

### 1. 基础推理接口

```python
# 命令行接口完整参数说明
python infer.py \
    --input <输入图像文件夹> \        # 必需：包含可见光图像的文件夹
    --output <输出图像文件夹> \       # 必需：保存红外图像的文件夹
    --ckpt <模型权重路径> \           # 必需：预训练模型路径
    --steps <去噪步数> \              # 可选：默认100，范围20-200
    --resolution <分辨率> \           # 可选：默认512
    --cfg-text <文本引导强度> \       # 可选：默认7.5
    --cfg-image <图像条件强度> \      # 可选：默认1.5
    --cfg-seg <分割引导强度> \        # 可选：默认1.5
    --seed <随机种子> \               # 可选：用于结果复现
    --config configs/generate.yaml    # 必需：配置文件路径
```

**注意事项：**
1. 输入文件夹必须包含可见光图像（支持 jpg, png 格式）
2. 需要对应的分割图文件夹，命名为 `<input>_seg`
3. 分割图必须是 PNG 格式，文件名与输入图像对应（去除扩展名）

### 2. Python API 封装

```python
from diffv2ir_api import DiffV2IR

# 初始化模型
model = DiffV2IR(
    config_path="configs/generate.yaml",
    checkpoint_path="pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt",
    device="cuda"
)

# 单张图像转换
result = model.convert(
    input_image_path="path/to/visible_image.jpg",
    output_path="path/to/infrared_output.png",
    steps=50,  # 去噪步数，默认100
    seed=None  # 随机种子，None表示随机
)

# 批量转换
results = model.batch_convert(
    input_folder="path/to/input_folder",
    output_folder="path/to/output_folder",
    steps=50
)
```

### 3. REST API 服务

创建 `api_server.py`:

```python
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
model = None

@app.route('/init', methods=['POST'])
def init_model():
    global model
    config = request.json
    model = DiffV2IR(
        config_path=config.get('config_path', 'configs/generate.yaml'),
        checkpoint_path=config.get('checkpoint_path'),
        device=config.get('device', 'cuda')
    )
    return jsonify({"status": "success", "message": "Model initialized"})

@app.route('/convert', methods=['POST'])
def convert_image():
    if model is None:
        return jsonify({"error": "Model not initialized"}), 400
    
    # 接收 base64 编码的图像
    data = request.json
    image_base64 = data['image']
    steps = data.get('steps', 50)
    
    # 解码图像
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    
    # 保存临时文件
    temp_input = "temp_input.png"
    temp_output = "temp_output.png"
    image.save(temp_input)
    
    # 执行转换
    model.convert(temp_input, temp_output, steps=steps)
    
    # 读取结果并编码
    with open(temp_output, 'rb') as f:
        result_base64 = base64.b64encode(f.read()).decode()
    
    # 清理临时文件
    os.remove(temp_input)
    os.remove(temp_output)
    
    return jsonify({
        "status": "success",
        "result": result_base64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 输入输出规范

### 输入要求

1. **图像格式**：支持 PNG, JPG, JPEG
2. **图像尺寸**：
   - 推荐尺寸：512x512 或 256x256
   - 支持任意尺寸（会自动调整到最接近的 64 的倍数）
3. **文件结构**：
```
input_folder/
├── image1.jpg
├── image2.png
└── ...
```

4. **分割图（可选）**：
如需使用分割增强，需准备对应的分割图：
```
input_folder_seg/
├── image1.png  # 对应 input_folder/image1.jpg 的分割图
├── image2.png
└── ...
```

### 输出格式

1. **图像格式**：PNG（无损）
2. **图像尺寸**：与输入图像调整后的尺寸相同
3. **文件命名**：保持与输入文件相同的文件名

## 参数调优指南

### 核心参数

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `steps` | 100 | 20-200 | 去噪步数，越多质量越好但速度越慢 |
| `cfg_text` | 7.5 | 5.0-10.0 | 文本引导强度 |
| `cfg_image` | 1.5 | 1.0-3.0 | 图像条件强度 |
| `cfg_seg` | 1.5 | 1.0-3.0 | 分割图引导强度 |
| `resolution` | 512 | 256-1024 | 处理分辨率 |
| `seed` | None | 任意整数 | 随机种子，用于结果复现 |

### 性能优化建议

1. **快速预览**：使用 `steps=20-30` 进行快速预览
2. **高质量输出**：使用 `steps=100-150` 获得最佳质量
3. **批量处理**：建议批大小不超过 4 张（取决于 GPU 显存）
4. **显存优化**：
   - 8GB 显存：最大处理 512x512 图像
   - 16GB 显存：可处理 1024x1024 图像

## 前端集成示例

### React 集成示例

```javascript
// DiffV2IRService.js
class DiffV2IRService {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
        this.initialized = false;
    }
    
    async initialize(config) {
        const response = await fetch(`${this.apiUrl}/init`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });
        const result = await response.json();
        this.initialized = true;
        return result;
    }
    
    async convertImage(imageFile, options = {}) {
        if (!this.initialized) {
            throw new Error('Model not initialized');
        }
        
        // 读取图像文件
        const base64 = await this.fileToBase64(imageFile);
        
        // 发送请求
        const response = await fetch(`${this.apiUrl}/convert`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                image: base64,
                steps: options.steps || 50
            })
        });
        
        const result = await response.json();
        if (result.status === 'success') {
            return `data:image/png;base64,${result.result}`;
        }
        throw new Error(result.error);
    }
    
    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => {
                const base64 = reader.result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = reject;
        });
    }
}

// React 组件
function ImageConverter() {
    const [service] = useState(new DiffV2IRService('http://localhost:5000'));
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    
    useEffect(() => {
        service.initialize({
            checkpoint_path: 'pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt'
        });
    }, []);
    
    const handleConvert = async (file) => {
        setLoading(true);
        try {
            const infraredImage = await service.convertImage(file, {steps: 50});
            setResult(infraredImage);
        } catch (error) {
            console.error('Conversion failed:', error);
        }
        setLoading(false);
    };
    
    return (
        <div>
            <input type="file" onChange={(e) => handleConvert(e.target.files[0])} />
            {loading && <div>Converting...</div>}
            {result && <img src={result} alt="Infrared result" />}
        </div>
    );
}
```

## 完整示例 - 从零开始

以下是一个完整的从零开始配置和运行 DiffV2IR 的示例：

```bash
# 1. 创建工作目录
mkdir ~/diffv2ir_workspace
cd ~/diffv2ir_workspace

# 2. 克隆代码
git clone https://github.com/Jian-Zhang-3DV/DiffV2IR.git
cd DiffV2IR

# 3. 创建并激活虚拟环境
conda create -n DiffV2IR python=3.10.15 -y
conda activate DiffV2IR

# 4. 安装 PyTorch (CUDA 12.8)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# 5. 安装所有依赖
pip install -r requirements.txt

# 6. 下载 BLIP 和 CLIP 模型到本地
python download_models.py

# 7. 创建模型目录
mkdir -p pretrained/DiffV2IR/IR-500k/finetuned_checkpoints

# 8. 下载 DiffV2IR 主模型（手动从夸克网盘或 HuggingFace 下载后）
# 假设已下载到 ~/Downloads
mv ~/Downloads/after_phase_2.ckpt pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/

# 9. 准备测试图像
mkdir -p test_input test_input_seg test_output
# 复制一些测试图像到 test_input
cp ~/sample_images/*.jpg test_input/

# 10. 创建简单的分割图（如果没有 SAM）
for img in test_input/*; do
    filename=$(basename "$img" | cut -d. -f1)
    # 使用 Python PIL 创建白色图像
    python -c "
from PIL import Image
img = Image.open('$img')
seg = Image.new('RGB', img.size, (255, 255, 255))
seg.save('test_input_seg/${filename}.png')
"
done

# 11. 运行推理
python infer.py \
    --input test_input \
    --output test_output \
    --ckpt pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt \
    --steps 50 \
    --config configs/generate.yaml

# 12. 查看结果
ls -la test_output/
```

## 错误处理

### 常见错误及解决方案

1. **CUDA Out of Memory**
   ```bash
   # 错误信息：RuntimeError: CUDA out of memory
   
   # 解决方案：
   # 1. 减小分辨率
   python infer.py --resolution 256 ...
   
   # 2. 减少批处理（单张处理）
   # 3. 清理 GPU 缓存
   python -c "import torch; torch.cuda.empty_cache()"
   
   # 4. 使用较小的步数
   python infer.py --steps 20 ...
   ```

2. **模型加载失败**
   ```bash
   # 错误信息：FileNotFoundError 或 RuntimeError: Error(s) in loading state_dict
   
   # 检查模型文件
   ls -lh pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/
   
   # 验证模型完整性（应该约 5.7GB）
   du -h pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/*.ckpt
   
   # 确认 PyTorch 版本
   python -c "import torch; print(torch.__version__)"  # 应该是 2.8.0
   ```

3. **依赖包冲突**
   ```bash
   # 错误信息：ImportError 或 ModuleNotFoundError
   
   # 重新创建干净的环境
   conda deactivate
   conda env remove -n DiffV2IR
   conda create -n DiffV2IR python=3.10.15 -y
   conda activate DiffV2IR
   # 重新安装所有依赖...
   ```

4. **分割图缺失**
   ```bash
   # 错误信息：FileNotFoundError: [Errno 2] No such file or directory: 'xxx_seg/xxx.png'
   
   # 解决方案 1：创建简单的白色分割图
   python -c "
import os
from PIL import Image
import glob

input_dir = 'test_input'
seg_dir = 'test_input_seg'
os.makedirs(seg_dir, exist_ok=True)

for img_path in glob.glob(f'{input_dir}/*'):
    img = Image.open(img_path)
    filename = os.path.splitext(os.path.basename(img_path))[0]
    seg = Image.new('RGB', img.size, (255, 255, 255))
    seg.save(f'{seg_dir}/{filename}.png')
    print(f'Created segmentation for {filename}')
"
   
   # 解决方案 2：使用 SAM 生成（需要先下载 SAM 模型）
   python generate_sam_masks.py --input test_input --output test_input_seg
   ```

5. **CLIP 模型下载失败**
   ```bash
   # 错误信息：HTTPError 或连接超时
   
   # 手动下载 CLIP 模型
   mkdir -p ~/.cache/clip
   wget -O ~/.cache/clip/ViT-L-14.pt \
     https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
   ```

6. **BLIP 模型下载失败**
   ```bash
   # 错误信息：URLError 或 HTTPError
   
   # 使用下载脚本下载到本地目录
   python download_models.py --models blip
   
   # 或手动下载到指定位置
   mkdir -p models/blip
   wget -O models/blip/model_base_caption_capfilt_large.pth \
     https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth
   ```

7. **模型路径问题**
   ```bash
   # 如果模型不在默认位置，设置环境变量
   export DIFFV2IR_MODEL_DIR=/your/models/directory
   
   # 或修改 model_paths.py 中的路径配置
   ```

## 监控与日志

### 性能监控
```python
import time
import psutil
import GPUtil

def monitor_conversion(input_path, output_path):
    start_time = time.time()
    
    # 获取初始资源使用
    cpu_before = psutil.cpu_percent()
    mem_before = psutil.virtual_memory().percent
    gpu_before = GPUtil.getGPUs()[0].memoryUsed
    
    # 执行转换
    model.convert(input_path, output_path)
    
    # 计算资源使用
    duration = time.time() - start_time
    cpu_usage = psutil.cpu_percent() - cpu_before
    mem_usage = psutil.virtual_memory().percent - mem_before
    gpu_usage = GPUtil.getGPUs()[0].memoryUsed - gpu_before
    
    return {
        'duration': duration,
        'cpu_usage': cpu_usage,
        'memory_usage': mem_usage,
        'gpu_memory': gpu_usage
    }
```

### 日志配置
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diffv2ir.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DiffV2IR')
```

## 部署建议

### Docker 部署

```dockerfile
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app/DiffV2IR

# 安装 Python 依赖
RUN pip install -r DiffV2IR/requirements.txt

# 下载模型（可选，也可以挂载）
# RUN wget -O /app/models/after_phase_2.ckpt <model_url>

# 暴露端口
EXPOSE 5000

# 启动服务
CMD ["python", "DiffV2IR/api_server.py"]
```

### 生产环境配置

1. **使用 Gunicorn 部署 Flask 应用**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

2. **使用 Nginx 反向代理**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location /api/diffv2ir {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 50M;
    }
}
```

3. **负载均衡配置**（多 GPU 服务器）
```python
# 使用 Celery 进行任务队列管理
from celery import Celery

app = Celery('diffv2ir', broker='redis://localhost:6379')

@app.task
def convert_image_task(input_path, output_path, **kwargs):
    return model.convert(input_path, output_path, **kwargs)
```

## 最佳实践

1. **模型预热**：服务启动后先执行一次推理，避免首次请求延迟
2. **缓存机制**：对相同输入实施缓存，避免重复计算
3. **异步处理**：使用消息队列处理大批量转换任务
4. **健康检查**：实现 `/health` 端点监控服务状态
5. **版本管理**：为不同模型版本提供独立的端点

## 快速检查清单

在运行推理前，请确保以下所有项目都已完成：

- [ ] Python 3.10.15 环境已创建并激活
- [ ] PyTorch 2.8.0 和 CUDA 12.8 已安装
- [ ] 所有依赖包已正确安装（包括 Git 依赖）
- [ ] 预训练模型已下载到正确位置（至少一个 .ckpt 文件）
- [ ] 输入图像文件夹已准备
- [ ] 对应的分割图文件夹已准备（命名为 input_seg）
- [ ] 输出文件夹已创建

## 验证安装

运行以下脚本验证环境配置：

```python
# save as check_installation.py
import sys
import os

def check_installation():
    errors = []
    warnings = []
    
    # 检查 Python 版本
    if sys.version_info[:2] != (3, 10):
        warnings.append(f"Python version is {sys.version_info.major}.{sys.version_info.minor}, recommended 3.10")
    
    # 检查必要的包
    required_packages = [
        'torch', 'torchvision', 'numpy', 'transformers',
        'omegaconf', 'einops', 'PIL', 'cv2', 'clip'
    ]
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            errors.append(f"✗ {package} not installed")
    
    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            warnings.append("CUDA not available, will run on CPU (very slow)")
    except:
        pass
    
    # 检查模型文件
    model_paths = [
        'pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt'
    ]
    
    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            print(f"✓ Model found: {path}")
            model_found = True
    
    if not model_found:
        errors.append("✗ No pretrained models found")
    
    # 检查配置文件
    if os.path.exists('configs/generate.yaml'):
        print("✓ Config file found")
    else:
        errors.append("✗ configs/generate.yaml not found")
    
    # 总结
    print("\n" + "="*50)
    if errors:
        print("❌ Installation incomplete. Errors:")
        for error in errors:
            print(f"  {error}")
    else:
        print("✅ Installation successful!")
    
    if warnings:
        print("\n⚠️  Warnings:")
        for warning in warnings:
            print(f"  {warning}")
    
    return len(errors) == 0

if __name__ == "__main__":
    success = check_installation()
    sys.exit(0 if success else 1)
```

运行验证：
```bash
python check_installation.py
```

## 最小测试示例

创建一个最小测试来验证推理功能：

```bash
# 创建测试脚本
cat > test_inference.py << 'EOF'
import os
from PIL import Image
import numpy as np

# 创建测试目录
os.makedirs('minimal_test/input', exist_ok=True)
os.makedirs('minimal_test/input_seg', exist_ok=True)
os.makedirs('minimal_test/output', exist_ok=True)

# 创建一个简单的测试图像
test_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
test_img.save('minimal_test/input/test.jpg')

# 创建对应的分割图
seg_img = Image.new('RGB', (512, 512), (255, 255, 255))
seg_img.save('minimal_test/input_seg/test.png')

print("Test data created. Run inference with:")
print("python infer.py \\")
print("    --input minimal_test/input \\")
print("    --output minimal_test/output \\")
print("    --ckpt pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt \\")
print("    --steps 20 \\")
print("    --config configs/generate.yaml")
EOF

python test_inference.py
```

## 技术支持

- GitHub Issues: https://github.com/Jian-Zhang-3DV/DiffV2IR/issues
- 原始项目: https://github.com/your-original-repo/DiffV2IR
- 论文链接: https://arxiv.org/abs/2503.19012
- 项目主页: https://diffv2ir.github.io/

## 许可证

本项目基于 Stable Diffusion 代码库开发，遵循相应的开源许可证。

## 更新日志

- 2024-08-24: 更新集成指南，添加完整的环境配置和推理说明
- 2024-08-21: 初始版本发布