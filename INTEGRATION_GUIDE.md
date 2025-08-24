# DiffV2IR 前端集成指南

## 项目简介

DiffV2IR 是一个基于扩散模型的可见光到红外图像转换系统，通过视觉-语言理解实现高质量的图像转换。该系统包含两个核心模块：
- **渐进学习模块 (PLM)**：使用多阶段知识学习实现从全波段到目标波长的红外转换
- **视觉-语言理解模块 (VLUM)**：结合语言描述和分割图增强语义感知能力

## 环境配置

### 1. 系统要求
- Python 3.10.15
- CUDA 支持的 GPU（建议显存 >= 8GB）
- Linux/Unix 系统（推荐 Ubuntu 20.04+）

### 2. 创建虚拟环境
```bash
conda create -n DiffV2IR python=3.10.15
conda activate DiffV2IR
```

### 3. 安装依赖
```bash
cd DiffV2IR
pip install -r requirements.txt
```

主要依赖包括：
- torch==1.13.1
- torchvision==0.14.1
- numpy==1.26.0
- transformers==4.26.1
- pytorch-lightning==1.4.2
- diffusers
- opencv-python==4.10.0.84

### 4. 下载预训练模型

#### 模型权重
从以下链接下载预训练模型：
- 夸克网盘：https://pan.quark.cn/s/e2f28304ee90 (访问码：EWCz)
- HuggingFace：https://huggingface.co/datasets/Lidong26/IR-500K/tree/main

将下载的模型文件放置在：
```
DiffV2IR/pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/
```

可用的预训练模型：
- `after_phase_2.ckpt` - PLM 第二阶段后的通用模型
- `M3FD.ckpt` - 在 M3FD 数据集上微调的模型

#### BLIP 模型（自动下载）
首次运行时会自动下载 BLIP 模型用于图像描述生成。

#### SAM 模型（可选）
如需使用 SAM 增强分割功能：
```
DiffV2IR/SAM_models/sam_vit_h_4b8939.pth
```

## API 接口设计

### 1. 基础推理接口

```python
# 命令行接口
python infer.py \
    --input <输入图像文件夹> \
    --output <输出图像文件夹> \
    --ckpt <模型权重路径> \
    --steps <去噪步数> \
    --config configs/generate.yaml
```

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

## 错误处理

### 常见错误及解决方案

1. **CUDA Out of Memory**
   - 减小 `resolution` 参数
   - 减少批处理大小
   - 使用较小的模型

2. **模型加载失败**
   - 检查模型文件路径
   - 确认模型文件完整性
   - 验证 PyTorch 版本兼容性

3. **依赖包冲突**
   - 使用指定版本的依赖包
   - 在独立的虚拟环境中运行

4. **分割图缺失**
   - 使用 `generate_sam_masks.py` 生成分割图
   - 或禁用分割增强功能

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

## 技术支持

- GitHub Issues: https://github.com/your-repo/issues
- 论文链接: https://arxiv.org/abs/2503.19012
- 项目主页: https://diffv2ir.github.io/

## 许可证

本项目基于 Stable Diffusion 代码库开发，遵循相应的开源许可证。