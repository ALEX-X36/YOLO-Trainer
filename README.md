# 🐟 YOLO 鱼类识别训练器

基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 的目标检测模型训练平台，提供可视化 WebUI（[Gradio](https://www.gradio.app/)），支持 YOLOv8/YOLOv11 全系列模型的训练、监控、评估与导出。专为 SeeFish 鱼类识别场景设计，也适用于任意目标检测任务。

---

## ✨ 功能特性

| 模块 | 功能 |
|------|------|
| 📦 **数据集管理** | 上传 ZIP 格式的 YOLO 数据集，自动验证结构、提取并注册；支持浏览详情与删除 |
| 📥 **模型下载** | 预下载 YOLOv8/YOLOv11 全系列 10 款预训练权重，实时显示进度、速度与 ETA |
| ⚙️ **训练配置** | 可视化配置 20+ 超参数（学习率、优化器、增强策略等），支持 5 种优化器、6 种输入尺寸 |
| 📊 **训练监控** | 实时 epoch 进度条、7 项指标仪表盘 (mAP/Precision/Recall/Loss)、滚动日志面板 |
| 📈 **结果与导出** | 查看训练曲线（results/混淆矩阵/PR/F1）、最优/最终权重路径；支持 6 种格式模型导出 |

---

## 📁 项目结构

```
Trainer/
├── app.py                  # 主入口 — Gradio WebUI 构建 + 事件处理
├── config.py               # 全局配置 — 模型变体、超参数默认值、路径常量
├── download_manager.py     # 模型下载管理 — 缓存检测、线程下载、进度上报
├── trainer.py              # 训练引擎 — 后台训练线程、队列通信、设备检测
├── dataset_manager.py      # 数据集管理 — ZIP 验证、提取、CRUD 操作
├── callbacks.py            # 训练回调 — Ultralytics 事件 → 队列消息桥接
├── utils.py                # 工具函数 — YAML 读写、目录管理、文件统计
├── ui_components.py        # UI 组件 — 各 Tab 的 Gradio 组件构建函数（独立版）
├── requirements.txt        # Python 依赖清单
├── datasets/               # 上传的数据集存放目录（自动创建）
├── outputs/                # 训练输出目录（自动创建）
├── exports/                # 模型导出目录（自动创建）
└── logs/                   # 日志目录（自动创建）
```

### 模块详解

#### `app.py` — 主入口 & UI

应用入口，构建 5 个功能标签页，包含所有事件处理函数：

- **5 个标签页**：数据集管理 → 模型下载 → 训练配置 → 训练监控 → 结果与导出
- **事件处理函数**：`handle_upload`、`handle_start_training`、`handle_monitor_refresh`、`handle_export_model`、`handle_refresh_model_status`、`handle_download_model` 等
- **启动参数**：`--port`（默认 7860）、`--host`（默认 127.0.0.1）、`--share`（生成公网链接）

#### `config.py` — 全局配置

| 配置项 | 说明 | 内容 |
|--------|------|------|
| `MODEL_VARIANTS` | 支持模型 | YOLOv8n/s/m/l/x、YOLOv11n/s/m/l/x 共 10 款 |
| `MODEL_SIZES` | 模型文件大小 (MB) | 用于下载进度估算 |
| `OPTIMIZER_CHOICES` | 优化器 | auto / SGD / Adam / AdamW / RMSProp |
| `IMAGE_SIZE_CHOICES` | 输入尺寸 | 320 / 416 / 512 / 640 / 800 / 1024 |
| `EXPORT_FORMATS` | 导出格式 | PyTorch / ONNX / TensorRT / OpenVINO / CoreML / TFLite |
| `DEFAULT_TRAINING_PARAMS` | 默认超参数 | 见下方超参数参考表 |
| `UPLOAD_REQUIREMENTS` | 数据集格式说明 | ZIP 结构的 Markdown 文档 |

#### `download_manager.py` — 模型下载

- **`find_model_file(model_id)`**：在 3 个位置查找已缓存模型（torch hub / ultralytics cache / 项目目录）
- **`check_all_models()`**：遍历 10 个模型，返回结构化下载状态
- **`download_model(name, model_id, queue)`**：后台线程下载，通过 `queue.Queue` 每秒上报进度事件（`start` / `progress` / `complete` / `error` / `already_cached`）

下载进度监控原理：启动 `YOLO(model_id)` 子线程触发 Ultralytics 内置下载，同时主控线程每秒轮询目标文件大小，计算百分比、瞬时/Avg 速度及 ETA。

#### `trainer.py` — 训练引擎

- **`TrainingState`**：线程安全的训练状态数据类（线程引用、停止标志、日志队列、epoch 计数、指标字典）
- **`run_training(params, state)`**：后台训练主函数 — 加载模型 → 注册回调 → 构建参数 → 调用 `model.train()`
- **`stop_training(state)`**：设置停止标志，通知 Ultralytics 在当前 epoch 后停止
- **`get_device_info()`**：返回 GPU（名称、显存）或 CPU+系统内存信息
- **`parse_training_results(output_dir)`**：解析训练输出目录，提取指标 CSV、权重文件、可视化图表

#### `dataset_manager.py` — 数据集管理

- **`validate_zip_structure(zip_path)`**：验证 ZIP 是否包含 `data.yaml`、`images/`、`labels/` 三层必需结构
- **`extract_dataset(zip_path, name)`**：解压并自动修复 `data.yaml` 中的路径
- **`list_datasets()` / `get_dataset_info()` / `delete_dataset()`**：数据集 CRUD
- **`validate_yolo_annotations(label_dir, class_count)`**：验证 YOLO 标注格式（归一化坐标范围、类别 ID 有效性）

#### `callbacks.py` — 训练回调

- **`create_log_callback(queue)`**：返回 Ultralytics 回调字典，在 `on_pretrain_routine_end`、`on_fit_epoch_end`、`on_train_end` 时将事件推入 UI 队列
- **`LogCaptureHandler`**：自定义 `logging.Handler`，捕获 Ultralytics 日志输出并转发到队列
- **`setup_log_capture` / `remove_log_capture`**：安装/卸载日志捕获

#### `utils.py` — 工具函数

| 函数 | 说明 |
|------|------|
| `read_yaml(path)` / `write_yaml(path, data)` | YAML 文件读写 |
| `count_images(dir)` / `count_labels(dir)` | 递归统计图片/标注文件数 |
| `ensure_dir(path)` | 确保目录存在 |
| `generate_run_name(dataset, model)` | 生成唯一训练运行名 `{dataset}_{model}_YYYYMMDD_HHMMSS` |
| `list_subdirs(dir)` | 列出子目录（按修改时间降序） |
| `get_dir_size_mb(dir)` / `format_size(mb)` | 目录大小计算与格式化 |

---

## 🚀 快速开始

### 环境要求

- **Python** >= 3.10
- **CUDA** (推荐) >= 11.8 + 兼容的 NVIDIA 驱动
- **显存建议**：至少 4 GB（YOLOv8n）；训练大模型 (l/x) 推荐 8+ GB

### 安装

```bash
# 1. 克隆项目（或进入项目目录）
cd Trainer

# 2. 创建虚拟环境（推荐）
python -m venv venv

# 3. 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt
```

**核心依赖**（详见 `requirements.txt`）：

| 依赖 | 版本 | 用途 |
|------|------|------|
| `torch` | >= 2.0.0 | 深度学习框架 |
| `ultralytics` | >= 8.0.0 | YOLO 训练/推理引擎 |
| `gradio` | >= 4.0.0 | Web UI 框架 |
| `numpy` | >= 1.24.0 | 数值计算 |
| `pandas` | >= 2.0.0 | 数据处理 (results.csv) |
| `matplotlib` | >= 3.7.0 | 训练曲线渲染 |
| `seaborn` | >= 0.12.0 | 增强可视化 |
| `Pillow` | >= 10.0.0 | 图像处理 |
| `PyYAML` | >= 6.0 | YAML 配置解析 |
| `psutil` | >= 5.9.0 | 系统资源监控 |
| `onnx` / `onnxruntime` | >= 1.14.0 / 1.15.0 | ONNX 模型导出（可选） |

### 启动

```bash
python app.py
```

默认启动在 `http://127.0.0.1:7860`，浏览器自动打开。

命令行参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `7860` | WebUI 监听端口 |
| `--host` | `127.0.0.1` | 监听地址（`0.0.0.0` 监听所有网卡） |
| `--share` | `False` | 生成 Gradio 公开分享链接（临时公网访问） |

示例：

```bash
python app.py --port 8080 --host 0.0.0.0
python app.py --share
```

---

## 📖 使用指南

### 工作流概览

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 1. 上传   │ → │ 2. 下载   │ → │ 3. 训练   │ → │ 4. 监控   │ → │ 5. 导出   │
│   数据集   │   │   预训练   │   │   配置     │   │   进度     │   │   模型     │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### Tab 1 — 📦 数据集管理

上传并管理训练数据集。

**操作步骤：**
1. 准备好符合 YOLO 格式的 ZIP 数据集（结构要求见下方）
2. 点击选择 ZIP 文件，可自定义数据集名称（留空则自动使用文件名）
3. 点击 **上传并验证** — 系统自动检查结构合法性
4. 右侧列表显示所有已上传的数据集，选中可查看详情

**数据集 ZIP 必须结构：**

```
my_fish_dataset.zip
├── data.yaml              # 数据集配置文件
├── images/
│   ├── train/             # 训练图片（.jpg / .png / .bmp / .tiff / .webp）
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/               # 验证图片
│       ├── img_101.jpg
│       └── ...
└── labels/
    ├── train/             # 训练标注（.txt，与训练图片同名）
    │   ├── img_001.txt
    │   └── ...
    └── val/               # 验证标注
        ├── img_101.txt
        └── ...
```

**`data.yaml` 格式：**

```yaml
path: ./                    # 数据集根目录（extract_dataset 会自动修复为实际路径）
train: images/train         # 训练集图片路径（相对 path）
val: images/val             # 验证集图片路径
nc: 3                       # 类别数量
names:                      # 类别名称（从 0 开始编号）
  0: goldfish
  1: koi
  2: catfish
```

**YOLO 标注格式（每行一个目标）：**

```
class_id cx cy w h
```

- 所有坐标均为**归一化值**（相对于图片宽高，0~1）
- `class_id`：类别编号（整数，从 0 开始）
- `cx, cy`：目标中心点坐标
- `w, h`：目标宽高

示例（一张图片有 2 个目标）：
```
0 0.5125 0.3500 0.1250 0.2500
1 0.2300 0.6800 0.0800 0.1200
```

### Tab 2 — 📥 模型下载

在实际训练前预先下载 YOLO 预训练权重，避免训练时等待。

**操作步骤：**
1. 点击 **🔄 刷新状态** — 查看 10 款模型的缓存情况
2. 从下拉菜单选择目标模型，点击 **📥 下载选中模型**
3. 或直接点击 **📥 下载全部** 依次下载所有未缓存模型
4. 进度条和日志面板实时显示下载速度、文件大小和预计剩余时间

**支持的预训练模型：**

| 模型 | 文件名 | 大小 | 适用场景 |
|------|--------|------|----------|
| YOLOv8n | `yolov8n.pt` | ~6 MB | 轻量级、边缘设备 |
| YOLOv8s | `yolov8s.pt` | ~22 MB | 小模型、实时检测 |
| YOLOv8m | `yolov8m.pt` | ~50 MB | 中等精度/速度平衡 |
| YOLOv8l | `yolov8l.pt` | ~84 MB | 高精度 |
| YOLOv8x | `yolov8x.pt` | ~130 MB | 最高精度 |
| YOLOv11n | `yolo11n.pt` | ~5 MB | 新一代轻量级 |
| YOLOv11s | `yolo11s.pt` | ~18 MB | 新一代小模型 |
| YOLOv11m | `yolo11m.pt` | ~39 MB | 新一代中等模型 |
| YOLOv11l | `yolo11l.pt` | ~67 MB | 新一代大模型 |
| YOLOv11x | `yolo11x.pt` | ~109 MB | 新一代超大型 |

模型缓存位置：`~/.cache/torch/hub/checkpoints/`（PyTorch Hub 默认缓存目录）

### Tab 3 — ⚙️ 训练配置

可视化配置所有训练超参数。

**基本设置：**
- **训练数据集**：从已上传的数据集中选择
- **YOLO 模型**：选择预训练模型变体（与 Tab 2 下载的对应）

**超参数参考表：**

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| **Epochs** | 100 | 1–500 | 总训练轮数；小数据集 100~300，大数据集 50~100 |
| **Batch Size** | 16 | 1–128 | 批次大小；显存不足时减小（如 8/4） |
| **Image Size** | 640 | 320–1024 | 输入图片尺寸；小目标检测建议用较大值 |
| **Optimizer** | auto | SGD/Adam/AdamW/RMSProp | 优化器；`auto` 由 Ultralytics 自动选择 |
| **LR (初始)** | 0.01 | 0.0001–0.1 | 初始学习率；SGD 通常用 0.01，Adam 用 0.001 |
| **LR (最终因子)** | 0.01 | 0.001–0.1 | 最终学习率 = lr0 × lrf，余弦调度在训练结束时降至该值 |
| **Weight Decay** | 0.0005 | 0–0.01 | L2 正则化系数 |
| **Momentum** | 0.937 | 0.5–0.999 | SGD 动量 |
| **Warmup Epochs** | 3 | 0–20 | 学习率预热轮数；从 lr0/100 线性增长至 lr0 |
| **Patience** | 50 | 5–200 | 早停耐心值；连续 N 个 epoch 无改善则停止 |
| **Close Mosaic** | 10 | 0–50 | 最后 N 个 epoch 关闭 Mosaic 增强 |
| **Workers** | 8 | 1–32 | 数据加载子进程数；CPU 核心数以内 |
| **Seed** | 0 | 0–9999 | 随机种子，设为 0 则随机 |
| **Dropout** | 0.0 | 0.0–0.5 | 分类头 Dropout 比率；用于防过拟合 |
| **Warmup Momentum** | 0.8 | 0.0–1.0 | 预热期间动量初始值 |
| **Warmup Bias LR** | 0.1 | 0.0–1.0 | 预热期间偏置项学习率 |

**训练选项：**

| 选项 | 默认 | 说明 |
|------|------|------|
| 预训练权重 | ✅ 开启 | 加载 COCO 预训练权重；关闭则从头训练 |
| 混合精度 (AMP) | ✅ 开启 | 自动混合精度训练，省显存、加速 |
| 余弦学习率 | ❌ 关闭 | 使用余弦退火调度替代线性衰减 |
| 多尺度训练 | ❌ 关闭 | 每 batch 随机缩放图片 (0.5×~1.5×) |
| 从检查点恢复 | ❌ 关闭 | 从上次中断的 checkpoint 继续训练 |

点击 **▶️ 开始训练** 启动后台训练，点击 **⏹️ 停止训练** 在当前 epoch 结束后优雅停止。

### Tab 4 — 📊 训练监控

训练过程中的实时监控面板。

**监控内容：**
- **Epoch 进度**：进度条 + "当前epoch/总epochs" 数字显示
- **实时指标仪表盘**（每秒刷新）：
  - mAP@50 — 验证集 mAP at IoU=0.5
  - mAP@50-95 — 验证集 mAP at IoU=0.5~0.95（更严格）
  - Precision — 精确率
  - Recall — 召回率
  - Box Loss — 边界框损失
  - Cls Loss — 分类损失
  - DFL Loss — 分布式焦点损失
- **GPU 信息**：GPU 名称 + 总显存
- **训练日志**：滚动显示 Ultralytics 完整输出（含每个 epoch 的 Loss 和 mAP 汇总）

### Tab 5 — 📈 结果与导出

查看历史训练运行结果、可视化曲线以及导出模型。

**结果查看：**
- 从下拉菜单选择训练运行，自动显示：
  - 📊 **运行详情表**：CSV 最后一行的所有指标
  - 📈 **训练曲线**（4 张图）：综合结果图、混淆矩阵、PR 曲线、F1 曲线

**模型导出：**

支持将训练好的 `best.pt` 导出为 6 种格式：

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| **PyTorch** | `.pt` | 原始 PyTorch 格式，直接复制 |
| **ONNX** | `.onnx` | 跨平台推理格式 |
| **TensorRT** | `.engine` | NVIDIA GPU 高性能推理 |
| **OpenVINO** | 目录 | Intel CPU/VPU 推理 |
| **CoreML** | `.mlpackage` | Apple 设备推理 |
| **TFLite** | `.tflite` | 移动端/嵌入式推理 |

> ⚠️ 部分导出格式需要额外环境依赖（如 `onnx`、`openvino`、`coremltools`），导出失败时会显示提示。

---

## 🏗️ 架构设计

### 线程模型

```
┌─────────────────────────────────────────────────┐
│                    Gradio UI                      │
│  (主线程 — 唯一的 Gradio 事件循环)                  │
│                                                    │
│  ┌─────────────┐  ┌──────────────┐                │
│  │ gr.Timer(1s) │  │ Handler (click)│               │
│  │ 轮询 Queue   │  │ 启动 Thread   │               │
│  └──────┬───────┘  └──────┬───────┘                │
│         │                 │                        │
│         ▼                 ▼                        │
│  ┌────────────────┐  ┌────────────────┐            │
│  │ queue.Queue()  │◄─│ 训练线程         │            │
│  │ 线程安全消息队列│  │ (daemon thread) │            │
│  └────────────────┘  │ run_training()  │            │
│                      │ download_model() │            │
│                      └────────────────┘            │
└─────────────────────────────────────────────────┘
```

**核心设计原则：**
1. **训练/下载在后台 daemon 线程执行** — 不阻塞 Gradio 事件循环
2. **线程间通过 `queue.Queue` 通信** — 后台线程推送事件，主线程通过 `gr.Timer` 每秒轮询
3. **事件类型驱动 UI 更新** — `start` / `epoch` / `progress` / `complete` / `error` 事件分别更新不同 UI 区域
4. **优雅停止** — 通过 `threading.Event` + `model.stop_training` 实现 epoch 边界停止

### 队列事件协议

**训练队列事件：**

| 事件类型 | 关键字段 | 说明 |
|----------|----------|------|
| `start` | `total_epochs` | 训练开始，总 epoch 数 |
| `epoch` | `epoch`, `total`, `metrics` | 每个 epoch 结束，含指标字典 |
| `log` | `message` | Ultralytics 日志消息 |
| `end` | `reason`, `final_metrics` | 训练完成/停止 |
| `error` | `message` | 训练异常 |

**下载队列事件：**

| 事件类型 | 关键字段 | 说明 |
|----------|----------|------|
| `start` | `model_name`, `model_id` | 下载开始 |
| `progress` | `percent`, `downloaded_mb`, `speed_mbps`, `eta_s` | 下载进度（每秒） |
| `complete` | `path`, `size_mb`, `elapsed_s`, `speed_mbps` | 下载完成 |
| `already_cached` | `path`, `size_mb` | 已缓存，跳过 |
| `error` | `message` | 下载失败 |

---

## ⚙️ 高级配置

### 自定义默认超参数

编辑 [config.py](config.py) 中的 `DEFAULT_TRAINING_PARAMS` 字典：

```python
DEFAULT_TRAINING_PARAMS = {
    "epochs": 150,       # 修改默认训练轮数
    "batch": 32,         # 修改默认批次大小
    "imgsz": 800,        # 修改默认图片尺寸
    # ... 其他参数
}
```

### 添加新的模型变体

在 [config.py](config.py) 的 `MODEL_VARIANTS` 和 `MODEL_SIZES` 中添加：

```python
MODEL_VARIANTS = {
    # ... 已有模型 ...
    "YOLOv12n": "yolo12n.pt",  # 添加新模型
}

MODEL_SIZES = {
    # ... 已有大小 ...
    "yolo12n.pt": 6.8,  # 添加对应大小（可选，用于下载进度估算）
}
```

### 自定义 CSS

编辑 [config.py](config.py) 中的 `CUSTOM_CSS` 变量，或在 [app.py](app.py) 中修改 `DESKTOP_CSS`。

---

## ❓ 常见问题

### Q: 训练时提示 "CUDA out of memory"

**解决方案：**
- 减小 `Batch Size`（如从 16 降至 8 或 4）
- 减小 `Image Size`（如从 640 降至 416）
- 选择更小的模型变体（如从 YOLOv8l 降至 YOLOv8s）
- 确认没有其他进程占用显存（`nvidia-smi` 查看）

### Q: 下载模型时速度很慢或失败

**解决方案：**
- 检查网络连接（模型从 GitHub Releases 下载）
- 如在国内，可设置 HuggingFace 镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- 也可以手动下载 `.pt` 文件放到 `~/.cache/torch/hub/checkpoints/` 目录

### Q: 模型导出的格式报错 "缺少依赖"

**解决方案：**
- ONNX 导出：`pip install onnx onnxruntime`
- TensorRT 导出：需要安装 TensorRT SDK
- OpenVINO 导出：`pip install openvino`
- CoreML 导出：`pip install coremltools`（仅 macOS）
- TFLite 导出：`pip install tensorflow`

### Q: 训练过程中如何暂停/恢复？

不支持暂停。可以在当前 epoch 完成后 **停止训练**（点击 ⏹️ 停止训练），然后在训练配置中勾选 **"从检查点恢复"**，使用相同的模型和数据集重新开始训练。Ultralytics 会自动从 `last.pt` 恢复。

### Q: 数据集上传后图片数为 0

**检查：**
- ZIP 中是否包含 `images/train/` 和 `images/val/` 子目录
- 图片扩展名是否为支持的格式（`.jpg` / `.jpeg` / `.png` / `.bmp` / `.tiff` / `.webp`）
- ZIP 不要嵌套多余的顶层目录（系统会自动检测并剥离单层前缀）

### Q: 训练输出的文件在哪里？

在 `outputs/<run_name>/` 目录下：
- `weights/best.pt` — 最优权重（mAP 最高）
- `weights/last.pt` — 最后 epoch 权重
- `results.csv` — 完整指标记录
- `results.png` — 综合训练曲线图
- `confusion_matrix.png` — 混淆矩阵
- `PR_curve.png` / `F1_curve.png` — 精度-召回 / F1 曲线

---

## 📄 许可

本项目基于 [Ultralytics AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 开发。使用 YOLO 模型需遵守 Ultralytics 的许可条款。

---

## 🔗 相关链接

- [Ultralytics YOLO 文档](https://docs.ultralytics.com/)
- [Gradio 文档](https://www.gradio.app/docs)
- [SeeFish 项目](https://github.com/ALEX-X36/SeeFish) — 使用本训练器产出的模型进行鱼类识别
