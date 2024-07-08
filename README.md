# florence2-ft-simple
简单的微调属于你的florence2模型

## 介绍
本项目用于微调 [Florence2](https://huggingface.co/microsoft/Florence-2-large) 系列模型的 `<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>` 指令。

## 环境设置

### 1. 克隆项目

```bash
git clone https://github.com/facok/florence2-ft-simple
cd florence2-ft-simple
```

### 2. 创建和激活虚拟环境

使用 virtualenv 或 conda 创建并激活虚拟环境：

#### 使用 virtualenv (linux)
```bash
python -m venv venv
source venv/bin/activate
```
#### 使用 virtualenv (windows)
```bash
python -m venv venv
.\venv\Scripts\activate
```

#### 使用 conda
```bash
conda create -n myenv python=3.10
conda activate myenv
```

### 3. 安装依赖项
#### 通过 pip 安装 torch (linux)
```bash
pip install torch
```
#### 通过 pip 安装 torch (windows)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
#### 通过 conda 安装 torch
```bash
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```
#### 继续安装剩余依赖
```bash
pip install -r requirements.txt
```

## 文件说明
`dataset.py` 包含数据集相关的类和方法

`train.py` 包含训练模型的主要逻辑

`config.py` 包含配置解析的逻辑

`main.py` 项目的主入口文件，负责初始化和启动训练过程

## 参数说明

| 参数名称               | 类型       | 默认值  | 必需  | 说明 |
|------------------------|------------|---------|-------|------|
| `--images_dir`         | `str`      | 无      | 是    | 图像文件所在的目录路径。 |
| `--texts_dir`          | `str`      | 无      | 是    | 打标文件所在的目录路径。每个文本文件应对应一个图像文件。 |
| `--model_dir`          | `str`      | 无      | 是    | 预训练模型所在的目录路径。 |
| `--output_dir`         | `str`      | 无      | 是    | 模型保存的目录路径。 |
| `--batch_size`         | `int`      | 1       | 否    | 训练和验证时的批处理大小。 |
| `--num_workers`        | `int`      | 0       | 否    | `DataLoader` 的工作进程数。 |
| `--epochs`             | `int`      | 3       | 否    | 训练的轮数。 |
| `--learning_rate`      | `float`    | 1e-6    | 否    | 训练的学习率。 |
| `--accumulation_steps` | `int`      | 8       | 否    | 梯度累积的步数，用于模拟更大批处理大小。 |
| `--task_type`          | `str`      | 无      | 是    | 任务类型，需要微调的任务种类。可选值为：`<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`。 |
| `--precision`          | `str`      | `bf16`  | 否    | 训练的精度。可选值为：`fp32`, `fp16`, `bf16`。 |
| `--train_split`        | `float`    | 0.8     | 否    | 训练数据和验证数据的比例，范围在 0.0 到 1.0 之间。 |
| `--save_best_model`    | `store_true` | 无    | 否    | 仅在验证损失减少时保存模型。 |



## 运行训练脚本
#### 在项目根目录下运行以下命令，启动训练过程：

```bash

python main.py --images_dir <path-to-images> --texts_dir <path-to-texts> --model_dir <path-to-model> --output_dir <path-to-output> --task_type <task-type> --batch_size <batch-size> --epochs <epochs> --learning_rate <learning-rate> --accumulation_steps <accumulation-steps> --precision <precision> --train_split <train-split> --save_best_model
```
#### 示例命令 （linux）
```bash

python main.py \
    --images_dir ./data/images \
    --texts_dir ./data/texts \
    --model_dir ./Florence-2-large \
    --output_dir ./output \
    --task_type "<MORE_DETAILED_CAPTION>" \
    --batch_size 1 \
    --epochs 3 \
    --learning_rate 1e-6 \
    --accumulation_steps 8 
```
## 日志记录
暂时没有

## 其它说明
在BF16精度，BS 1的情况下，微调 [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) 大约需要 <22G显存

## 感谢
https://github.com/andimarafioti/florence2-finetuning
