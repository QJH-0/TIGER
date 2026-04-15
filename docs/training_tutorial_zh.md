# TIGER 训练流程教程（MiniLibriMix 默认）

本文档面向当前仓库的实际用法：默认使用 `MiniLibriMix` 训练。

## 1. 环境准备

```bash
git clone https://github.com/JusperLee/TIGER.git
cd TIGER
pip install -r requirements.txt
```

首次使用 W&B：

```bash
wandb login
```

## 2. 数据准备与索引生成

原始数据目录（默认）：

- `D:\Paper\datasets\MiniLibriMix`

使用项目脚本生成训练索引（JSON）：

```bash
python DataPreProcess/process_librimix.py --in_dir "D:/Paper/datasets/MiniLibriMix" --out_dir "D:/Paper/TIGER/DataPreProcess/MiniLibriMix" --splits train val test --speakers mix_both s1 s2
```

生成后目录结构应为：

- `DataPreProcess/MiniLibriMix/train/{mix_both.json,s1.json,s2.json}`
- `DataPreProcess/MiniLibriMix/val/{mix_both.json,s1.json,s2.json}`
- `DataPreProcess/MiniLibriMix/test/{mix_both.json,s1.json,s2.json}`

每条 JSON 记录格式：

```json
["绝对路径.wav", 采样点长度]
```

## 3. 配置文件说明

默认训练配置：`configs/tiger-large.yml`

当前已切换为：

- `datamodule.data_name: Libri2MixModuleRemix`
- `train_dir: DataPreProcess/MiniLibriMix/train`
- `valid_dir: DataPreProcess/MiniLibriMix/val`
- `test_dir: DataPreProcess/MiniLibriMix/test`

> 配置中原 EchoSet 路径已注释保留，便于回切。

## 4. 启动训练

```bash
python audio_train.py --conf_dir configs/tiger-large.yml
```

训练输出目录：

- `Experiments/checkpoint/<exp_name>/`

关键产物：

- `best_model.pth`
- `best_k_models.json`
- `last.ckpt` 与多份 epoch ckpt

## 5. 评估模型

```bash
python audio_test.py --conf_dir configs/tiger-large.yml
```

结果默认输出：

- `Experiments/checkpoint/<exp_name>/results/metrics.csv`

## 6. 可选数据维护脚本

如果你调整了 MiniLibriMix 切分，可使用：

```bash
# 按样本 ID 从 val 同步切分一半到 test（跨 mix_both/mix_clean/noise/s1/s2 一致）
python DataPreProcess/split_val_to_test.py

# 同步 metadata 目录的 CSV（val/test）
python DataPreProcess/sync_minilibrimix_metadata.py
```

## 7. 常见问题

- **找不到 JSON**：确认 `DataPreProcess/MiniLibriMix/*` 已生成且路径与配置一致。
- **W&B 报错**：先执行 `wandb login`，或在服务器设置 `WANDB_API_KEY`。
- **显存不足**：减小 `batch_size`，或降低模型规模参数。
- **多卡问题**：先单卡跑通，再切换 DDP。

# TIGER 模型训练流程教程

本文档基于当前仓库的真实脚本整理，覆盖从数据准备到训练、评估与排错的完整流程。

## 1. 环境准备

```bash
git clone https://github.com/JusperLee/TIGER.git
cd TIGER
pip install -r requirements.txt
```

W&B 只需在同一机器同一环境登录一次：

```bash
wandb login
```

## 2. 数据准备

项目训练依赖 `json` 索引文件，而不是直接扫音频目录。`datamodule` 会读取：

- `mix.json` / `mix_both.json`
- `s1.json`
- `s2.json`

每条记录格式为：

```json
["/abs/path/to/audio.wav", 采样点长度]
```

---

### 2.1 EchoSet 数据（默认流程）

1) 按仓库约定放好原始数据（`train/val/test`）。  
2) 生成训练所需 JSON：

```bash
python DataPreProcess/process_echoset.py --in_dir <你的EchoSet原始目录> --out_dir DataPreProcess/EchoSet
```

生成后应包含：

- `DataPreProcess/EchoSet/train/{mix.json,s1.json,s2.json}`
- `DataPreProcess/EchoSet/val/{mix.json,s1.json,s2.json}`
- `DataPreProcess/EchoSet/test/{mix.json,s1.json,s2.json}`

---

### 2.2 MiniLibriMix 数据（你当前本地流程）

你已经把 `val` 一半切到 `test`，并保持 `mix_both/mix_clean/noise/s1/s2` 同名一一对应。后续可复用以下脚本：

```bash
# 1) 切分 val -> test（按样本ID同步移动）
python DataPreProcess/split_val_to_test.py

# 2) 同步 metadata（生成/更新 val/test 的 mixture_*.csv）
python DataPreProcess/sync_minilibrimix_metadata.py
```

> 注意：`metadata/*.csv` 主要用于数据管理与核对；当前训练 datamodule 读取的是 `json` 索引。  
> 如果你要直接用 MiniLibriMix 训练，需要再准备与 datamodule 对齐的 `json` 目录结构。

## 3. 配置训练参数

主配置示例：`configs/tiger-large.yml`  
核心字段如下：

- `audionet`：模型结构参数（如 `out_channels`、`num_blocks`）。
- `training`：训练设置（GPU、epoch、early stop）。
- `optimizer` / `scheduler`：优化器与学习率策略。
- `datamodule`：数据模块与路径。
- `exp.exp_name`：实验名（决定输出目录名）。

### 3.1 最小必改项

1) GPU 列表：

```yaml
training:
  gpus: [0]   # 单卡示例
```

2) 数据路径（必须指向 JSON 所在目录）：

```yaml
datamodule:
  data_name: EchoSetDataModule
  data_config:
    train_dir: DataPreProcess/EchoSet/train
    valid_dir: DataPreProcess/EchoSet/val
    test_dir: DataPreProcess/EchoSet/test
```

3) 实验名：

```yaml
exp:
  exp_name: TIGER-EchoSet-Run1
```

## 4. 启动训练

```bash
python audio_train.py --conf_dir configs/tiger-large.yml
```

训练脚本会自动完成：

- 创建实验目录：`Experiments/checkpoint/<exp_name>`
- 备份本次配置：`conf.yml`
- 保存 checkpoint：`{epoch}.ckpt`、`last.ckpt`
- 导出最优模型：`best_model.pth`
- 保存 top-k 信息：`best_k_models.json`
- 记录日志到 W&B（项目名固定为 `Real-work-dataset`）

## 5. 训练完成后评估

```bash
python audio_test.py --conf_dir configs/tiger-large.yml
```

默认会从：

- `Experiments/checkpoint/<exp_name>/best_model.pth`

加载模型并在 `test` 集上跑推理，结果输出到：

- `Experiments/checkpoint/<exp_name>/results/metrics.csv`

## 6. 常见问题与排查

### 6.1 W&B 登录问题

- 首次运行前执行 `wandb login`。
- 无交互环境可用环境变量：
  - `WANDB_API_KEY=...`

### 6.2 数据路径错误/找不到 JSON

检查配置里的 `train_dir/valid_dir/test_dir` 是否都包含：

- `mix*.json`
- `s1.json`
- `s2.json`

且 JSON 里音频路径必须真实可读。

### 6.3 显存不足

优先降低：

- `data_config.batch_size`
- `audionet.audionet_config` 的通道/块数
- `training.gpus`（先单卡跑通）

### 6.4 DDP 启动异常

当前脚本默认启用了 `DDPStrategy`。如果先做单卡调试，建议先把 `gpus` 配成单卡并确认能正常跑通，再扩展到多卡。

## 7. 推荐执行顺序（实战版）

1) 准备/核对数据（必要时运行预处理脚本）  
2) 修改 `configs/tiger-large.yml`（GPU、路径、实验名）  
3) `wandb login`（仅首次）  
4) 运行训练 `audio_train.py`  
5) 训练结束运行 `audio_test.py`  
6) 检查 `best_model.pth` 与 `metrics.csv`

