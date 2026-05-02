# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

TIGER（Time-frequency Interleaved Gain Extraction and Reconstruction）是一个高效语音分离模型的训练、评测与推理仓库。支持三类训练：全精度 TIGER、BinaryTIGER 二值化训练、BinaryTIGER 蒸馏训练。

- 论文：[arXiv:2410.01469](https://arxiv.org/abs/2410.01469)
- 推荐工作流：本地验证代码/配置/数据链路，Kaggle（双 T4 GPU）正式训练

## 常用命令

```powershell
# 环境安装
pip install -r requirements.txt

# 测试
pytest                                          # 全部测试
pytest tests\test_config_contracts.py           # 单个测试文件
pytest tests\test_config_contracts.py -k test_name  # 单个测试用例

# 本地连通性验证（1 epoch, batch 1, 2s 片段）
python audio_train.py --conf_dir configs\tiger-small-local.yml --epoch 1 --batch_size 1 --segment 2.0
python audio_train.py --conf_dir configs\tiger-small-local-binary.yml --epoch 1 --batch_size 1 --segment 2.0
python audio_train.py --conf_dir configs\tiger-small-local-binary-distill-1.yml --epoch 1 --batch_size 1 --segment 2.0
python audio_train.py --conf_dir configs\tiger-small-local-binary-distill-2.yml --epoch 1 --batch_size 1 --segment 2.0
python audio_train.py --conf_dir configs\tiger-small-local-binary-distill-3.yml --epoch 1 --batch_size 1 --segment 2.0

# 正式训练（Kaggle 双 T4）
python audio_train.py --conf_dir configs\tiger-small-kaggle-t4x2.yml
python audio_train.py --conf_dir configs\tiger-small-kaggle-t4x2-binary-B0.yml
python audio_train.py --conf_dir configs\tiger-small-kaggle-t4x2-binary-distill-1.yml
python audio_train.py --conf_dir configs\tiger-small-kaggle-t4x2-binary-distill-2.yml
python audio_train.py --conf_dir configs\tiger-small-kaggle-t4x2-binary-distill-3.yml

# 阶段零：敏感度验证（S0-S4）
python audio_train.py --conf_dir configs\tiger-small-local-binary-S0.yml
python audio_train.py --conf_dir configs\tiger-small-local-binary-S1.yml
python audio_train.py --conf_dir configs\tiger-small-local-binary-S2.yml
python audio_train.py --conf_dir configs\tiger-small-local-binary-S3.yml
python audio_train.py --conf_dir configs\tiger-small-local-binary-S4.yml

# 阶段一：组合二值化（B1-B4）
python audio_train.py --conf_dir configs\tiger-small-local-binary-B1.yml
python audio_train.py --conf_dir configs\tiger-small-local-binary-B2.yml
python audio_train.py --conf_dir configs\tiger-small-local-binary-B3.yml
python audio_train.py --conf_dir configs\tiger-small-local-binary-B4.yml

# 断点续训
python audio_train.py --conf_dir configs\tiger-small-local.yml --resume
python audio_train.py --conf_dir configs\tiger-small-local.yml --resume_ckpt "Experiments\TIGER-MiniLibriMix\checkpoints\last.ckpt"

# 评测
python audio_test.py --conf_dir configs\tiger-small-local.yml

# 推理
python inference_speech.py --audio_path test\mix.wav --output_dir separated_audio
python inference_dnr.py --audio_path test\test_mixture_466.wav

# Gradio 演示
python app.py

# 数据索引生成
python DataPreProcess\process_librimix.py --in_dir "D:\Paper\datasets\MiniLibriMix" --out_dir "D:\Paper\TIGER\DataPreProcess\MiniLibriMix" --splits train val test --speakers mix_both s1 s2
python DataPreProcess\build_mini_librimix_index.py --in_dir "D:\Paper\datasets\MiniLibriMix" --out_dir "D:\Paper\TIGER\DataPreProcess\MiniLibriMix-mini"

# 实验结果对比分析
python analyze_experiments.py                          # 分析所有阶段
python analyze_experiments.py --phase stage0           # 只分析阶段零
python analyze_experiments.py --baseline-loss 1.2345   # 手动指定基线 val_loss
python analyze_experiments.py --table                  # 生成主结果表格
python analyze_experiments.py --table --output table.md  # 保存表格到文件

# 论文图表生成
python plot_figures.py                                 # 生成所有图表
python plot_figures.py --figures 1,2,4                 # 生成指定图表
python plot_figures.py --output figures/               # 指定输出目录

# 效率指标计算
python evaluated_mac_params.py                         # 计算 FLOPs/BOPs/模型大小
python evaluated_mac_params.py --measure-latency       # 测量推理延迟
python evaluated_mac_params.py --measure-latency --device cuda --num-runs 200
```

## 架构

### 核心模块 (`look2hear/`)

```
audio_train.py (入口)
    ├── configs/*.yml (声明式配置)
    ├── look2hear/datas/     数据加载：JSON 索引 -> DataLoader
    ├── look2hear/models/    模型定义
    ├── look2hear/system/    训练系统（PyTorch Lightning Modules）
    ├── look2hear/layers/    网络层（二值化层、蒸馏损失、CNN/RNN 组件）
    ├── look2hear/losses/    损失函数（PIT、MixIT、SI-SDR）
    ├── look2hear/metrics/   评测指标（SI-SDR、SDR）
    └── look2hear/utils/     工具（模型转换器、配置解析、Lightning 辅助）
```

### 模型 (`look2hear/models/`)

- `tiger.py` — `TIGER`：主模型，STFT → 子带分割 → BN → Recurrent 分离器 → 掩膜 → iSTFT。Recurrent 内部 freq_path 与 frame_path 交替处理（UConvBlock 多尺度下采样 + MultiHeadSelfAttention2D 全频帧自注意力）
- `binary_tiger.py` — `BinaryTIGER`：包装 TIGER，将符合条件的 Conv1d/Conv2d/Linear 转换为二值版本
- `tiger_dnr.py` — `TIGERDNR`：3 个 TIGER 实例（对话/音效/音乐）
- `teacher_tiger.py` — 蒸馏用教师模型加载

### 训练系统 (`look2hear/system/`)

两类训练路径共享基类 `AudioLightningModule`，区别在于：

| 训练类型 | System 类 | Model 类 | 额外配置节 |
|---------|----------|---------|-----------|
| 全精度 | `AudioLightningModule` | `TIGER` | — |
| 二值化 | `BinaryAudioLightningModule` | `BinaryTIGER` | `binary_stage_epochs` |
| 二值化+蒸馏 | `BinaryDistillAudioLitModule` | `BinaryTIGER` | `distillation` |

- `BinaryAudioLightningModule` 管理阶段转换：activation_warmup → weight_binarize → finetune，支持模块级冻结和检查点验证
- `BinaryDistillAudioLitModule` 继承 `BinaryAudioLightningModule`，集成 D1/D2/D3 蒸馏损失，支持参数分组、蒸馏预热、损失校准、余弦 lambda 调度

### 二值化工具 (`look2hear/layers/binary_layers.py`, `look2hear/utils/model_converter.py`)

- `BinaryConv1d`, `BinaryConv2d`, `BinaryLinear` — 二值化卷积/全连接层（Conv2d 覆盖 FFI 的 Q/K/V 投影和 concat_block）
- `RSign`, `RPReLU` — 二值化激活函数
- `TIGERBinaryConverter` — 将 TIGER 层转换为二值版本，支持保护策略和选择性二值化（`binarize_scope`）

### 蒸馏损失 (`look2hear/layers/kd_losses.py`)

- `SI_SNR_KDLoss` — D1: SI-SNR 输出蒸馏（PIT 感知）
- `Subband_KDLoss` — D2: 子带选择性蒸馏（低频加权）
- `Combined_KDLoss` — D3: 输出 + 子带联合蒸馏（共享 PIT 排列）

### 配置结构 (`configs/*.yml`)

每个 YAML 包含：`audionet`（模型）、`loss`、`training`、`optimizer`、`scheduler`、`datamodule`、`exp`，以及可选的 `distillation`、`binary_stage_epochs`。

关键配置项：
- `binary_config.binarize_scope`：选择性二值化的模块类别列表（`["bn", "mask", "dw", "pw", "f3a"]`）
- `training.freeze_scope`：模块级冻结（阶段零敏感度验证用）
- `training.warmup_ckpt`：共享预热 checkpoint 路径（阶段一用）
- `training.checkpoint_epoch` / `checkpoint_threshold`：检查点验证
- `training.ablation`：消融实验开关（`disable_rsign`, `skip_warmup`, `use_original_prelu`）
- `optimizer.param_groups`：参数分组（`fp32_lr`, `binary_lr`）

配置文件分类：
- 本地配置（`tiger-small-local*.yml`）：用于 smoke test，使用 `MiniLibriMix-mini` 索引
- Kaggle 配置（`*kaggle-t4x2*.yml`）：正式训练，使用完整 `MiniLibriMix` 索引
- 阶段零配置（`*S0*~S4*`）：敏感度验证，15 epochs，无 scheduler
- 阶段一配置（`*B0*`、`*B1*~B4*`）：B0 为全量二值化正式训练并提供 B1–B4 的 `warmup_ckpt`；B1–B4 为组合二值化
- 消融实验配置（`*A1*~A4*`）：消融实验开关（`disable_rsign`, `skip_warmup`, `use_original_prelu`）
- 蒸馏配置（`*distill-d1*`, `*distill-d2*`, `*distill*`）：D1/D2/D3 蒸馏训练

### 数据流

原始音频 → `DataPreProcess/` 生成 JSON 索引 → `Libri2MixModuleRemix` 读取索引 → DataLoader → 模型

索引路径：`DataPreProcess/MiniLibriMix/{train,val,test}/{mix_both,s1,s2}.json`

### 实验输出

训练产物保存在 `Experiments\<exp_name>\`，包含：`conf.yml`、`best_model.pth`、`best_k_models.json`、`checkpoints/`、`results/metrics.csv`、`history.csv`（阶段零）。

## 开发规范

- **语言**：所有 AI 回复使用简体中文，技术术语保留英文
- **代码注释**：所有注释和 docstring 使用中文，每个函数必须有中文 docstring
- **Git 提交**：Conventional Commits 格式，描述用中文，如 `fix(model): 修复二值化层梯度问题`
- **提交前**：必须通过 pytest 回归测试
- **Windows 兼容**：路径使用反斜杠 `\`，PowerShell 语法，UTF-8 编码
- **W&B**：`audio_train.py` 启动时执行 `wandb.login()`，未配置时训练可能阻塞
