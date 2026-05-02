# TIGER

TIGER 语音分离项目的训练、评测与推理仓库，当前代码主线默认使用 `MiniLibriMix` 和 `MiniLibriMix-mini` 索引。

当前推荐用法：

- 本地环境：主要用于检查代码、配置、数据链路能不能跑通
- Kaggle 环境：主要用于正式训练

仓库当前包含三类训练能力：

- 全精度 TIGER
- BinaryTIGER 二值化训练
- BinaryTIGER 蒸馏训练

BinaryTIGER 的二值化覆盖 TIGER 中符合条件的 Conv1d 和 Conv2d 层：
- Conv1d：UConvBlock（MAS）、MLP、InjectionMultiSum 等（`protect_1x1_conv: false`，关键 1x1 投影通过 `protect_patterns` 显式保护）
- Conv2d：FFI（MultiHeadSelfAttention2D）的 Q/K/V 投影和 Recurrent.concat_block 深度可分离卷积
- 保护层：BandSplit 首层投影、F³A 注意力投影、mask 输出头（通过 `protect_patterns` 配置）

## 项目结构

- `audio_train.py`：训练入口
- `audio_test.py`：评测入口
- `inference_speech.py`：语音分离推理脚本
- `inference_dnr.py`：对话 / 音效 / 音乐分离推理脚本
- `app.py`：Gradio 演示入口
- `configs\`：训练配置
- `DataPreProcess\`：数据索引生成脚本
- `look2hear\`：模型、数据模块、训练系统与工具代码
- `tests\`：回归测试
- `论文\`：技术方案文档（二值化、蒸馏、综合实验）
- `docs\tiger_original_code_mapping_zh.md`：原始 TIGER 代码结构说明

## 环境安装

建议使用 Python 3.10+ 与独立虚拟环境。

```powershell
pip install -r requirements.txt
```

如果使用 W&B 记录实验，首次需要登录：

```powershell
wandb login
```

## 数据准备

### 1. 生成完整 MiniLibriMix 索引

原始数据目录示例：

- `D:\Paper\datasets\MiniLibriMix`

生成训练 / 验证 / 测试索引：

```powershell
python DataPreProcess\process_librimix.py --in_dir "D:\Paper\datasets\MiniLibriMix" --out_dir "D:\Paper\TIGER\DataPreProcess\MiniLibriMix" --splits train val test --speakers mix_both s1 s2
```

输出目录：

- `DataPreProcess\MiniLibriMix\train\`
- `DataPreProcess\MiniLibriMix\val\`
- `DataPreProcess\MiniLibriMix\test\`

### 2. 生成 MiniLibriMix-mini 索引

用于快速检查训练链路、评测链路和配置是否可用：

```powershell
python DataPreProcess\build_mini_librimix_index.py --in_dir "D:\Paper\datasets\MiniLibriMix" --out_dir "D:\Paper\TIGER\DataPreProcess\MiniLibriMix-mini"
```

默认生成固定规模的小索引：

- `train=20`
- `val=10`
- `test=10`

## 主要配置文件

### 本地配置

本地配置主要用于 smoke test、调试和连通性验证，不作为正式训练主入口。
本地阶段零（S0-S4）默认依赖本地全精度训练产物 `Experiments\TIGER-MiniLibriMix-Local\best_model.pth`；本地蒸馏（D1-D3）默认依赖本地全精度教师 `Experiments\TIGER-MiniLibriMix-Local\best_model.pth` 与本地二值化学生 `Experiments\TIGER-Small-Binary-Local\best_model.pth`。
本地配置统一使用 `MiniLibriMix-mini` 子集，目标是快速验证代码、配置和数据链路。

- `configs\tiger-small-local.yml`：全精度 small 配置，默认使用 `DataPreProcess\MiniLibriMix-mini`
- `configs\tiger-small-local-binary.yml`：本地二值化验证配置（AdamW, 余弦退火, 80 epochs）
- `configs\tiger-small-local-binary-distill-1.yml`：本地蒸馏配置 1（D1 输出蒸馏 smoke test）
- `configs\tiger-small-local-binary-distill-2.yml`：本地蒸馏配置 2（D2 子带选择性蒸馏 smoke test）
- `configs\tiger-small-local-binary-distill-3.yml`：本地蒸馏配置 3（D3 联合蒸馏 smoke test）

### Kaggle 配置

Kaggle 配置统一使用完整 `MiniLibriMix` 数据集，目标是产出正式实验结果。

- `configs\tiger-small-kaggle-t4x2.yml`：全精度 small 正式训练
- `configs\tiger-small-kaggle-t4x2-binary-B0.yml`：阶段一 B0（与 B1–B4 同系列；详见下文「Kaggle 阶段配置 — 阶段一」）
- `configs\tiger-small-kaggle-t4x2-binary-distill-1.yml`：Kaggle 蒸馏配置 1（D1 输出蒸馏）
- `configs\tiger-small-kaggle-t4x2-binary-distill-2.yml`：Kaggle 蒸馏配置 2（D2 子带选择性蒸馏）
- `configs\tiger-small-kaggle-t4x2-binary-distill-3.yml`：Kaggle 蒸馏配置 3（D3 联合蒸馏）

### 本地阶段配置

#### 阶段零配置（本地敏感度验证）

用于独立测试每个模块二值化后的性能退化程度：

- `configs\tiger-small-local-binary-S0.yml`：BN 单独二值化
- `configs\tiger-small-local-binary-S1.yml`：mask 单独二值化
- `configs\tiger-small-local-binary-S2.yml`：DW 单独二值化
- `configs\tiger-small-local-binary-S3.yml`：PW 单独二值化
- `configs\tiger-small-local-binary-S4.yml`：F3A 单独二值化

#### 阶段一配置（本地组合二值化）

从共享预热 checkpoint 出发，组合不同模块进行二值化训练：

- `configs\tiger-small-local-binary-B1.yml`：BN + mask
- `configs\tiger-small-local-binary-B2.yml`：BN + mask + DW
- `configs\tiger-small-local-binary-B3.yml`：BN + mask + DW + PW
- `configs\tiger-small-local-binary-B4.yml`：全二值化（含 F3A）

运行 B1–B4 前需先完成一次本地二值化预热（`configs\tiger-small-local-binary.yml`），默认产物目录为 `Experiments\TIGER-Small-Binary-Local\`；阶段一配置中的 `warmup_ckpt` 指向其中的 `checkpoints\best.ckpt`。

#### 消融实验配置（本地）

- `configs\tiger-small-local-binary-A1.yml`：有/无 RSign
- `configs\tiger-small-local-binary-A2.yml`：有/无 Step1 预热
- `configs\tiger-small-local-binary-A3.yml`：RPReLU vs PReLU
- `configs\tiger-small-local-binary-A4.yml`：D2 选择性子带权重 vs 统一子带权重

### Kaggle 阶段配置

#### 阶段零配置（Kaggle 敏感度验证）

- `configs\tiger-small-kaggle-t4x2-binary-S0.yml`：BN 单独二值化
- `configs\tiger-small-kaggle-t4x2-binary-S1.yml`：mask 单独二值化
- `configs\tiger-small-kaggle-t4x2-binary-S2.yml`：DW 单独二值化
- `configs\tiger-small-kaggle-t4x2-binary-S3.yml`：PW 单独二值化
- `configs\tiger-small-kaggle-t4x2-binary-S4.yml`：F3A 单独二值化

#### 阶段一配置（Kaggle 组合二值化）

- `configs\tiger-small-kaggle-t4x2-binary-B0.yml`：B0，共享 activation warmup + 全量二值化正式训练（B1–B4 的 `warmup_ckpt` 默认指向本实验的 `checkpoints/best.ckpt`）
- `configs\tiger-small-kaggle-t4x2-binary-B1.yml`：BN + mask
- `configs\tiger-small-kaggle-t4x2-binary-B2.yml`：BN + mask + DW
- `configs\tiger-small-kaggle-t4x2-binary-B3.yml`：BN + mask + DW + PW
- `configs\tiger-small-kaggle-t4x2-binary-B4.yml`：全二值化（含 F3A）

#### 阶段三配置（Kaggle 消融实验）

- `configs\tiger-small-kaggle-t4x2-binary-A1.yml`：有/无 RSign
- `configs\tiger-small-kaggle-t4x2-binary-A2.yml`：有/无 Step1 预热
- `configs\tiger-small-kaggle-t4x2-binary-A3.yml`：RPReLU vs PReLU
- `configs\tiger-small-kaggle-t4x2-binary-A4.yml`：D2 选择性子带权重 vs 统一子带权重

## 本地验证

本地环境建议只做快速验证，确认数据索引、模型构建、训练入口和评测入口都能正常运行。

### 全精度连通性验证

```powershell
python audio_train.py --conf_dir configs\tiger-small-local.yml --epoch 1 --batch_size 1 --segment 2.0
```

### 二值化连通性验证

```powershell
python audio_train.py --conf_dir configs\tiger-small-local-binary.yml --epoch 1 --batch_size 1 --segment 2.0
```

### 蒸馏连通性验证

```powershell
python audio_train.py --conf_dir configs\tiger-small-local-binary-distill-3.yml --epoch 1 --batch_size 1 --segment 2.0
```

`configs\tiger-small-local-binary-distill-3.yml` 默认依赖以下权重：

- `distillation.teacher_ckpt`
- `distillation.student_init_ckpt`

开始本地蒸馏验证前，应先确认这两个路径对应的 `best_model.pth` 已存在。

如果要分别验证 D1 / D2，可直接使用：

```powershell
python audio_train.py --conf_dir configs\tiger-small-local-binary-distill-1.yml --epoch 1 --batch_size 1 --segment 2.0
python audio_train.py --conf_dir configs\tiger-small-local-binary-distill-2.yml --epoch 1 --batch_size 1 --segment 2.0
```

### 断点续训

恢复当前实验目录下的 `last.ckpt`：

```powershell
python audio_train.py --conf_dir configs\tiger-small-local.yml --resume
```

显式指定 checkpoint：

```powershell
python audio_train.py --conf_dir configs\tiger-small-local.yml --resume_ckpt "Experiments\TIGER-MiniLibriMix\checkpoints\last.ckpt"
```

## 本地评测

```powershell
python audio_test.py --conf_dir configs\tiger-small-local.yml
```

评测默认读取当前实验目录下导出的 `best_model.pth`，并将结果写入：

- `Experiments\<exp_name>\results\metrics.csv`

## Kaggle 配置使用

正式训练建议统一在 Kaggle 上进行（双 T4 GPU）。

Kaggle 配置默认假设：

- 原始数据集挂载在 `/kaggle/input/<your-dataset-name>`
- 预处理后索引输出到 `/kaggle/working/DataPreProcess/...`
- 仓库代码位于 `/kaggle/working/TIGER`

### 1. 生成 Kaggle 用索引

如果你在 Kaggle Notebook 中挂载的是完整 `MiniLibriMix` 数据集，先生成 JSON 索引：

```bash
python DataPreProcess/process_librimix.py --in_dir /kaggle/input/MiniLibriMix --out_dir /kaggle/working/DataPreProcess/MiniLibriMix --splits train val test --speakers mix_both s1 s2
```

如果你的 Kaggle Dataset 名称不是 `MiniLibriMix`，把 `--in_dir` 改成实际挂载路径。

如果只想做快速连通性验证，可以生成小索引：

```bash
python DataPreProcess/build_mini_librimix_index.py --in_dir /kaggle/input/MiniLibriMix --out_dir /kaggle/working/DataPreProcess/MiniLibriMix-mini
```

### 2. 四阶段实验流程总览

综合实验方案定义了四阶段实验流程，每个阶段的产出是下一阶段的输入：

```
阶段零（敏感度验证）→ 阶段一（组合二值化）→ 阶段二（蒸馏补偿）→ 阶段三（消融实验）
    S0-S4                 B0+B1-B4               D0-D3                  A1-A4
```

**依赖关系：**

| 阶段 | 依赖 | 产出 |
|------|------|------|
| 阶段零 S0-S4 | 全精度 checkpoint | 各模块敏感度报告（history.csv） |
| 阶段一 B0 + B1–B4 | B0：`warmup_ckpt` 来自 FP32；B1–B4：`warmup_ckpt` 来自 B0 + FP32 基线 val_loss | B0 全量二值化与各组合二值化模型 |
| 阶段二 D0-D3 | 阶段一最优 checkpoint + 全精度教师模型 | 蒸馏后的二值化模型 |
| 阶段三 A1-A4 | 阶段一/阶段二 checkpoint | 消融实验对比数据 |

### 3. Kaggle 各阶段详细训练命令

#### 前置步骤：全精度训练

所有阶段都依赖全精度训练的产出（教师模型 + 基线 val_loss）：

```bash
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2.yml
```

产出路径：`Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2/best_model.pth`

#### 阶段零：敏感度验证（S0-S4）

**目的：** 独立测试每个模块二值化后的性能退化程度，为阶段一的组合策略提供依据。

**在 Kaggle 上运行：**

```bash
# S0: BN 单独二值化
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-S0.yml

# S1: mask 单独二值化
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-S1.yml

# S2: DW 单独二值化
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-S2.yml

# S3: PW 单独二值化
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-S3.yml

# S4: F3A 单独二值化
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-S4.yml
```

**产出：** 各实验目录与对应 YAML 中 `exp.exp_name` 一致（Kaggle 阶段零目录名均带 `-Kaggle-T4x2` 后缀），例如：

- `Experiments/TIGER-Small-Binary-S0-BN-Kaggle-T4x2/history.csv`
- `Experiments/TIGER-Small-Binary-S1-Mask-Kaggle-T4x2/history.csv`
- ……S2–S4 同理（`S2-DW`、`S3-PW`、`S4-F3A` + `-Kaggle-T4x2`）

**分析：** 比较各实验的 `val_loss` 上升幅度，确定哪些模块对二值化敏感。

#### 阶段一：组合二值化（B0 + B1–B4）

**目的：** B0 完成共享 activation warmup 与全量二值化正式训练；B1–B4 基于阶段零结论，组合不同模块进行二值化训练。

**前置步骤：** 使用 `configs/tiger-small-kaggle-t4x2-binary-B0.yml` 完成与阶段一相同的 20 epoch `activation_warmup`（仅训练 RSign/RPReLU）及后续 `weight_binarize`。默认 Lightning checkpoint 写入：

`Experiments/TIGER-Small-Binary-B0-Kaggle-T4x2/checkpoints/best.ckpt`

该路径与 B1–B4 配置中的 `training.warmup_ckpt` 一致。若你改用其他目录名存放预热产物，请同步修改各 B 配置里的 `warmup_ckpt`。

**在 Kaggle 上运行：**

```bash
# B0: 共享 activation warmup + 全量二值化正式训练（B1–B4 的 warmup_ckpt 来源）
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-B0.yml

# B1: BN + mask（阈值 10%）
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-B1.yml

# B2: BN + mask + DW（阈值 15%）
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-B2.yml

# B3: BN + mask + DW + PW（阈值 20%）
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-B3.yml

# B4: 全二值化含 F3A（阈值 25%）
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-B4.yml
```

**注意：** 阶段一配置中的 `fp32_baseline_val_loss` 需要填入全精度训练的 `val_loss` 值；默认的 `0.0` 只是占位，未填写时 30 epoch 检查点验证不会生效。

**产出：** B0 与 B1–B4 的模型均保存在与 `exp.exp_name` 一致的目录，例如 `Experiments/TIGER-Small-Binary-B0-Kaggle-T4x2/`、`Experiments/TIGER-Small-Binary-B1-BN-Mask-Kaggle-T4x2/`、…、`Experiments/TIGER-Small-Binary-B4-Full-Kaggle-T4x2/`

#### 阶段二：蒸馏补偿（D0-D3）

**目的：** 用知识蒸馏弥补二值化带来的精度损失。

**前置步骤：** 需要全精度教师模型和阶段一最优二值化模型。

**在 Kaggle 上运行：**

```bash
# D0: 无蒸馏基线（工程上直接复用阶段一最优二值化模型 B*，不单独再跑一个 distill YAML）

# D1: SI-SNR 输出蒸馏
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-distill-1.yml

# D2: 子带选择性蒸馏
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-distill-2.yml

# D3: 输出 + 子带联合蒸馏
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-distill-3.yml
```

**产出：** D1/D2/D3 分别对应 `Experiments/TIGER-Small-Binary-Distill-D1-Kaggle-T4x2/`、`Experiments/TIGER-Small-Binary-Distill-D2-Kaggle-T4x2/`、`Experiments/TIGER-Small-Binary-Distill-Kaggle-T4x2/`（与各自蒸馏 YAML 的 `exp.exp_name` 一致）

#### 阶段三：消融实验（A1-A4）

**目的：** 验证各设计选择的贡献。

**在 Kaggle 上运行：**

```bash
# A1: 有/无 RSign
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-A1.yml

# A2: 有/无 Step1 预热
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-A2.yml

# A3: RPReLU vs PReLU
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-A3.yml

# A4: D2 选择性子带权重 vs 统一子带权重
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-A4.yml
```

### 4. Kaggle 推荐执行顺序

如果你要在 Kaggle 上完整复现实验链路，建议严格按“训练 -> 评测 -> 分析 -> 进入下一阶段”的顺序执行。下面的顺序更贴合真实实验流程。

**Step 1：生成 Kaggle 用索引**

```bash
python DataPreProcess/process_librimix.py --in_dir /kaggle/input/MiniLibriMix --out_dir /kaggle/working/DataPreProcess/MiniLibriMix --splits train val test --speakers mix_both s1 s2
```

**Step 2：训练 FP32 基线**

```bash
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2.yml
```

**Step 3：评测 FP32 基线**

```bash
python audio_test.py --conf_dir configs/tiger-small-kaggle-t4x2.yml
```

这一步会补齐基线实验目录中的 `results\metrics.csv` 与 `final_metrics.json` 测试指标，后续阶段都默认依赖这个基线目录：

- `Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2/`

**Step 4：阶段零 S0-S4 敏感度实验**

S0-S4 都依赖 FP32 基线权重，可以分开跑，也可以并行跑：

```bash
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-S0.yml
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-S1.yml
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-S2.yml
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-S3.yml
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-S4.yml
```

阶段零训练结束后，先做阶段分析，再决定阶段一重点保留哪些组合：

```bash
python analyze_experiments.py --phase stage0 --baseline /kaggle/working/Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2
python plot_figures.py --stage-summary stage0 --output paper_figures
```

**Step 5：阶段一 B0 + B1-B4 组合二值化**

先跑 B0。B1-B4 的 `training.warmup_ckpt` 默认指向 B0 的 `checkpoints/best.ckpt`，所以 B0 是阶段一入口，不应跳过。

```bash
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-B0.yml
```

然后再跑 B1-B4，四个实验可以并行：

```bash
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-B1.yml
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-B2.yml
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-B3.yml
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-B4.yml
```

阶段一训练完成后，建议至少对准备进入后续主线的候选模型做评测；通常至少包含 `B0` 和你打算作为 `B*` 候选的几个组合。最直接的做法是逐个改对应 YAML 的 `exp.exp_name` 后运行 `audio_test.py`：

```bash
python audio_test.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-B0.yml
python audio_test.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-B1.yml
python audio_test.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-B2.yml
python audio_test.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-B3.yml
python audio_test.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-B4.yml
```

然后再分析阶段一结果，确定后续蒸馏和消融默认使用的 `B*`：

```bash
python analyze_experiments.py --phase stage1 --baseline /kaggle/working/Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2
python plot_figures.py --stage-summary stage1 --output paper_figures
```

**Step 6：阶段二 D1-D3 蒸馏补偿**

蒸馏前先确认：

- `distillation.teacher_ckpt` 指向 FP32 基线的 `best_model.pth`
- `distillation.student_init_ckpt` 指向你选定的阶段一二值模型；仓库默认写的是 `B0`

随后运行：

```bash
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-distill-1.yml
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-distill-2.yml
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-distill-3.yml
```

训练完成后评测 D1-D3：

```bash
python audio_test.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-distill-1.yml
python audio_test.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-distill-2.yml
python audio_test.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-distill-3.yml
```

再做阶段分析：

```bash
python analyze_experiments.py --phase stage2 --baseline /kaggle/working/Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2
python plot_figures.py --stage-summary stage2 --output paper_figures
```

**Step 7：阶段三 A1-A4 消融实验**

```bash
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-A1.yml
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-A2.yml
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-A3.yml
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-A4.yml
```

训练完成后评测 A1-A4：

```bash
python audio_test.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-A1.yml
python audio_test.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-A2.yml
python audio_test.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-A3.yml
python audio_test.py --conf_dir configs/tiger-small-kaggle-t4x2-binary-A4.yml
```

再做阶段分析：

```bash
python analyze_experiments.py --phase stage3 --baseline /kaggle/working/Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2
python plot_figures.py --stage-summary stage3 --output paper_figures
```

**Step 8：全部阶段完成后统一汇总**

最后再统一生成总分析、论文图表和效率指标，不要穿插到中间阶段：

```bash
python analyze_experiments.py --baseline /kaggle/working/Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2
python analyze_experiments.py --table
python analyze_experiments.py --table --output results/main_table.md
python plot_figures.py --figures all --output paper_figures
python evaluated_mac_params.py --config configs/tiger-small-kaggle-t4x2.yml
```

### 5. 蒸馏配置的额外要求

`configs/tiger-small-kaggle-t4x2-binary-distill-1.yml`、`configs/tiger-small-kaggle-t4x2-binary-distill-2.yml`、`configs/tiger-small-kaggle-t4x2-binary-distill-3.yml` 默认依赖两个已有权重：

- `distillation.teacher_ckpt`: 全精度训练产出的 `best_model.pth`
- `distillation.student_init_ckpt`: 二值化学生初始化权重（默认指向 `Experiments/TIGER-Small-Binary-B0-Kaggle-T4x2/best_model.pth`，即 `tiger-small-kaggle-t4x2-binary-B0.yml` 正式二值化训练的产物；若你的阶段一最优模型在其他目录，请改成对应路径）

这意味着蒸馏前至少要先完成：

- 一次全精度 Kaggle 训练
- 一次可提供 `student_init_ckpt` 的二值化训练（默认与 `tiger-small-kaggle-t4x2-binary-B0.yml` 一致）

如果你的 Notebook 工作目录不是 `/kaggle/working/TIGER`，或者实验名被你改过，需要同步修改这两个路径。

### 6. 各阶段产出路径汇总

| 阶段 | 实验 | 产出路径 |
|------|------|---------|
| 前置 | 全精度 | `Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2/` |
| 阶段零 | S0 | `Experiments/TIGER-Small-Binary-S0-BN-Kaggle-T4x2/` |
| 阶段零 | S1 | `Experiments/TIGER-Small-Binary-S1-Mask-Kaggle-T4x2/` |
| 阶段零 | S2 | `Experiments/TIGER-Small-Binary-S2-DW-Kaggle-T4x2/` |
| 阶段零 | S3 | `Experiments/TIGER-Small-Binary-S3-PW-Kaggle-T4x2/` |
| 阶段零 | S4 | `Experiments/TIGER-Small-Binary-S4-F3A-Kaggle-T4x2/` |
| 阶段一 | B0 | `Experiments/TIGER-Small-Binary-B0-Kaggle-T4x2/` |
| 阶段一 | B1 | `Experiments/TIGER-Small-Binary-B1-BN-Mask-Kaggle-T4x2/` |
| 阶段一 | B2 | `Experiments/TIGER-Small-Binary-B2-BN-Mask-DW-Kaggle-T4x2/` |
| 阶段一 | B3 | `Experiments/TIGER-Small-Binary-B3-BN-Mask-DW-PW-Kaggle-T4x2/` |
| 阶段一 | B4 | `Experiments/TIGER-Small-Binary-B4-Full-Kaggle-T4x2/` |
| 阶段二 | D1 | `Experiments/TIGER-Small-Binary-Distill-D1-Kaggle-T4x2/` |
| 阶段二 | D2 | `Experiments/TIGER-Small-Binary-Distill-D2-Kaggle-T4x2/` |
| 阶段二 | D3 | `Experiments/TIGER-Small-Binary-Distill-Kaggle-T4x2/` |
| 阶段三 | A1 | `Experiments/TIGER-Small-Binary-A1-NoRSign-Kaggle-T4x2/` |
| 阶段三 | A2 | `Experiments/TIGER-Small-Binary-A2-NoWarmup-Kaggle-T4x2/` |
| 阶段三 | A3 | `Experiments/TIGER-Small-Binary-A3-PReLU-Kaggle-T4x2/` |
| 阶段三 | A4 | `Experiments/TIGER-Small-Binary-A4-NoSubbandWeight-Kaggle-T4x2/` |

### 9. 实验顺序里的结果分析

`analyze_experiments.py` 不是独立工具说明，它接在每个阶段训练后面，用来判断这一阶段的结果是否足够进入下一阶段。

#### 阶段零完成后：看敏感度，决定阶段一组合

```bash
python analyze_experiments.py --phase stage0 --baseline /kaggle/working/Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2
```

这一步主要看：

- `vs基线`：相对 FP32 的 `val_loss` 上升百分比
- S0-S4 哪些模块更鲁棒，哪些更敏感

它的作用是为阶段一提供依据，决定 B1-B4 哪些组合值得重点保留。

#### 阶段一完成后：看组合二值化结果，确定 B*

```bash
python analyze_experiments.py --phase stage1 --baseline /kaggle/working/Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2
```

这一步主要看：

- B0 全量二值化和 B1-B4 组合二值化谁更稳
- 哪个组合在阈值内最好

它的作用是选出阶段二和阶段三默认继续使用的二值化基线 `B*`。

#### 阶段二完成后：看蒸馏补偿是否有效

```bash
python analyze_experiments.py --phase stage2 --baseline /kaggle/working/Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2
```

这一步主要看：

- D0 / D1 / D2 / D3 的 `best_val_loss`
- 蒸馏后相对 `B*` 是否有恢复

它的作用是判断哪种蒸馏方案最值得写入主结果。

#### 阶段三完成后：看消融结论是否成立

```bash
python analyze_experiments.py --phase stage3 --baseline /kaggle/working/Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2
```

这一步主要看：

- A1-A4 对主线设计的影响
- 哪些设计选择是必要的，哪些只是次要收益

它的作用是支撑论文里的设计解释，而不是继续选下一阶段输入。

### 10. 全部实验完成后的总汇总

当前面四个阶段都完成以后，再统一做总汇总和论文主表：

```bash
# 汇总所有阶段
python analyze_experiments.py --baseline /kaggle/working/Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2

# 生成主结果表（Markdown，可直接整理进论文）
python analyze_experiments.py --table
python analyze_experiments.py --table --output results/main_table.md
```

这些输出分别用于：

1. **终端表格**：快速检查每个阶段的实验名、状态、`best_val_loss`、相对基线上升百分比、`final_val_loss`、训练 epoch 数
2. **`experiment_summary.json`**：给后续脚本和绘图提供统一的结构化数据
3. **主结果表格**：优先读取 `final_metrics.json` 中已有的 `SI-SNRi / SDR / 模型大小`；如果还没补齐 `PESQ / STOI / BOPs`，对应列先保留为 `—`

如果实验目录不在默认路径，可以改用：

```bash
python analyze_experiments.py --baseline-loss 1.2345
```

### 11. 全部实验完成后的论文图表

`plot_figures.py` 服务的是论文和展示，不负责决定训练是否继续。建议在四个阶段全部跑完后统一生成：

```bash
python plot_figures.py --figures all --output paper_figures
```

各图和实验阶段的对应关系如下：

| 图表 | 关联阶段 | 用途 |
|------|---------|------|
| 图 1 | 阶段零 | 模块敏感度柱状图：看 S0-S4 哪些模块更适合二值化 |
| 图 2 | 阶段一 | 精度-压缩权衡曲线：看 B1/B2/B3 与 FP32、B* 的关系 |
| 图 3 | 阶段一 | 训练动态与检查点：看组合二值化在训练过程中的稳定性 |
| 图 4 | 阶段二 | 蒸馏效果对比：看 D0/D1/D2/D3 的补偿幅度 |
| 图 5 | 阶段二 | 蒸馏训练动态：看 D0-D3 的 `val_loss` 曲线 |
| 图 6 | 阶段三 | 消融实验雷达图：看 A1-A4 对主线设计的影响 |
| 图 7 | 全流程汇总 | 推理效率帕累托前沿：把最终精度和效率放在一起看 |

图表数据自动从各实验目录的 `final_metrics.json` 和 `history.csv` 读取。若某张图缺少真实实验数据，脚本会回退到《综合实验方案》中的预期数据，并在图标题中显式标注“含预期数据回退”。

### 12. 全部实验完成后的效率指标补充

`evaluated_mac_params.py` 也不是用来决定阶段流转的，而是在最终结果确定后补论文表格和效率图。

```bash
# 计算 FLOPs/BOPs/参数量/模型大小
python evaluated_mac_params.py --config configs/tiger-small-kaggle-t4x2.yml

# 测量推理延迟（CPU）
python evaluated_mac_params.py --config configs/tiger-small-kaggle-t4x2.yml --measure-latency

# 测量推理延迟（GPU）
python evaluated_mac_params.py --config configs/tiger-small-kaggle-t4x2.yml --measure-latency --device cuda

# 自定义测量参数
python evaluated_mac_params.py --config configs/tiger-small-kaggle-t4x2.yml --measure-latency --device cpu --num-runs 200 --input-length 32000
```

这些指标最后用于：

- `fp32_flops`：给 FP32 基线的算力量级
- `binary_bops`：给二值化模型的位运算量
- `equivalent_flops`：把 BOPs 折算成便于比较的浮点口径
- `model_size_mb`：补到主结果表和压缩对比图
- `mean_ms / std_ms / min_ms / max_ms`：补到最终部署或推理效率结论里

## 推理

### 语音分离

```powershell
python inference_speech.py --audio_path test\mix.wav --output_dir separated_audio
```

### DnR 分离

```powershell
python inference_dnr.py --audio_path test\test_mixture_466.wav
```

## Gradio 演示

```powershell
python app.py
```

该入口会加载 Hugging Face 上的预训练模型，并提供：

- 音频语音分离
- 音频 DnR 分离
- 视频语音分离
- 视频 DnR 分离

## 实验输出

训练产物默认保存在：

- `Experiments\<exp_name>\`

常见文件包括：

- `conf.yml`
- `wandb_run.json`
- `best_model.pth`
- `best_k_models.json`
- `checkpoints\best.ckpt`
- `checkpoints\last.ckpt`
- `results\metrics.csv`
- `history.csv`（每 epoch 的 train_loss、val_loss、lr 记录）
- `final_metrics.json`（训练摘要；运行 `audio_test.py` 后会补充 `SI-SNRi / SDR` 等测试指标）

## 测试

```powershell
pytest
```

如果只想先检查关键配置约束：

```powershell
pytest tests\test_config_contracts.py
```

## 注意事项

- `audio_train.py` 在启动时会执行 `wandb.login()`；未配置 W&B 时训练可能被阻塞。
- 本地配置统一使用 `tiger-small-local-*` 命名；Kaggle 配置统一使用 `tiger-small-kaggle-t4x2-*` 命名。
- `configs\tiger-small-local-binary-distill-1.yml`、`configs\tiger-small-local-binary-distill-2.yml`、`configs\tiger-small-local-binary-distill-3.yml` 默认 `teacher_ckpt` / `student_init_ckpt` 指向已有实验输出，换环境后通常需要同步修改。
- `configs\tiger-small-local-binary-A4.yml` 现在是论文中的 A4 消融：保留 D2 子带蒸馏，仅把五段子带权重统一为 `1.0`。
- `inference_speech.py` 和 `app.py` 依赖联网下载预训练模型缓存。

## 论文

- [TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation](https://arxiv.org/abs/2410.01469)

```bibtex
@article{xu2024tiger,
  title={TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation},
  author={Xu, Mohan and Li, Kai and Chen, Guo and Hu, Xiaolin},
  journal={arXiv preprint arXiv:2410.01469},
  year={2024}
}
```
