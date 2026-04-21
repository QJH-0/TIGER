# TIGER（MiniLibriMix 默认版）

本仓库用于语音分离模型 TIGER 的训练、评估与推理。  
当前默认数据流程已切换为 **MiniLibriMix**。

## 1. 项目文档入口

- 项目树形结构说明：`docs/project_tree_zh.md`
- 二值化与蒸馏技术方案：`docs/binary_distill_mapping_zh.md`

## 2. 环境安装

```bash
git clone https://github.com/JusperLee/TIGER.git && cd TIGER && pip install -r requirements.txt
```

上面这一步是**首次安装依赖**：在同一个 Python 环境下后续训练/评估无需重复执行；只有在你修改了 `requirements.txt` 或切换到新的虚拟环境时，才需要重新安装。  
W&B 首次使用需要登录一次：

```bash
wandb login
```

W&B 命名约定：

- `project` 固定为 `tiger-speech-separation`
- `run name` 自动生成为 `<model>-<dataset>-bs<batch>-seg<segment>-<exp_name>`
- 训练期指标统一按 `train/*`、`val/*` 分组
- 当前默认上报的核心指标为 `train/loss`、`train/learning_rate`、`val/loss`、`val/si_snr`
- W&B 横轴统一绑定到 `epoch`
- 上传到 W&B 的配置会被展平成单层 key，并统一使用下划线，例如 `training_epochs`、`optimizer_lr`、`datamodule_data_config_batch_size`

说明：

- 训练阶段只使用验证集 `val` 做早停和 checkpoint 监控
- `test` 集不再参与 `trainer.fit()` 期间的进度条和 W&B 日志
- 最终测试统一通过 `python audio_test.py --conf_dir ...` 单独执行
- 如果 W&B 项目里已经存在旧的自动面板，需要在项目页面把 Workspace 从 `Automatic` 切到 `Manual`
- Manual Workspace 里只保留这 4 张训练期图：`train/loss`、`train/learning_rate`、`val/loss`、`val/si_snr`
- 若旧项目里某些图仍显示 `trainer/global_step`，在面板编辑页把 X-Axis 改成 `epoch` 后保存；新 run 会优先按代码里的 `epoch` 定义生成

示例：

```text
project  = tiger-speech-separation
run name = tiger-libri2mixmoduleremix-bs4-seg3.0-tiger-minilibrimix
```

## 3. 默认数据与索引生成（MiniLibriMix）

默认原始数据目录：

- `D:\Paper\datasets\MiniLibriMix`

生成训练索引（JSON）到项目内：

```bash
python DataPreProcess/process_librimix.py --in_dir "D:/Paper/datasets/MiniLibriMix" --out_dir "D:/Paper/TIGER/DataPreProcess/MiniLibriMix" --splits train val test --speakers mix_both s1 s2
```

生成完成后目录应为：

- `DataPreProcess/MiniLibriMix/train/{mix_both.json,s1.json,s2.json}`
- `DataPreProcess/MiniLibriMix/val/{mix_both.json,s1.json,s2.json}`
- `DataPreProcess/MiniLibriMix/test/{mix_both.json,s1.json,s2.json}`

如果你只想快速验证训练和测试流程是否正常，可以生成一个更小的冒烟测试索引：

```bash
python DataPreProcess/build_mini_librimix_index.py --in_dir "D:/Paper/datasets/MiniLibriMix" --out_dir "D:/Paper/TIGER/DataPreProcess/MiniLibriMix-mini"
```

该脚本会扫描同一份原始 MiniLibriMix 目录，并生成固定大小的小索引：

- `train=20`
- `val=10`
- `test=10`

输出目录结构为：

- `DataPreProcess/MiniLibriMix-mini/train/{mix_both.json,s1.json,s2.json}`
- `DataPreProcess/MiniLibriMix-mini/val/{mix_both.json,s1.json,s2.json}`
- `DataPreProcess/MiniLibriMix-mini/test/{mix_both.json,s1.json,s2.json}`

JSON 格式与全量索引一致。要使用这套小索引，只需要把训练配置里的 `train_dir`、`valid_dir`、`test_dir` 改成 `DataPreProcess/MiniLibriMix-mini/...` 对应路径。

## 4. Kaggle T4x2 训练流程（MiniLibriMix）

如果你在 Kaggle 的 `T4 x2` GPU 环境上训练，先将原始 MiniLibriMix 数据集上传为 Kaggle Dataset。下面假设挂载路径为：

- `/kaggle/input/MiniLibriMix`

先在 Kaggle Notebook 中生成训练所需的 JSON 索引：

```bash
python DataPreProcess/process_librimix.py --in_dir /kaggle/input/MiniLibriMix --out_dir /kaggle/working/DataPreProcess/MiniLibriMix --splits train val test --speakers mix_both s1 s2
```

生成完成后，使用双 T4 配置开始训练：

```bash
python audio_train.py --conf_dir configs/tiger-large-kaggle-t4x2.yml
```

如果你要在 Kaggle 上验证二值化与蒸馏链路，配置文件为：

- `configs/tiger-small-kaggle-t4x2-binary.yml`
- `configs/tiger-small-kaggle-t4x2-distill.yml`

推荐的 Kaggle 实验顺序为：

1. 先跑全精度基线：

```bash
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2.yml
```

2. 再跑二值化训练：

```bash
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-binary.yml
```

3. 待 `/kaggle/working/Experiments/TIGER-Small-Binary-Kaggle-T4x2/best_model.pth` 生成后，再跑独立蒸馏训练：

```bash
python audio_train.py --conf_dir configs/tiger-small-kaggle-t4x2-distill.yml
```

其中 Kaggle 蒸馏配置默认使用：

- `teacher_ckpt: /kaggle/working/Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2/best_model.pth`
- `student_init_ckpt: /kaggle/working/Experiments/TIGER-Small-Binary-Kaggle-T4x2/best_model.pth`

说明：

- `teacher_ckpt` 需要是已训练好的全精度 TIGER 权重
- `student_init_ckpt` 需要先完成二值化训练后才会生成
- 如果这两个文件路径与你的 Notebook 输出目录不一致，需要同步修改 `configs/tiger-small-kaggle-t4x2-distill.yml`

当前项目的文档约定如下：

- 二值化与蒸馏方案以 `docs/binary_distill_mapping_zh.md` 为准
- 训练链路保持分离：先二值化训练，再独立蒸馏训练
- teacher 固定为全精度 `TIGER`
- student 固定为完成二值化训练后的 `BinaryTIGER`
- 蒸馏阶段不与二值化主训练混合执行

方案设计要点：

- 二值化改造以 `nn.Conv1d` 为统一目标，便于对 TIGER 的编码器、FFI、MSA、频带恢复等模块做低侵入替换
- 默认保护敏感层，包括 `BandSplit` 首层投影、注意力 `Q/K/V` 投影、输出 `mask` 层和恢复路径关键层
- 蒸馏采用 `DISPATCH` 选择性蒸馏，只在 teacher 明显优于 student 的时频 patch 上传递知识
- 二值化阶段与蒸馏阶段分别维护独立配置，便于复现实验与做消融分析

当前实现已对齐的工程细节：

- `BinaryConv1d` 使用 STE 做权重二值化，`RSign` 使用可学习阈值 `alpha`
- `RPReLU` 使用 `beta / gamma / zeta` 三组参数做分布重塑
- 转换器已支持 `protect_patterns`，默认可保护 `bandsplit.proj`、`q_proj`、`k_proj`、`v_proj`、`mask_gen`、`istft`
- 二值训练系统已兼容三阶段配置键：`activation_warmup -> weight_binarize -> finetune`
- 蒸馏训练会分别记录 `train/task_loss`、`train/kd_loss`、`train/loss`

该配置文件已预设：

- `gpus: [0, 1]`
- `batch_size: 4`
- `segment: 3.0`
- `num_workers: 4`
- `train_dir: /kaggle/working/DataPreProcess/MiniLibriMix/train`
- `valid_dir: /kaggle/working/DataPreProcess/MiniLibriMix/val`
- `test_dir: /kaggle/working/DataPreProcess/MiniLibriMix/test`

如果你的 Kaggle Dataset 名称不是 `MiniLibriMix`，把 `--in_dir` 改成实际挂载路径即可。

## 5. 默认训练配置

默认配置文件：`configs/tiger-large.yml`

已配置为：

- `data_name: Libri2MixModuleRemix`
- `train_dir: DataPreProcess/MiniLibriMix/train`
- `valid_dir: DataPreProcess/MiniLibriMix/val`
- `test_dir: DataPreProcess/MiniLibriMix/test`

本地二值化/蒸馏配置文件为：

- `configs/tiger-small-binary.yml`
- `configs/tiger-small-distill.yml`

这两份文件都参考 `configs/tiger-small.yml` 派生，并做了以下本地化调整：

- 数据路径改为 `DataPreProcess/MiniLibriMix-mini/...`
- `gpus: [0]`
- `num_workers: 0`
- 二值化配置使用 `BinaryTIGER + BinaryAudioLightningModule`
- 蒸馏配置使用 `BinaryTIGER + DistillAudioLightningModule`
- 蒸馏配置默认 `teacher_ckpt` 为 `Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2/best_model.pth`
- 蒸馏配置默认 `student_init_ckpt` 为 `Experiments/TIGER-MiniLibriMix-Binary/best_model.pth`
- 二值阶段配置键使用 `training.binary_stage_epochs.activation_warmup` 与 `training.binary_stage_epochs.weight_binarize`
- `binary_config` 中已显式提供 `protect_patterns`

说明：在跑 `configs/tiger-small-distill.yml` 之前，需要先跑 `configs/tiger-small-binary.yml`，否则本地 `student_init_ckpt` 不存在。

文档层面的训练约束与 Kaggle 版一致：

- 二值化训练与蒸馏训练严格分离
- 蒸馏配置只负责加载 teacher 与二值 student，不负责二值阶段切换
- 后续代码改造应围绕方案文档中的保护策略、`BinaryConv1d` 替换机制和 `DISPATCH` 蒸馏流程展开

当前本地二值配置的关键字段示例：

```yaml
binary_config:
  protect_first_layer: true
  protect_attention: true
  # 注意：注意力保护采用“严格匹配”，避免把仅命名包含 `att` 的非注意力分支（如 `globalatt` MLP）
  # 误判为注意力模块而被保护，导致二值化覆盖率下降。
  protect_1x1_conv: true
  protect_output_layer: true
  enable_binary_linear: false
  protect_patterns:
    - bandsplit.proj
    - q_proj
    - k_proj
    - v_proj
    - mask_gen
    - istft

training:
  binary_stage_epochs:
    activation_warmup: 20
    weight_binarize: 280
```

## 6. 训练与评估

训练：

```bash
python audio_train.py --conf_dir configs/tiger-small.yml
```

本地二值化 smoke test：

```bash
python audio_train.py --conf_dir configs/tiger-small-binary.yml --epoch 1 --batch_size 1 --segment 2.0
```

本地蒸馏 smoke test：

```bash
python audio_train.py --conf_dir configs/tiger-small-distill.yml --epoch 1 --batch_size 1 --segment 2.0
```

蒸馏训练期间可重点关注以下日志：

- `train/task_loss`
- `train/kd_loss`
- `train/loss`
- `train/selected_patches_ratio`

推荐的本地小数据集验证顺序：

1. 先确认 `DataPreProcess/MiniLibriMix-mini/...` 已生成
2. 先跑 `configs/tiger-small-binary.yml`
3. 待 `Experiments/TIGER-MiniLibriMix-Binary/best_model.pth` 生成后，再跑 `configs/tiger-small-distill.yml`
4. 如果只想做最小冒烟，可以直接使用 `--epoch 1 --batch_size 1 --segment 2.0`

从断点继续训练：
```bash
# 恢复当前实验目录下的 last.ckpt
python audio_train.py --conf_dir configs/tiger-small.yml --resume

# 显式指定 checkpoint 路径，或写成 --resume_ckpt last
python audio_train.py --conf_dir configs/tiger-small.yml --resume_ckpt Experiments/checkpoint/TIGER-MiniLibriMix/last.ckpt
```

评估：

```bash
python audio_test.py --conf_dir configs/tiger-small.yml
```

训练产物默认输出到：

- `Experiments/checkpoint/<exp_name>/`

关键文件包括：

- `best_model.pth`
- `best_k_models.json`
- `results/metrics.csv`

训练保存策略：

- `last.ckpt` 每轮更新，用于断点续训
- 仅保留 1 个最优 checkpoint（按 `val/loss` 监控，文件名为 `best.ckpt`）
- `audio_test.py` 默认加载 `best_model.pth` 做评估；断点续训应加载 `.ckpt`
- 可在配置中设置 `main_args.resume_from_checkpoint: true`，也可通过命令行 `--resume` / `--resume_ckpt` 控制；命令行优先级更高

## 7. 推理示例

```bash
# 语音分离
python inference_speech.py --audio_path test/mix.wav

# DnR 示例
python inference_dnr.py --audio_path test/test_mixture_466.wav
```

## 8. 常见问题

- **找不到数据**：确认 `DataPreProcess/MiniLibriMix/`* 已生成且配置路径一致。
- **W&B 报错**：执行 `wandb login`，或在服务器中设置 `WANDB_API_KEY`。
- **显存不足**：先减小 `batch_size`，再降低模型规模参数。

## 9. 论文与引用

- 论文：[TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation](https://arxiv.org/abs/2410.01469)

```bibtex
@article{xu2024tiger,
  title={TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation},
  author={Xu, Mohan and Li, Kai and Chen, Guo and Hu, Xiaolin},
  journal={arXiv preprint arXiv:2410.01469},
  year={2024}
}
```
