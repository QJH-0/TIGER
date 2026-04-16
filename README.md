# TIGER（MiniLibriMix 默认版）

本仓库用于语音分离模型 TIGER 的训练、评估与推理。  
当前默认数据流程已切换为 **MiniLibriMix**。

## 1. 项目文档入口

- 项目树形结构说明：`docs/project_tree_zh.md`
- 训练流程教程（详细版）：`docs/training_tutorial_zh.md`

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

## 6. 训练与评估

训练：

```bash
python audio_train.py --conf_dir configs/tiger-small.yml
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
