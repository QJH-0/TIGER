# TIGER（MiniLibriMix 默认版）

本仓库用于语音分离模型 TIGER 的训练、评估与推理。  
当前默认数据流程已切换为 **MiniLibriMix**。

## 1. 项目文档入口

- 项目树形结构说明：`docs/project_tree_zh.md`
- 训练流程教程（详细版）：`docs/training_tutorial_zh.md`

## 2. 环境安装

```bash
git clone https://github.com/JusperLee/TIGER.git
cd TIGER
pip install -r requirements.txt
```

W&B 首次使用需要登录一次：

```bash
wandb login
```

## 3. 默认数据与索引生成（MiniLibriMix）

默认原始数据目录：

- `D:\Paper\datasets\MiniLibriMix`

生成训练索引（JSON）到项目内：

```bash
python DataPreProcess/process_librimix.py --in_dir "D:/Paper/datasets/MiniLibriMix" --out_dir "D:/Paper/TIGER/DataPreProcess/MiniLibriMix" --splits train val test --speakers mix_both s1 s2
```

生成后目录应为：

- `DataPreProcess/MiniLibriMix/train/{mix_both.json,s1.json,s2.json}`
- `DataPreProcess/MiniLibriMix/val/{mix_both.json,s1.json,s2.json}`
- `DataPreProcess/MiniLibriMix/test/{mix_both.json,s1.json,s2.json}`

## 4. 默认训练配置

默认配置文件：`configs/tiger-large.yml`

已配置为：

- `data_name: Libri2MixModuleRemix`
- `train_dir: DataPreProcess/MiniLibriMix/train`
- `valid_dir: DataPreProcess/MiniLibriMix/val`
- `test_dir: DataPreProcess/MiniLibriMix/test`

## 5. 训练与评估

训练：

```bash
python audio_train.py --conf_dir configs/tiger-large.yml
```

评估：

```bash
python audio_test.py --conf_dir configs/tiger-large.yml
```

训练产物默认输出到：

- `Experiments/checkpoint/<exp_name>/`

关键文件包括：

- `best_model.pth`
- `best_k_models.json`
- `results/metrics.csv`

## 6. 推理示例

```bash
# 语音分离
python inference_speech.py --audio_path test/mix.wav

# DnR 示例
python inference_dnr.py --audio_path test/test_mixture_466.wav
```

## 7. 常见问题

- **找不到数据**：确认 `DataPreProcess/MiniLibriMix/*` 已生成且配置路径一致。
- **W&B 报错**：执行 `wandb login`，或在服务器中设置 `WANDB_API_KEY`。
- **显存不足**：先减小 `batch_size`，再降低模型规模参数。

## 8. 论文与引用

- 论文：[TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation](https://arxiv.org/abs/2410.01469)

```bibtex
@article{xu2024tiger,
  title={TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation},
  author={Xu, Mohan and Li, Kai and Chen, Guo and Hu, Xiaolin},
  journal={arXiv preprint arXiv:2410.01469},
  year={2024}
}
```
