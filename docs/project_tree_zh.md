# TIGER 项目树形结构说明

本文档用于快速理解项目目录职责（以当前仓库结构为准）。

```text
TIGER/
├── .github/                        # GitHub 工作流与仓库配置
├── assets/                         # README/网页展示图片资源
├── configs/                        # 训练配置文件（YAML）
│   └── tiger-large.yml             # 当前默认训练配置（已切到 MiniLibriMix 索引）
├── DataPreProcess/                 # 数据预处理与索引生成脚本
│   ├── MiniLibriMix/               # 已生成的训练索引目录
│   │   ├── train/                  # train 索引（mix_both/s1/s2.json）
│   │   ├── val/                    # val 索引
│   │   └── test/                   # test 索引
│   ├── process_librimix.py         # 生成 LibriMix/MiniLibriMix JSON 索引
│   ├── split_val_to_test.py        # 按样本ID从 val 同步切分到 test
│   ├── sync_minilibrimix_metadata.py # 同步 metadata CSV（val/test）
│   ├── process_echoset.py          # EchoSet 数据索引脚本
│   └── preprocess_lrs2_audio.py    # LRS2 数据索引脚本
├── docs/                           # 项目文档
│   ├── training_tutorial_zh.md     # 训练流程教程（中文）
│   └── project_tree_zh.md          # 当前文件：树形结构说明
├── look2hear/                      # 核心代码
│   ├── datas/                      # DataModule 与 Dataset 定义
│   ├── models/                     # 模型实现（含 TIGER）
│   ├── losses/                     # 损失函数
│   ├── metrics/                    # 评测指标
│   ├── layers/                     # 网络层组件
│   ├── system/                     # Lightning 系统封装与训练流程
│   └── utils/                      # 通用工具函数
├── test/                           # 示例测试音频
├── audio_train.py                  # 训练入口脚本
├── audio_test.py                   # 测试/评估入口脚本
├── inference_speech.py             # 单条语音推理脚本
├── inference_dnr.py                # DnR 音频推理脚本
├── requirements.txt                # Python 依赖
└── README.md                       # 项目使用说明（默认 MiniLibriMix）
```

## 核心运行链路

1. `DataPreProcess/process_librimix.py` 生成 `DataPreProcess/MiniLibriMix/*/*.json`。  
2. `configs/tiger-large.yml` 指向上述 JSON 索引。  
3. `audio_train.py` 读取配置并启动训练。  
4. `audio_test.py` 读取训练产物做测试并输出 `metrics.csv`。  

