# 原始 TIGER 模型代码结构与论文模块对应说明

本文只分析仓库里的原始全精度 TIGER 主线，不展开二值化和蒸馏分支。对应的训练入口请按环境选择 **`configs/tiger-small-local.yml`**（本地）或 **`configs/tiger-small-kaggle-t4x2.yml`**（Kaggle）等现存配置，配合 `audio_train.py` + `look2hear/models/tiger.py`。（旧版 `configs/tiger-small.yml` 已移除。）

## 1. 先看项目里“原始 TIGER”真正走的是哪条链路

原始模型的运行链路是：

1. `audio_train.py`
   - 读取上述 `configs/tiger-small-*.yml` 之一（如 `tiger-small-local.yml`）
   - 实例化数据模块 `Libri2MixModuleRemix`
   - 实例化音频模型 `look2hear.models.TIGER`
   - 实例化训练封装 `look2hear.system.AudioLightningModule`
2. `look2hear/datas/Libri2Mix16.py`
   - 负责从 `mix_both.json / s1.json / s2.json` 读取混合语音和目标语音
3. `look2hear/models/tiger.py`
   - 这是原始 TIGER 主体实现
4. `look2hear/system/audio_litmodule.py`
   - 负责 `training_step / validation_step / optimizer / scheduler`，不改模型结构

可以把它理解成两层：

- 训练工程层：`audio_train.py`、`AudioLightningModule`、`DataModule`
- 论文模型层：`TIGER`、`Recurrent`、`UConvBlock`、`MultiHeadSelfAttention2D`

## 2. 原始 TIGER 的代码结构

原始 TIGER 相关代码主要集中在一个文件里：

- `look2hear/models/tiger.py`

这个文件内部不是只放一个 `TIGER` 类，而是把论文里几个核心子模块都写在一起，结构从下往上大致是：

1. 基础卷积单元
   - `ConvNormAct`
   - `ConvNorm`
   - `DilatedConvNorm`
   - `ATTConvActNorm`
2. 多尺度与融合组件
   - `Mlp`
   - `InjectionMultiSum`
   - `InjectionMulti`
   - `UConvBlock`
3. 时频交错主干组件
   - `MultiHeadSelfAttention2D`
   - `Recurrent`
4. 整体模型
   - `TIGER`

这说明作者采用的是“把论文主干集中写在单文件里”的实现方式：基础层先定义，再逐层拼成论文模块，最后由 `TIGER.forward()` 串成端到端推理流程。

## 3. 论文模块和代码模块的对应关系

| 论文中的模块 | 代码中的位置 | 说明 |
| --- | --- | --- |
| 输入波形 | `TIGER.forward()` | 接收 `(B,T)`、`(B,C,T)` 或一维波形 |
| Encoder / analysis transform | `torch.stft` in `TIGER.forward()` | 论文里的时域到时频域分析变换，在代码里直接用 STFT，不单独封装成类 |
| Band-split module | `self.band_width` + `forward()` 中的子带切分循环 | 用不均匀子带划分频谱 |
| 子带压缩 / embedding | `self.BN` | 每个子带一套 `GroupNorm + 1x1 Conv1d` |
| Separator | `self.separator = Recurrent(...)` | 原始 TIGER 的主干 |
| FFI block / 时频交错更新 | `Recurrent.freq_time_process()` | 每次先过 frequency path，再过 frame path |
| MSA / 多尺度聚合 | `UConvBlock` | 论文里的多尺度上下文建模主要落在这里 |
| F3A / 全频帧注意力 | `MultiHeadSelfAttention2D` | 负责补充全局长程依赖 |
| 归一化 | `LayerNormalization4D`、`GroupNorm` | 路径内部和子带压缩都用了归一化 |
| Gain extraction / mask estimation | `self.mask` | 每个子带一套 mask head |
| Reconstruction / synthesis transform | 复数 mask 重建 + `torch.istft` | 从子带估计频谱回到波形 |

## 4. 从论文到代码：每个模块是怎么写出来的

### 4.1 STFT 不是单独 Encoder 类，而是直接写在 `forward`

论文里通常会把编码器和解码器画成模块，但这里没有写成 `Encoder` / `Decoder` 类，而是直接在 `TIGER.forward()` 里调用：

- `torch.stft(...)`
- `torch.istft(...)`

这样写的逻辑很直接：

1. TIGER 本质上是时频域分离模型
2. 时频变换本身没有可学习参数
3. 因此代码里直接把 STFT/iSTFT 放在前向流程里，避免再做一层无意义封装

所以论文中的“编码器/解码器”在代码里更准确地说是“分析变换/合成变换”，不是神经网络子类。

### 4.2 Band-split 是“静态子带规划 + 运行时切片”

`TIGER.__init__()` 里先计算 `band_width`，把整个频率轴拆成不均匀子带：

- 低频更细
- 高频更粗
- 最后一个子带负责补齐剩余频点

这部分对应论文里利用先验频带结构降低复杂度的思想。

代码逻辑分两步：

1. 在 `__init__()` 里确定子带宽度表 `self.band_width`
2. 在 `forward()` 里用一个循环按这个表从 STFT 结果中切出 `subband_spec` 和 `subband_spec_RI`

这里的实现重点不是复杂算子，而是张量组织：

- `spec`: 复数频谱
- `spec_RI`: 把实部、虚部拆成两个通道
- 每个子带都保留自己的频宽 `BW`

也就是说，论文里的 band-split 在代码里没有单独类，而是“一个频带配置表 + 一段切片逻辑”。

### 4.3 子带压缩 `BN` 是原始 TIGER 的第一层可学习映射

每个子带的原始输入维度不同，因为 `BW` 不同。要让后面的 separator 统一处理，必须先映射到相同的特征维度 `feature_dim`。

这就是 `self.BN` 的作用：

- 输入：`2 * BW`，因为实部和虚部拼接了
- 处理：`GroupNorm(1, 2*BW) + Conv1d(2*BW, feature_dim, 1)`
- 输出：统一的 `feature_dim`

这一步对应论文里的子带投影/压缩层。

编写逻辑很清楚：

1. 先按子带处理，避免所有频点共享同一投影
2. 用 `ModuleList` 保存每个子带自己的压缩层
3. 最终把所有子带特征 `stack` 成 `[B, nband, N, T]`

所以 separator 接收到的不是原始频谱，而是“每个子带已经变成统一维度后的子带嵌入”。

### 4.4 Separator 的核心是 `Recurrent`，不是传统 RNN

虽然类名叫 `Recurrent`，但它不是 LSTM/GRU。这里的 “recurrent” 指的是“反复迭代时频交错更新”。

`Recurrent` 的结构是：

- `freq_path`
  - `UConvBlock`
  - `MultiHeadSelfAttention2D`
  - `LayerNormalization4D`
- `frame_path`
  - `UConvBlock`
  - `MultiHeadSelfAttention2D`
  - `LayerNormalization4D`

然后在 `forward()` 中循环 `self.iter` 次，每次调用 `freq_time_process()`。

这对应论文里“time-frequency interleaved”这件事的真正落点：

1. 先在频带维上建模
2. 再在时间帧维上建模
3. 重复多次，让特征不断细化

这里的代码写法有两个关键点：

1. 通过 `permute/view` 在不同路径下重解释张量维度
   - 做频带路径时，把 `nband` 当成序列长度
   - 做时间路径时，把 `T` 当成序列长度
2. 用残差和 `concat_block` 维持多轮迭代稳定性

因此，论文里的“交错建模”在代码里不是抽象概念，而是非常具体的维度重排策略。

### 4.5 `freq_time_process()` 是论文 FFI block 最接近的代码实现

如果只找一个最像论文框图里核心 block 的函数，就是 `Recurrent.freq_time_process()`。

它的执行顺序是：

1. 频带路径
   - 输入重排到按频带处理的视角
   - 过 `freq_path[0]`，即 `UConvBlock`
   - 过 `freq_path[1]`，即 `MultiHeadSelfAttention2D`
   - 过 `freq_path[2]`，即归一化
   - 残差相加
2. 时间路径
   - 再重排到按时间处理的视角
   - 过 `frame_path[0]`
   - 过 `frame_path[1]`
   - 过 `frame_path[2]`
   - 残差相加

这个函数说明论文模块在代码里不是一一对应到类名，而是常常对应到“一个流程函数 + 若干子模块”。

### 4.6 `UConvBlock` 才是论文里多尺度模块的主要承载体

`UConvBlock` 不是简单卷积块，而是一个小型 U 形多尺度结构。它承担了论文里多尺度聚合那一部分。

内部主要由四段组成：

1. `proj_1x1`
   - 先把输入投影到内部通道空间
2. `spp_dw`
   - 多层 depthwise/dilated 风格的 `Conv1d`
   - 逐层降采样，形成不同时间尺度的特征
3. `globalatt`
   - 把多尺度特征池化到同一长度后求和
   - 再用 `Mlp` 做全局建模
4. `loc_glo_fus` + `last_layer`
   - 把全局特征逐步注入回局部尺度
   - 最后上采样式融合，恢复高分辨率表示

最后再接一个 `res_conv` 并与输入残差相加。

这段代码体现的编写逻辑是：

- 论文想表达“多尺度上下文”
- 代码上就做成“下采样分支 -> 全局聚合 -> 逐层回注 -> 残差输出”

所以阅读 `UConvBlock` 时，不要只盯单个卷积层，而要把它看成一个完整的编码-聚合-融合单元。

### 4.7 `InjectionMultiSum` 是多尺度特征融合的关键胶水层

论文里通常会说局部与全局特征要做 selective fusion。代码里，这部分主要由 `InjectionMultiSum` 完成。

它做的事情是：

1. 对局部特征做一条投影
2. 对全局特征做两条投影
   - 一条生成门控 `sigmoid`
   - 一条生成要注入的全局特征
3. 把全局特征插值到与局部特征同长度
4. 输出 `local * gate + global`

这就是典型的“门控注入”写法。

也就是说，论文里的“选择性融合”在代码里不是单独的大模块，而是靠这种很小但很关键的融合单元实现的。

### 4.8 `MultiHeadSelfAttention2D` 对应论文里的全频帧注意力

这个类对应论文中的注意力模块，代码实现逻辑是：

1. 每个 head 分别生成 `Q/K/V`
   - 用 `ATTConvActNorm(..., kernel_size=1, is2d=True)`
2. 把 `[B, E, T, F]` 变形成适合做 attention 的形状
3. 在时间维上做注意力矩阵 `softmax(QK^T / sqrt(d))`
4. 用注意力权重更新 `V`
5. 多头拼接后再做一次投影
6. 残差相加

这里很重要的一点是：它虽然名字叫 `2D` attention，但真正的 attention 计算是在重排后的时间轴上做的，频率维和通道维被展平进 embedding。

所以这段代码体现的是一种工程化实现：

- 论文说“full-frequency-frame attention”
- 代码则把“全频信息”塞进 embedding，把“帧间关系”交给 self-attention

### 4.9 `mask` 头对应论文里的 gain extraction / reconstruction

`self.mask` 是另一个按子带拆分的 `ModuleList`。每个子带各自预测自己的复数 mask。

单个子带的逻辑是：

1. 输入 `sep_output[:, i]`
2. 经过 `PReLU + Conv1d`
3. reshape 成 `[B, 2, 2, K, BW, T]`
4. 其中一部分做门控，形成实部和虚部 mask
5. 与原始子带复数谱相乘，得到估计子带频谱

这里代码里还有两个很重要的工程细节：

1. mask 的实部和虚部分开预测
2. 对多源 mask 做了和约束
   - 实部和约束到 1
   - 虚部做零均值约束

这说明论文里的 gain extraction 在代码里并不是“直接输出分离频谱”，而是“先输出子带复数 mask，再恢复频谱”。

### 4.10 重建阶段就是“拼回全频谱再 iSTFT”

所有子带的估计频谱最终会：

1. `torch.cat` 回完整频率轴
2. reshape 成 `[B * num_sources, F, T]`
3. 输入 `torch.istft(...)`

这一步就是论文里的 reconstruction/synthesis 过程。

到这里为止，原始 TIGER 的前向链路就闭合了：

`waveform -> STFT -> subband embedding -> interleaved separator -> subband mask -> full-band spectrum -> iSTFT`

## 5. 用“论文视角”重新理解 `TIGER.forward()`

如果按论文顺序读代码，`TIGER.forward()` 可以压缩成下面 7 步：

1. 输入波形做 STFT
2. 把复数谱切成多个非均匀子带
3. 每个子带做归一化和 1x1 压缩，映射到统一通道维
4. 把所有子带堆叠后送入 `separator`
5. `separator` 内部重复执行频带路径与时间路径交错更新
6. 每个子带预测复数 mask 并作用到原始子带频谱上
7. 拼回全频谱并做 iSTFT

因此，`forward()` 既是推理代码，也是论文主流程最直观的落地位置。

## 6. 原始 TIGER 和仓库扩展分支的边界

这个仓库后面又加入了：

- `look2hear/models/binary_tiger.py`（二值化学生网络）
- `look2hear/utils/model_converter.py`（`TIGERBinaryConverter`：Conv → BinaryConv、插入 RSign、PReLU → RPReLU、保护策略与 `binarize_scope`）
- `look2hear/layers/binary_layers.py`（`BinaryConv1d/2d`、`RSign`、`RPReLU`、Clipped STE、EMA Scale）
- `look2hear/system/binary_audio_litmodule.py`（二值化训练阶段与 EMA 更新）
- `look2hear/system/binary_distill_litmodule.py`（D1/D2/D3 与任务损失联合、`audio_train.py` 中 `BinaryDistillAudioLitModule`）
- `look2hear/layers/kd_losses.py`（PIT 对齐、SI-SNR 蒸馏、子带加权 STFT 损失）
- `论文/` 目录下《二值化技术方案》《蒸馏技术方案》《综合实验方案》（与上述实现对齐的定稿说明）

但这些都不是原始 TIGER 论文主干本身，而是在原始 `TIGER` 基础上的扩展训练路线。

如果你的目标是理解论文原始模型，优先只看下面几个文件：

- `audio_train.py`
- `configs/tiger-small-local.yml` 或 `configs/tiger-small-kaggle-t4x2.yml`（任选其一作入口）
- `look2hear/models/tiger.py`
- `look2hear/system/audio_litmodule.py`
- `look2hear/datas/Libri2Mix16.py`

## 7. 总结：论文模块在代码里是怎么组织的

原始 TIGER 的实现不是“一个论文模块对应一个 Python 文件”，而是三层嵌套：

1. `TIGER`
   - 负责整条前向主流程
2. `Recurrent`
   - 负责时频交错更新
3. `UConvBlock` 与 `MultiHeadSelfAttention2D`
   - 分别承担多尺度建模和全局注意力

更具体地说：

- 论文里的大模块，代码里往往落成一个类
- 论文里的中模块，代码里常常落成一个子类或一个 `ModuleList`
- 论文里的小步骤，代码里通常落成一次 `permute/view`、一段循环或一次残差连接

所以读这个项目时，最有效的顺序不是先看训练脚本，而是：

1. 先看 `TIGER.__init__()`，搞清有哪些子模块
2. 再看 `TIGER.forward()`，搞清数据流
3. 再看 `Recurrent.freq_time_process()`，搞清时频交错逻辑
4. 最后看 `UConvBlock` 和 `MultiHeadSelfAttention2D`，搞清论文细节如何落地

这就是原始 TIGER 从论文结构到代码结构的主要对应关系。
