# TIGER 原始结构与二值化对应（含 Mermaid）

本文档基于 TIGER 论文与本仓库实现，给出：

- **原始 TIGER（全精度）结构**：按论文的五大组件 + separator(FFI block)内部结构展开
- **二值化后的对应结构**：说明哪些层被替换为二值算子、哪些模块被保护保持全精度，以及它们与原始结构的对应关系

参考论文：[`TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation`](https://arxiv.org/abs/2410.01469)

---

## 1. 术语对照（论文 ↔ 仓库代码）

> 说明：论文中的符号 \(F,T,K,N\) 等在不同实现里维度排列可能转置，但模块拓扑一致。

- **Encoder/Decoder**（论文）↔ `torch.stft` / `torch.istft`（`look2hear/models/tiger.py:TIGER.forward`）
- **Band-split module**（论文 3.2）↔ `band_width` 子带划分 + 逐子带 `BN[i]` 压缩（`TIGER.__init__`）
- **Separator**（论文 3.3）↔ `Recurrent`（`TIGER.separator`）
  - **FFI block**（论文 Figure 2）↔ `Recurrent.freq_time_process()` 的一次 “Frequency path → Frame path” 更新
  - **MSA**（论文 3.3.1）↔ `UConvBlock`（代码里注释称 MAS/MSA 的承载块）
  - **F3A**（论文 3.3.2 Full-frequency-frame attention）↔ `MultiHeadSelfAttention2D`
  - **LayerNorm** ↔ `LayerNormalization4D`（`normalizations.get("LayerNormalization4D")`）
- **Band restoration module**（论文 3.4）↔ `mask[i]` 子带掩码/增益头 + 子带拼回全频（`TIGER.forward` 里 `self.mask` 与 `torch.cat`）
- **二值化包装器** ↔ `look2hear/models/binary_tiger.py:BinaryTIGER`
- **二值算子** ↔ `look2hear/layers/binary_layers.py:BinaryConv1d / BinaryLinear`
- **二值化转换规则** ↔ `look2hear/utils/model_converter.py:TIGERBinaryConverter`

---

## 2. 原始 TIGER（全精度）整体结构（论文 3.1）

### 2.1 总体数据流（Mermaid）

```mermaid
flowchart TD
  A[Mixture waveform] --> B[STFT]

  B --> C[Band-split]
  C --> D[RI merge]
  D --> E[Subband BN\nGN + 1×1 Conv1d]
  E --> F[Stack Z]

  F --> G[Separator\nFFI × B (shared)]

  G --> H[Mask head (per-band)]
  H --> I[Concat mask]
  I --> J[Apply mask]
  J --> K[iSTFT]
```

图外补充（避免塞进图里）：

- **整体**：waveform → STFT 得到复数谱 \(X\) → 子带划分/压缩得到 \(Z\in\mathbb{R}^{N\times K\times T}\) → separator 输出 → mask 复原到全频 → iSTFT 回时域。
- **Band-split/BN**：每个子带把 real/imag 合并成频率维（\(2G_k\)），再用 `GroupNorm + 1×1 Conv1d` 映射到统一通道数 \(N\)。

### 2.2 Separator（FFI block）内部结构（论文 3.3）

论文的 separator 由 **多个 FFI blocks** 串联构成，并且 **这些 blocks 共享参数**。每个 FFI block 内部按顺序执行：

- **Frequency path**：沿 “子带维（K）” 建模跨频带上下文
- **Frame path**：沿 “时间帧维（T）” 建模时序上下文

每条 path 都是同构的三段：**MSA → F3A → LayerNorm**，并配残差连接。

```mermaid
flowchart LR
  subgraph FFI[FFI block (shared)]
    IN[Z] --> FP[Freq path]
    FP --> TP[Frame path]
    TP --> OUT[Z']
  end

  subgraph PATH[Path = MSA + F3A + LN]
    P0[in] --> MSA[MSA / UConvBlock]
    MSA --> F3A[F3A / MHSA2D]
    F3A --> LN[LN4D]
    LN --> PR[+res]
  end

  FP -.内部即.-> PATH
  TP -.内部即.-> PATH
```

图外补充：

- **Freq path**：主要沿子带维 \(K\) 建模跨频带上下文。
- **Frame path**：主要沿时间帧维 \(T\) 建模时序上下文。
- 两条 path 的模块拓扑一致，仅“处理维度”不同。

### 2.3 MSA（论文 3.3.1）与代码实现要点

论文 MSA 分三阶段：**Encoding（多尺度下采样）→ Fusing（选择性注意力融合）→ Decoding（逐层上采样恢复）**。

在本仓库中，这个思想主要由 `UConvBlock` 承载（`look2hear/models/tiger.py`）：

- **多尺度下采样**：`spp_dw`（多层 stride=2 的 depthwise/组卷积，扩大感受野/降低分辨率）
- **全局聚合**：把不同尺度特征 `adaptive_avg_pool1d` 到同一尺度后相加，再经 `globalatt`（`Mlp`）
- **选择性注入（SA/Selective Attention）**：`InjectionMultiSum`（`sigmoid(global_act)` 门控）实现 \( \sigma(x)\odot y + z \) 形式的融合
- **逐层解码/上采样融合**：`last_layer` + `InjectionMultiSum` 逐层把低分辨率信息注入高分辨率

### 2.4 F3A（论文 3.3.2）与代码实现要点

论文 F3A 的核心是：对每个 head 生成 Q/K/V（1×1 Conv2d），把 “时间帧维 + embedding 维” 合并后做注意力，得到跨全频（或全帧）上下文的聚合，再 concat 投影并残差。

本仓库用 `MultiHeadSelfAttention2D` 实现该模块：

- **Q/K/V 生成**：`ATTConvActNorm(..., kernel_size=1, is2d=True)`
- **注意力矩阵**：`attn_mat = softmax(Q K^T / sqrt(dim))`
- **concat + 投影 + 残差**：`attn_concat_proj` + `x + residual`

---

## 3. 二值化后的对应结构（BinaryTIGER）

### 3.1 “二值化”在本仓库里的定义

本仓库的二值化实现是 **结构不变、算子替换**：

- 先构建原始 `TIGER`
- 再用 `TIGERBinaryConverter` 遍历模型，把满足条件的 `nn.Conv1d`（以及可选的 `nn.Linear`）替换为二值版本
- 二值算子使用 **STE（Straight-Through Estimator）**：前向权重取 sign 投影，反向仍对潜在实值权重传梯度（`BinaryConv1d/_ste_sign`）

入口类：`look2hear/models/binary_tiger.py:BinaryTIGER`

### 3.2 二值化替换规则（代码真实口径）

`look2hear/utils/model_converter.py:TIGERBinaryConverter` 的默认策略（关键点）：

- **保护第一层 Conv1d**：遇到的第一个 `nn.Conv1d` 保持全精度（`protect_first_layer=True`）
- **保护注意力相关模块**：名称段包含 `queries/keys/values/attn_concat_proj/attn/...` 或类名含 `attention`（`protect_attention=True`）
- **保护 1×1 Conv1d**：所有 `kernel_size == 1` 的 `Conv1d` 保持全精度（`protect_1x1_conv=True`）
- **保护输出头（mask/output）**：名字包含 `mask` 或 `output`（`protect_output_layer=True`）
- **可选二值化 Linear**：默认关闭（`enable_binary_linear=False`）

因此，“二值化覆盖”的主要对象通常是：**kernel_size > 1 的 Conv1d**（尤其是 MSA/UConvBlock 内的 depthwise/膨胀卷积），而 **子带压缩的 1×1、注意力投影、mask 头** 都会保留全精度。

### 3.3 二值化后的整体结构（Mermaid：结构不变 + 标注替换点）

```mermaid
flowchart TD
  classDef fp fill:#eef6ff,stroke:#4e79a7,stroke-width:1px,color:#000;
  classDef bin fill:#fff1f2,stroke:#e15759,stroke-width:1px,color:#000;
  classDef protected fill:#f3f4f6,stroke:#6b7280,stroke-width:1px,color:#000;

  A[waveform]:::fp --> B[STFT]:::protected

  B --> C[Band-split]:::protected
  C --> D[BN (1×1)]:::protected
  D --> E[Separator]:::fp

  subgraph SEP[FFI (repeat, shared)]
    E0[Freq path]:::fp --> E1[Frame path]:::fp
  end

  subgraph PATH_FP[Path]
    P0[MSA]:::fp --> P1[F3A]:::protected --> P2[LN4D]:::protected
  end

  %% 重点：MSA 内部哪些 Conv1d 更可能二值化
  subgraph MSA_DETAIL[MSA 里：常见替换点]
    M0[1×1 conv]:::protected
    M1[k>1 conv]:::bin
    M2[attention-like]:::protected
  end

  E0 -.内部结构.-> PATH_FP
  E1 -.内部结构.-> PATH_FP

  E --> F[Mask head]:::protected --> G[iSTFT]:::protected --> H[outputs]:::fp
```

> 图例说明：
>
>- **灰色（protected）**：按默认规则保护，不做二值化（例如：所有 1×1 Conv1d、注意力投影、mask 头、STFT/iSTFT）
>- **浅红（bin）**：典型会被替换为 `BinaryConv1d` 的位置（例如：`UConvBlock.spp_dw` 里的 k=5 depthwise/组卷积）
>- **浅蓝（fp）**：结构模块（内部可能混合全精度与二值算子）

### 3.4 模块映射表（论文模块 → 代码模块 → 二值化策略）

- **Encoder (STFT)**：`torch.stft` → 不二值化（算子层面不替换）
- **Band-split + 子带压缩**：`band_width` + `BN[i]`（含 1×1 Conv1d）→ 默认保护（`protect_1x1_conv=True`）
- **Separator / FFI blocks**：`Recurrent`
  - **MSA**：`UConvBlock`
    - `proj_1x1` / `res_conv`（1×1）→ 默认保护
    - `spp_dw`（k=5，多为 depthwise/组卷积，且非 1×1）→ **主要二值化目标**（会替换成 `BinaryConv1d`）
    - `globalatt`（`Mlp`）：其中 depthwise k=5 Conv1d 可能被二值化；1×1 Conv1d 默认保护
    - `loc_glo_fus` / `last_layer`：是否二值化取决于其内部 Conv1d kernel_size（k=1 保护，k>1 可二值化）
  - **F3A**：`MultiHeadSelfAttention2D` → 默认保护（注意力相关模块）
  - **LayerNorm**：`LayerNormalization4D` → 不二值化
- **Band-restoration / Mask head**：`mask[i]`（输出层）→ 默认保护（`protect_output_layer=True`）
- **Decoder (iSTFT)**：`torch.istft` → 不二值化

---

## 4. 你可以如何使用这份图（实践建议）

- **想对齐论文结构**：优先看 “2.1 总体数据流” 与 “2.2 FFI block + path”
- **想看二值化落点**：优先看 “3.2 替换规则” 与 “3.3 二值化结构图”
- **想确认实际替换了哪些层**：运行 `BinaryTIGER().get_binarization_summary()` 可拿到 `converted_modules/protected_modules` 名单（代码已提供统计接口）

