# 第四章实现设计（多粒度时序解耦 + 对比学习增强）

本实现围绕两个核心模块：

1. **多粒度时序特征解耦（Multi-Granularity Decomposition）**
   - 将输入序列分为：
     - 高频湍流分量（HF）
     - 日内周期分量（Daily）
     - 日级趋势分量（Trend）
   - 每个分量使用独立编码器和预测头建模，最后通过可学习门控融合。

2. **时序对比学习增强（Temporal Contrastive Learning）**
   - 对同一批次样本做两次增强（jitter + 时间mask）得到正样本对。
   - 使用 InfoNCE 约束表示一致性，提升小样本场景泛化能力。

## 文件说明

- `model_ch4.py`
  - `MultiGranularityDecomposer`
  - `TemporalEncoder`
  - `ComponentPredictor`
  - `Chapter4WindPowerModel`

- `train_ch4.py`
  - `WindFarmDataset`
  - `timeseries_augment`
  - `info_nce_loss`
  - 联合训练循环（监督损失 + 对比损失）

## 训练目标

\[
\mathcal{L}=\mathcal{L}_{pred}+\lambda_{cl}\mathcal{L}_{InfoNCE}
\]

其中：
- \(\mathcal{L}_{pred}\)：masked MAE
- \(\mathcal{L}_{InfoNCE}\)：对比损失
- \(\lambda_{cl}\)：对比损失权重

## 与第三章代码结合建议

- 可以将 `TemporalEncoder` 替换为第三章中的 `SpatioTemporalLayer` 堆叠结构。
- 可把第三章的动态邻接机制作为分量编码器的空间建模部分（尤其 HF 分量）。
- 输出层继续保留物理约束（如风速cutoff mask）。
