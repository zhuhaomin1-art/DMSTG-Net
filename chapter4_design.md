# 第四章实现设计（多粒度时序解耦 + 对比学习增强 v2）

本版本在基础方案上，进一步加入三个“论文创新亮点”：

1. **物理特性导向的异构编码器（Physics-Oriented Heterogeneous Encoder）**
   - 高频分量：扩张卷积（Dilated Conv）建模突变与湍流。
   - 周期分量：多头自注意力（Self-Attention）建模全局日内规律。
   - 趋势分量：深度可分离卷积保持平稳低频结构。

2. **分量级对比增强（Component-wise Contrastive Learning）**
   - 视图1：仅对高频分量加噪，周期与趋势保持不变。
   - 视图2：趋势分量跨样本互换，保持高频与周期不变。
   - 同时优化全局表示对比与分量表示（HF/Trend）对比。

3. **物理一致性约束（Physics-Consistency Constraint）**
   - 非负功率约束（输出端 `ReLU`）。
   - 爬坡率约束（相邻预测时刻变化率上限惩罚）。

## 文件说明

- `model_ch4.py`
  - `MultiGranularityDecomposer`：分解 `hf/daily/trend`
  - `TurbulenceEncoder` / `PeriodicEncoder` / `TrendEncoder`：异构编码器
  - `Chapter4WindPowerModel`：融合预测 + 全局/分量表示投影

- `train_ch4.py`
  - `make_componentwise_views`：分量级增强（高频加噪 + 趋势互换）
  - `info_nce_loss`：InfoNCE 对比损失
  - `physics_consistency_loss`：物理一致性惩罚
  - 联合目标：`L = L_pred + λ_cl * L_cl + λ_phy * L_phy`

## 训练目标

\[
\mathcal{L}=\mathcal{L}_{pred}+\lambda_{cl}\mathcal{L}_{cl}+\lambda_{phy}\mathcal{L}_{phy}
\]

其中：
- \(\mathcal{L}_{pred}\)：masked MAE
- \(\mathcal{L}_{cl}\)：全局 + 分量级 InfoNCE
- \(\mathcal{L}_{phy}\)：非负 + 爬坡率约束

## 与第三章代码结合建议

- 可将高频分支替换为第三章 `SpatioTemporalLayer + 动态邻接`，进一步增强风机间传播建模。
- 可将风速阈值门控（cutoff mask）并入 `pred` 后处理，与物理一致性约束协同。
- 可在趋势分支加入天气先验（如温度/气压低频先验）作为辅助通道。
