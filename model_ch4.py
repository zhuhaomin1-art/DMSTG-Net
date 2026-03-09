import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiGranularityDecomposer(nn.Module):
    """多粒度时序分解：高频(残差)、日内周期、日级趋势。"""

    def __init__(self, in_channels: int, period_len: int = 144, trend_kernel: int = 25):
        super().__init__()
        self.in_channels = in_channels
        self.period_len = period_len
        self.trend_pool = nn.AvgPool1d(kernel_size=trend_kernel, stride=1, padding=trend_kernel // 2)

    def _reshape_for_time_ops(self, x: torch.Tensor):
        # x: [B, C, N, T] -> [B*N, C, T]
        b, c, n, t = x.shape
        return x.permute(0, 2, 1, 3).reshape(b * n, c, t), b, n, t

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, N, T]
        Returns:
            hf, daily, trend: 三个分量，形状均为 [B, C, N, T]
        """
        x_bt, b, n, t = self._reshape_for_time_ops(x)

        # 趋势：滑动平均
        trend = self.trend_pool(x_bt)

        # 日内周期：按周期位置求均值
        daily = torch.zeros_like(x_bt)
        if t >= self.period_len:
            num_full = t // self.period_len
            for k in range(self.period_len):
                idx = torch.arange(k, num_full * self.period_len, self.period_len, device=x.device)
                if idx.numel() > 0:
                    mean_k = x_bt[:, :, idx].mean(dim=-1, keepdim=True)
                    daily[:, :, idx] = mean_k

            tail_start = num_full * self.period_len
            if tail_start < t:
                tail_idx = torch.arange(tail_start, t, device=x.device)
                ref_idx = torch.arange(tail_idx.numel(), device=x.device)
                daily[:, :, tail_idx] = daily[:, :, ref_idx]
        else:
            daily = x_bt.mean(dim=-1, keepdim=True).expand_as(x_bt)

        low = 0.5 * trend + 0.5 * daily
        hf = x_bt - low

        def recover(tensor_bt):
            return tensor_bt.reshape(b, n, self.in_channels, t).permute(0, 2, 1, 3)

        return recover(hf), recover(daily), recover(trend)


class TurbulenceEncoder(nn.Module):
    """高频分量编码器：扩张卷积强化突变感知。"""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 2), dilation=(1, 2)),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 4), dilation=(1, 4)),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class PeriodicEncoder(nn.Module):
    """周期分量编码器：时序自注意力建模全局日周期。"""

    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        # x: [B, C, N, T]
        b, c, n, t = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(b * n, t, c)
        seq = self.proj(seq)
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.norm1(seq + attn_out)
        seq = self.norm2(seq + self.ffn(seq))
        return seq.reshape(b, n, t, -1).permute(0, 3, 1, 2)


class TrendEncoder(nn.Module):
    """趋势分量编码器：深度可分离卷积，强调平稳长程结构。"""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 5), padding=(0, 2), groups=in_dim)
        self.pointwise = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.out = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.out(x)


class ComponentPredictor(nn.Module):
    """针对单一分量的预测头。"""

    def __init__(self, hidden_dim: int, out_horizon: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )
        self.out_horizon = out_horizon

    def forward(self, h: torch.Tensor):
        y = self.head(h)
        if y.shape[-1] >= self.out_horizon:
            return y[..., -self.out_horizon:]
        return F.interpolate(y, size=(y.shape[-2], self.out_horizon), mode="nearest")


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor):
        return F.normalize(self.mlp(z), dim=-1)


class Chapter4WindPowerModel(nn.Module):
    """
    第四章模型(v2)：
    - 多粒度分解
    - 异构编码（高频/周期/趋势）
    - 注意力门控融合
    - 分量级与全局对比表示
    """

    def __init__(self, in_dim: int = 14, hidden_dim: int = 128, out_horizon: int = 12, period_len: int = 144):
        super().__init__()
        self.decomposer = MultiGranularityDecomposer(in_channels=in_dim, period_len=period_len)

        self.hf_encoder = TurbulenceEncoder(in_dim, hidden_dim)
        self.daily_encoder = PeriodicEncoder(in_dim, hidden_dim)
        self.trend_encoder = TrendEncoder(in_dim, hidden_dim)

        self.hf_pred = ComponentPredictor(hidden_dim, out_horizon)
        self.daily_pred = ComponentPredictor(hidden_dim, out_horizon)
        self.trend_pred = ComponentPredictor(hidden_dim, out_horizon)

        self.fusion_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
            nn.Softmax(dim=1),
        )

        self.proj_global = ProjectionHead(hidden_dim * 3)
        self.proj_hf = ProjectionHead(hidden_dim)
        self.proj_daily = ProjectionHead(hidden_dim)
        self.proj_trend = ProjectionHead(hidden_dim)

    def encode_components(self, x: torch.Tensor):
        hf, daily, trend = self.decomposer(x)
        f_hf = self.hf_encoder(hf)
        f_daily = self.daily_encoder(daily)
        f_trend = self.trend_encoder(trend)
        return (hf, daily, trend), (f_hf, f_daily, f_trend)

    def component_representations(self, f_hf: torch.Tensor, f_daily: torch.Tensor, f_trend: torch.Tensor):
        z_hf = self.proj_hf(f_hf.mean(dim=(2, 3)))
        z_daily = self.proj_daily(f_daily.mean(dim=(2, 3)))
        z_trend = self.proj_trend(f_trend.mean(dim=(2, 3)))
        z_global = self.proj_global(torch.cat([f_hf.mean(dim=(2, 3)), f_daily.mean(dim=(2, 3)), f_trend.mean(dim=(2, 3))], dim=-1))
        return {"global": z_global, "hf": z_hf, "daily": z_daily, "trend": z_trend}

    def forward(self, x: torch.Tensor):
        (_, _, _), (f_hf, f_daily, f_trend) = self.encode_components(x)

        y_hf = self.hf_pred(f_hf)
        y_daily = self.daily_pred(f_daily)
        y_trend = self.trend_pred(f_trend)

        cat_feat = torch.cat([f_hf, f_daily, f_trend], dim=1)
        gate = self.fusion_attn(cat_feat)  # [B,3,1,1]

        pred_raw = gate[:, 0:1] * y_hf + gate[:, 1:2] * y_daily + gate[:, 2:3] * y_trend
        pred = F.relu(pred_raw)  # 物理一致性：非负功率

        reps = self.component_representations(f_hf, f_daily, f_trend)
        return {
            "pred": pred,
            "pred_raw": pred_raw,
            "gate": gate,
            "reps": reps,
            "components_pred": {"hf": y_hf, "daily": y_daily, "trend": y_trend},
        }
