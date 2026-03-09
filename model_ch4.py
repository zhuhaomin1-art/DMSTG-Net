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

        # 日内周期：时间步按 period_len 分组求均值（不足一组的尾部直接复用）
        daily = torch.zeros_like(x_bt)
        if t >= self.period_len:
            num_full = t // self.period_len
            for k in range(self.period_len):
                idx = torch.arange(k, num_full * self.period_len, self.period_len, device=x.device)
                if idx.numel() > 0:
                    mean_k = x_bt[:, :, idx].mean(dim=-1, keepdim=True)
                    daily[:, :, idx] = mean_k

            # 处理余量
            tail_start = num_full * self.period_len
            if tail_start < t:
                tail_idx = torch.arange(tail_start, t, device=x.device)
                ref_idx = torch.arange(tail_idx.numel(), device=x.device)
                daily[:, :, tail_idx] = daily[:, :, ref_idx]
        else:
            daily = x_bt.mean(dim=-1, keepdim=True).expand_as(x_bt)

        # 高频：去掉低频(趋势+周期)的一部分
        low = 0.5 * trend + 0.5 * daily
        hf = x_bt - low

        def recover(tensor_bt):
            return tensor_bt.reshape(b, n, self.in_channels, t).permute(0, 2, 1, 3)

        return recover(hf), recover(daily), recover(trend)


class TemporalEncoder(nn.Module):
    """轻量级时序编码器（可替换成你第三章的更强时空块）"""

    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


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
        # 取最后 out_horizon 步
        y = self.head(h)
        if y.shape[-1] >= self.out_horizon:
            return y[..., -self.out_horizon :]
        return F.interpolate(y, size=(y.shape[-2], self.out_horizon), mode="nearest")


class ProjectionHead(nn.Module):
    """对比学习投影头。"""

    def __init__(self, in_dim: int, proj_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor):
        z = self.mlp(z)
        return F.normalize(z, dim=-1)


class Chapter4WindPowerModel(nn.Module):
    """
    第四章模型：多粒度分解 + 分量建模 + 对比学习增强。
    """

    def __init__(self, in_dim: int = 14, hidden_dim: int = 128, out_horizon: int = 12, period_len: int = 144):
        super().__init__()
        self.decomposer = MultiGranularityDecomposer(in_channels=in_dim, period_len=period_len)

        self.hf_encoder = TemporalEncoder(in_dim, hidden_dim)
        self.daily_encoder = TemporalEncoder(in_dim, hidden_dim)
        self.trend_encoder = TemporalEncoder(in_dim, hidden_dim)

        self.hf_pred = ComponentPredictor(hidden_dim, out_horizon)
        self.daily_pred = ComponentPredictor(hidden_dim, out_horizon)
        self.trend_pred = ComponentPredictor(hidden_dim, out_horizon)

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
            nn.Softmax(dim=1),
        )

        self.proj_head = ProjectionHead(hidden_dim * 3)

    def encode_components(self, x: torch.Tensor):
        hf, daily, trend = self.decomposer(x)
        h_hf = self.hf_encoder(hf)
        h_daily = self.daily_encoder(daily)
        h_trend = self.trend_encoder(trend)
        return h_hf, h_daily, h_trend

    def aggregate_representation(self, h_hf: torch.Tensor, h_daily: torch.Tensor, h_trend: torch.Tensor):
        # 全局池化得到样本级表示用于对比学习
        z_hf = h_hf.mean(dim=(2, 3))
        z_daily = h_daily.mean(dim=(2, 3))
        z_trend = h_trend.mean(dim=(2, 3))
        z = torch.cat([z_hf, z_daily, z_trend], dim=-1)
        return self.proj_head(z)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, N, T]
        Returns:
            pred: [B, 1, N, out_horizon]
            rep:  [B, D] 对比学习表示
        """
        h_hf, h_daily, h_trend = self.encode_components(x)

        y_hf = self.hf_pred(h_hf)
        y_daily = self.daily_pred(h_daily)
        y_trend = self.trend_pred(h_trend)

        fused_h = torch.cat([h_hf, h_daily, h_trend], dim=1)
        weights = self.fusion_gate(fused_h)

        pred = (
            weights[:, 0:1, :, -y_hf.shape[-1] :] * y_hf
            + weights[:, 1:2, :, -y_daily.shape[-1] :] * y_daily
            + weights[:, 2:3, :, -y_trend.shape[-1] :] * y_trend
        )

        rep = self.aggregate_representation(h_hf, h_daily, h_trend)
        return pred, rep
