import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_ch4 import Chapter4WindPowerModel


class WindFarmDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        # x:[B,T,N,C], y:[B,T,N,1] -> [B,C,N,T]
        self.x = torch.tensor(data["x"], dtype=torch.float32).permute(0, 3, 2, 1)
        self.y = torch.tensor(data["y"], dtype=torch.float32).permute(0, 3, 2, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def masked_mae(pred: torch.Tensor, target: torch.Tensor):
    mask = (target > 0).float()
    mask = mask / (mask.mean() + 1e-6)
    return (torch.abs(pred - target) * mask).mean()


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2):
    b = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B,D]
    sim = torch.mm(z, z.t()) / temperature

    self_mask = torch.eye(2 * b, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)

    pos_idx = torch.cat([
        torch.arange(b, 2 * b, device=z.device),
        torch.arange(0, b, device=z.device),
    ])
    return F.cross_entropy(sim, pos_idx)


def make_physics_aware_views(model: Chapter4WindPowerModel, x: torch.Tensor, hf_noise_std: float = 0.03):
    """
    分量级语义干预增强：
    - view1: 高频分量加噪（保持趋势/周期）
    - view2: 趋势分量跨样本互换（保持高频/周期）
    """
    with torch.no_grad():
        hf, daily, trend = model.decomposer(x)

    # view1: 高频扰动
    hf_aug = hf + torch.randn_like(hf) * hf_noise_std
    view1 = hf_aug + daily + trend

    # view2: 趋势互换
    perm = torch.randperm(x.size(0), device=x.device)
    trend_swap = trend[perm]
    view2 = hf + daily + trend_swap

    return view1, view2


def physics_consistency_loss(pred: torch.Tensor, max_ramp: float = 0.25):
    """
    物理一致性约束：
    1) 非负（pred经relu后通常满足，这里保留稳健项）
    2) 相邻时刻爬坡率限制
    """
    non_negative_penalty = F.relu(-pred).mean()

    if pred.shape[-1] > 1:
        ramp = torch.abs(pred[..., 1:] - pred[..., :-1])
        ramp_penalty = torch.mean(F.relu(ramp - max_ramp) ** 2)
    else:
        ramp_penalty = torch.tensor(0.0, device=pred.device)

    return non_negative_penalty + ramp_penalty




def component_consistency_loss(model: Chapter4WindPowerModel, x: torch.Tensor, x_aug: torch.Tensor):
    """
    约束增强前后在低频分量(周期/趋势)上的一致性，避免增强破坏物理共性。
    """
    with torch.no_grad():
        _, daily_ref, trend_ref = model.decomposer(x)
    _, daily_aug, trend_aug = model.decomposer(x_aug)
    return F.l1_loss(daily_aug, daily_ref) + F.l1_loss(trend_aug, trend_ref)

def evaluate(model, loader, device):
    model.eval()
    maes = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            maes.append(masked_mae(out["pred"], y).item())
    return float(np.mean(maes))


def train(config):
    device = torch.device(config["device"])

    train_loader = DataLoader(
        WindFarmDataset(os.path.join(config["data_dir"], "train.npz")),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        WindFarmDataset(os.path.join(config["data_dir"], "val.npz")),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    model = Chapter4WindPowerModel(
        in_dim=config["in_dim"],
        hidden_dim=config["hidden_dim"],
        out_horizon=config["out_horizon"],
        period_len=config["period_len"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    best_val = 1e9
    os.makedirs(config["save_dir"], exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        total_loss = 0.0

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            # 监督分支
            out = model(x)
            pred = out["pred"]
            loss_pred = masked_mae(pred, y)

            # 分量级对比分支
            x1, x2 = make_physics_aware_views(model, x, hf_noise_std=config["hf_noise_std"])
            out1, out2 = model(x1), model(x2)

            loss_cl_global = info_nce_loss(out1["reps"]["global"], out2["reps"]["global"], config["temperature"])
            loss_cl_hf = info_nce_loss(out1["reps"]["hf"], out2["reps"]["hf"], config["temperature"])
            loss_cl_daily = info_nce_loss(out1["reps"]["daily"], out2["reps"]["daily"], config["temperature"])
            loss_cl_trend = info_nce_loss(out1["reps"]["trend"], out2["reps"]["trend"], config["temperature"])
            loss_cl = loss_cl_global + 0.5 * (loss_cl_hf + loss_cl_daily + loss_cl_trend)

            # 物理一致性分支
            loss_phy = physics_consistency_loss(pred, max_ramp=config["max_ramp"])
            loss_cons = component_consistency_loss(model, x, x1)

            loss = (
                loss_pred
                + config["lambda_cl"] * loss_cl
                + config["lambda_phy"] * loss_phy
                + config["lambda_cons"] * loss_cons
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(
                total=f"{loss.item():.4f}",
                pred=f"{loss_pred.item():.4f}",
                cl=f"{loss_cl.item():.4f}",
                phy=f"{loss_phy.item():.4f}",
                cons=f"{loss_cons.item():.4f}",
            )

        val_mae = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | train_loss={total_loss / len(train_loader):.4f} | val_mae={val_mae:.4f}")

        if val_mae < best_val:
            best_val = val_mae
            torch.save({"state_dict": model.state_dict(), "config": config}, os.path.join(config["save_dir"], "best_ch4.pth"))


if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data/npz_30min",
        "save_dir": "./checkpoints_ch4",
        "batch_size": 16,
        "epochs": 30,
        "lr": 1e-4,
        "in_dim": 14,
        "hidden_dim": 128,
        "out_horizon": 12,
        "period_len": 48,  # 30分钟采样 -> 48步/天
        "temperature": 0.2,
        "lambda_cl": 0.1,
        "lambda_phy": 0.2,
        "hf_noise_std": 0.03,
        "max_ramp": 0.25,
        "lambda_cons": 0.05,
    }
    train(config)
