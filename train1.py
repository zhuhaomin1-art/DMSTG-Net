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


def timeseries_augment(x: torch.Tensor, jitter_std: float = 0.02, drop_prob: float = 0.1):
    """生成对比学习视图：噪声 + 随机时间mask。"""
    noise = torch.randn_like(x) * jitter_std
    x_aug = x + noise

    if drop_prob > 0:
        keep = (torch.rand(x.shape[0], 1, 1, x.shape[-1], device=x.device) > drop_prob).float()
        x_aug = x_aug * keep
    return x_aug


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2):
    """SimCLR 风格 InfoNCE。"""
    b = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.mm(z, z.t()) / temperature

    mask = torch.eye(2 * b, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    pos = torch.cat([torch.arange(b, 2 * b, device=z.device), torch.arange(0, b, device=z.device)])
    loss = F.cross_entropy(sim, pos)
    return loss


def masked_mae(pred: torch.Tensor, target: torch.Tensor):
    mask = (target > 0).float()
    mask = mask / (mask.mean() + 1e-6)
    return (torch.abs(pred - target) * mask).mean()


def evaluate(model, loader, device):
    model.eval()
    maes = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred, _ = model(x)
            maes.append(masked_mae(pred, y).item())
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

            pred, rep = model(x)
            loss_pred = masked_mae(pred, y)

            # 对比学习分支
            x1 = timeseries_augment(x)
            x2 = timeseries_augment(x)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss_cl = info_nce_loss(z1, z2, temperature=config["temperature"])

            loss = loss_pred + config["lambda_cl"] * loss_cl

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", pred=f"{loss_pred.item():.4f}", cl=f"{loss_cl.item():.4f}")

        val_mae = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | train_loss={total_loss/len(train_loader):.4f} | val_mae={val_mae:.4f}")

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
        "period_len": 48,  # 若30分钟采样，日内周期=48
        "lambda_cl": 0.1,
        "temperature": 0.2,
    }
    train(config)
