import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
from dataclasses import dataclass, asdict, fields
# import your dataset
from src.dataset.dataset import BiomassDataset, compute_normalization_stats
from src.model.model import SmallCNN, PointWiseModel



import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class Config:
    """Configuration for the model architecture."""
    model_class: type = PointWiseModel
    batch_size: int = 16
    patch_size: int = 128
    epochs: int = 100
    lr: float = 1e-4

    exp_name: str = "0"
    run_name: str = "run_0"
    project_name: str = "geotessera-biomass"




# =============================
# TRAINING LOOP
# =============================
def train(
    cfg: Config,
    data_dir="data_uniform",
    num_workers=0
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. Initialize wandb
    log_name = f"{cfg.exp_name}_{cfg.run_name}"
    wandb.init(
        project=cfg.project_name,
        name=log_name,
        config=asdict(cfg)
    )



    # =========================
    # DATASET
    # =========================
    # In train.py inside the train() function:
    split_file = None
    print("Computing normalization stats")
    ae_stats = compute_normalization_stats(data_dir, sample_tiles=20, subdir="ae_embeddings", out="norm_stats_ae.json")

    train_ds = BiomassDataset(data_dir, patch_size=cfg.patch_size, split="train", split_ratio=(0.7, 0.15, 0.15), use_ae=True, augment=False)
    val_ds = BiomassDataset(data_dir, patch_size=cfg.patch_size, split="val", split_ratio=(0.7, 0.15, 0.15), use_ae=True, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print("passed dataset and loaders")


    # =========================
    # MODEL
    # =========================
    sample_x, sample_y, sample_name = train_ds[0]
    print(sample_x.shape, (sample_y.squeeze(-1)).shape)
    # print(sample_x, sample_y)
    print(sample_y.unique().numel(), "unique values in y")
    print(sample_x.unique().numel(), "unique values in x")
    print(sample_x.min(), sample_x.max(), sample_x.std())
    in_channels = sample_x.shape[0]

    model = cfg.model_class(in_channels).to(device)
    wandb.watch(model, log_freq=100)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    print(f"Model input channels: {in_channels}")

    # =========================
    # TRAINING LOOP
    # =========================
    pbar_epoch = tqdm(range(cfg.epochs), desc="Overall Progress")


    x_batch, y_batch, _ = next(iter(train_loader))

    print("x batch:", x_batch.shape)  # (B, C, P, P)
    print("y batch:", y_batch.shape)  # (B, P, P)

    B = x_batch.shape[0] ; cols = math.ceil(math.sqrt(B)) ; rows = math.ceil(B / cols)

    fig, axes = plt.subplots(rows, 2*cols, figsize=(4*2*cols, 4*rows))
    if rows == 1: axes = np.expand_dims(axes, axis=0)
    if axes.ndim == 1: axes = np.expand_dims(axes, axis=0)

    for i in range(B):
        row = i // cols ; col = i % cols
        x_vis = x_batch[i].mean(dim=0).cpu().numpy()

        axes[row, 2 * col].imshow(x_vis, cmap='viridis')
        axes[row, 2 * col].axis('off')

        # y_vis = y_batch[i].cpu().numpy()
        y_vis = y_batch[i].squeeze(0).cpu().numpy()

        axes[row, 2 * col + 1].imshow(y_vis, cmap='YlGn')
        axes[row, 2 * col + 1].axis('off')

    plt.tight_layout()
    plt.show()






    # return 0
    for epoch in range(cfg.epochs):

        # ---- train ----
        model.train() ; train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for x,y,_ in train_bar:
            # x,y = xy
            x = x.to(device) ; y = y.to(device).squeeze(-1) #.squeeze(-1) #.squeeze(1)
            print(x,y)
            
            pred = model(x)
            print("pred", pred)
            # print(f"pred {pred}, true {y}")
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("step ok")
            train_loss += loss.item()

            train_bar.set_postfix(batch_loss=f"{loss.item():.4f}")
            wandb.log({"batch_loss": loss.item()})

        avg_train_loss =  train_loss/len(train_loader)

        # ---- val ----
        model.eval() ; val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        with torch.no_grad():
            for x, y, _ in val_bar:
                x = x.to(device) ; y = y.to(device).squeeze(-1)

                pred = model(x)
                loss = loss_fn(pred, y)

                val_loss += loss.item()
                val_bar.set_postfix(val_loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)
        pbar_epoch.set_postfix({
            "T-Loss": f"{avg_train_loss:.4f}", 
            "V-Loss": f"{avg_val_loss:.4f}"
        })

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
 
    # SAVE MODEL    
    ckpt_dir = "checkpoints"
    exp_dir = os.path.join(ckpt_dir, cfg.exp_name)
    run_dir = os.path.join(exp_dir, cfg.run_name)
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(exp_dir): os.makedirs(exp_dir)
    if not os.path.exists(run_dir): os.makedirs(run_dir)

    torch.save(model.state_dict(), run_dir + f"/model.pth")
    artifact = wandb.Artifact('biomass-model', type='model')
    artifact.add_file(run_dir + f"/model.pth")
    wandb.log_artifact(artifact)
    wandb.finish()


def args_extract(parser: argparse.ArgumentParser):
    for field in fields(Config):
        # Determine the type (handling types like 'type' carefully)
        field_type = field.type if field.type != type else None
        parser.add_argument(
            f"--{field.name}", 
            type=field_type, 
            default=field.default
        )

    args = parser.parse_args()

    config_keys = {f.name for f in fields(Config)}


    extra_args = set(vars(args).keys()) - config_keys
    if extra_args:
        print(f"Arguments ignored (not in Config): {', '.join(extra_args)}")


    filtered_args = {k: v for k, v in vars(args).items() if k in config_keys}
    return filtered_args





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    filtered_args = args_extract(parser)
    config = Config(**filtered_args)

    train(cfg=config)

