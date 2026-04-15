import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# import your dataset
from dataset.dataset import BiomassDataset  
from model.model import SmallCNN

# =============================
# TRAINING LOOP
# =============================
def train(
    data_dir="data",
    batch_size=16,
    epochs=10,
    lr=1e-3,
    num_workers=2
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # DATASET
    # =========================
    train_ds = BiomassDataset(
        data_dir,
        patch_size=64,
        split="train",
        use_ae=True,
        augment=True
    )

    val_ds = BiomassDataset(
        data_dir,
        patch_size=64,
        split="val",
        use_ae=True,
        augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # =========================
    # MODEL
    # =========================
    sample_x, _ = train_ds[0]
    in_channels = sample_x.shape[0]

    model = SmallCNN(in_channels).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"Model input channels: {in_channels}")

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(epochs):

        # ---- train ----
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- val ----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = loss_fn(pred, y)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  train_loss: {train_loss:.4f}")
        print(f"  val_loss:   {val_loss:.4f}")

    # =========================
    # SAVE MODEL
    # =========================
    torch.save(model.state_dict(), "model.pt")
    print("Saved → model.pt")


if __name__ == "__main__":
    train()