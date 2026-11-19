import os
import csv
import time
import argparse

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # simpan grafik ke file tanpa membuka jendela
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import Adam
from Utils.getData import getImageLabel, get_dataset_root
from config import CONFIGS


def train_one_fold(cfg, model_name, fold, epochs, root, device):
    # ---------- Data ----------
    train_data = getImageLabel(fold=fold, split='Train', img_size=cfg['img_size'], root=root)
    valid_data = getImageLabel(fold=fold, split='Valid', img_size=cfg['img_size'], root=root)
    train_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=cfg['batch_size'], shuffle=False)
    print(f"  Data: {len(train_data)} train, {len(valid_data)} valid")

    # ---------- Model baru untuk setiap fold ----------
    model = cfg['model'](**cfg.get('model_args', {})).to(device)
    optimizer = Adam(model.parameters(), lr=cfg['lr'])
    loss_function = nn.NLLLoss()  # pasangan LogSoftmax di akhir model

    checkpoint_path = os.path.join("Checkpoints", model_name, f"{model_name}_fold{fold}.pt")
    best_valid_loss, best_epoch, wait = float('inf'), 0, 0
    history = []  # (epoch, train_loss, valid_loss, valid_acc)

    for epoch in range(epochs):
        start = time.time()

        # ---------- Training ----------
        model.train()
        train_loss = 0
        for src, trg in train_loader:
            src = torch.permute(src, (0, 3, 1, 2)).to(device)  # (B, H, W, 3) -> (B, 3, H, W)
            trg = torch.argmax(trg, dim=1).to(device)          # one-hot -> nomor kelas

            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---------- Validasi (tanpa belajar) ----------
        model.eval()
        valid_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for src, trg in valid_loader:
                src = torch.permute(src, (0, 3, 1, 2)).to(device)
                trg = torch.argmax(trg, dim=1).to(device)

                pred = model(src)
                valid_loss += loss_function(pred, trg).item()
                correct += (pred.argmax(dim=1) == trg).sum().item()
                total += trg.size(0)

        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        valid_acc = correct / total
        history.append((epoch + 1, train_loss, valid_loss, valid_acc))

        # ---------- Simpan model terbaik (valid loss terendah) ----------
        mark = ""
        if valid_loss < best_valid_loss:
            best_valid_loss, best_epoch, wait = valid_loss, epoch + 1, 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, checkpoint_path)
            mark = "  <- disimpan"
        else:
            wait += 1

        print(f"  Epoch {epoch + 1:3d}/{epochs} | train loss = {train_loss:.4f} | valid loss = {valid_loss:.4f} "
              f"| valid acc = {valid_acc:.4f} | {time.time() - start:.1f}s{mark}")

        # ---------- Early stopping (opsional) ----------
        if cfg.get('patience') is not None and wait >= cfg['patience']:
            print(f"  Early stopping: valid loss tidak membaik selama {cfg['patience']} epoch")
            break

    # ---------- Grafik loss ----------
    epochs_ran = [h[0] for h in history]
    plt.figure()
    plt.plot(epochs_ran, [h[1] for h in history], color="#931a00", label='Training')
    plt.plot(epochs_ran, [h[2] for h in history], color="#3399e6", label='Validation')
    plt.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8, label=f'Model terbaik (epoch {best_epoch})')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Fold {fold}")
    plt.legend()
    plt.savefig(os.path.join("Results", model_name, f"training_fold{fold}.png"))
    plt.close()

    # ---------- Riwayat per epoch ----------
    with open(os.path.join("Results", model_name, f"history_fold{fold}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "valid_loss", "valid_acc"])
        writer.writerows(history)

    return best_epoch, best_valid_loss


def main():
    parser = argparse.ArgumentParser(description="Training model klasifikasi Mpox dengan 5-fold cross validation")
    parser.add_argument("--model", required=True, choices=list(CONFIGS.keys()), help="nama model di config.py")
    parser.add_argument("--folds", type=int, default=5, help="jumlah fold yang dilatih, mulai dari fold 1 (uji coba: 1)")
    parser.add_argument("--epochs", type=int, default=None, help="mengganti jumlah epoch di config.py (uji coba: 2)")
    args = parser.parse_args()

    cfg = CONFIGS[args.model]
    epochs = args.epochs if args.epochs is not None else cfg['epoch']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)

    os.makedirs(os.path.join("Checkpoints", args.model), exist_ok=True)
    os.makedirs(os.path.join("Results", args.model), exist_ok=True)

    print(f"Model: {args.model} | device: {device} | fold: 1-{args.folds} | epoch: {epochs}")
    root = get_dataset_root()  # cukup sekali untuk semua fold

    summary = []
    for fold in range(1, args.folds + 1):
        print(f"\n===== Fold {fold} =====")
        start = time.time()
        best_epoch, best_valid_loss = train_one_fold(cfg, args.model, fold, epochs, root, device)
        minutes = (time.time() - start) / 60
        summary.append((fold, best_epoch, best_valid_loss, minutes))
        print(f"  Selesai: model terbaik di epoch {best_epoch} (valid loss {best_valid_loss:.4f}), {minutes:.1f} menit")

    with open(os.path.join("Results", args.model, "training_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "best_epoch", "best_valid_loss", "minutes"])
        writer.writerows(summary)
    print(f"\nSelesai. Model di Checkpoints/{args.model}/, grafik dan riwayat di Results/{args.model}/")


if __name__ == "__main__":
    main()
