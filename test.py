import os
import csv
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # simpan grafik ke file tanpa membuka jendela
import matplotlib.pyplot as plt
import seaborn as sn

from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, classification_report)
from Utils.getData import getImageLabel, get_dataset_root, CLASSES
from config import CONFIGS

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def plot_confusion_matrix(cf_matrix, title, path):
    # dinormalisasi per baris (seperti materi): setiap baris = proporsi tebakan untuk satu kelas asli
    row_sum = np.sum(cf_matrix, axis=1)[:, None]
    normalized = np.divide(cf_matrix, row_sum, out=np.zeros_like(cf_matrix, dtype=float), where=row_sum != 0)
    df_cm = pd.DataFrame(normalized, index=CLASSES, columns=CLASSES)

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.xlabel("Prediksi")
    plt.ylabel("Label asli")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def test_one_fold(cfg, model_name, fold, root, device):
    # ---------- Data Test ----------
    test_data = getImageLabel(fold=fold, split='Test', img_size=cfg['img_size'], root=root)
    test_loader = DataLoader(test_data, batch_size=cfg['batch_size'], shuffle=False)
    print(f"  Data: {len(test_data)} test")

    # ---------- Muat model terbaik fold ini ----------
    checkpoint_path = os.path.join("Checkpoints", model_name, f"{model_name}_fold{fold}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} tidak ditemukan. Jalankan train.py --model {model_name} dulu.")

    model = cfg['model'](**cfg.get('model_args', {})).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Model: {checkpoint_path} (epoch {checkpoint['epoch']})")

    # ---------- Prediksi (tanpa belajar) ----------
    prediction, ground_truth, probabilities = [], [], []
    model.eval()
    with torch.no_grad():
        for src, trg in test_loader:
            src = torch.permute(src, (0, 3, 1, 2)).to(device)  # (B, H, W, 3) -> (B, 3, H, W)

            pred = model(src)                                  # log-probabilitas
            probabilities.extend(torch.exp(pred).cpu().numpy())        # probabilitas (untuk ROC-AUC)
            prediction.extend(torch.argmax(pred, dim=1).cpu().numpy()) # kelas tebakan
            ground_truth.extend(torch.argmax(trg, dim=1).numpy())      # kelas asli

    y_true, y_pred, y_prob = np.array(ground_truth), np.array(prediction), np.array(probabilities)
    labels = list(range(len(CLASSES)))

    # ---------- Metrik ----------
    # precision, recall, F1 memakai rata-rata macro karena kelasnya tidak seimbang
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        "f1": f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
    }
    try:
        # ROC-AUC multi-kelas: one-vs-rest, rata-rata macro, dari probabilitas
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro', labels=labels)
    except ValueError as e:
        print(f"  ROC-AUC tidak bisa dihitung: {e}")
        metrics["roc_auc"] = float('nan')

    # ---------- Confusion matrix ----------
    cf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    print("  Confusion matrix (baris = label asli, kolom = prediksi):")
    print(cf_matrix)
    plot_confusion_matrix(cf_matrix, f"{model_name} - Fold {fold} (accuracy {metrics['accuracy']:.3f})",
                          os.path.join("Results", model_name, f"confusion_matrix_fold{fold}.png"))

    # ---------- Laporan per kelas ----------
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASSES, digits=4, zero_division=0)
    with open(os.path.join("Results", model_name, f"classification_report_fold{fold}.txt"), "w") as f:
        f.write(str(report))  # classification_report mengembalikan teks (str) karena output_dict tidak dipakai

    return metrics, cf_matrix


def main():
    parser = argparse.ArgumentParser(description="Testing model klasifikasi Mpox dengan 5-fold cross validation")
    parser.add_argument("--model", required=True, choices=list(CONFIGS.keys()), help="nama model di config.py")
    parser.add_argument("--folds", type=int, default=5, help="jumlah fold yang diuji, mulai dari fold 1 (uji coba: 1)")
    args = parser.parse_args()

    cfg = CONFIGS[args.model]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(os.path.join("Results", args.model), exist_ok=True)

    print(f"Model: {args.model} | device: {device} | fold: 1-{args.folds}")
    root = get_dataset_root()  # cukup sekali untuk semua fold

    rows = []
    total_cf = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for fold in range(1, args.folds + 1):
        print(f"\n===== Fold {fold} =====")
        metrics, cf_matrix = test_one_fold(cfg, args.model, fold, root, device)
        total_cf += cf_matrix
        rows.append([f"fold{fold}"] + [metrics[m] for m in METRICS])
        print("  " + " | ".join(f"{m} = {metrics[m]:.4f}" for m in METRICS))

    # ---------- Rata-rata dan simpangan baku antar-fold ----------
    values = np.array([r[1:] for r in rows], dtype=float)
    mean, std = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    rows.append(["mean"] + list(mean))
    rows.append(["std"] + list(std))

    with open(os.path.join("Results", args.model, "test_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold"] + METRICS)
        writer.writerows(rows)

    # confusion matrix gabungan semua fold yang diuji
    plot_confusion_matrix(total_cf, f"{args.model} - Gabungan {args.folds} fold",
                          os.path.join("Results", args.model, "confusion_matrix_all.png"))

    print(f"\n===== Rata-rata {args.folds} fold =====")
    for i, m in enumerate(METRICS):
        print(f"  {m:9s} = {mean[i]:.4f} ± {std[i]:.4f}")
    print(f"\nSelesai. Metrik di Results/{args.model}/test_metrics.csv, confusion matrix di Results/{args.model}/")


if __name__ == "__main__":
    main()
