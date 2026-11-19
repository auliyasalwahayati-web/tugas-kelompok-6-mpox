import torch
import torch.nn as nn


class SimpleCNNReg(nn.Module):
    """
    Model 2: SimpleCNN dengan regularisasi untuk mengurangi overfitting.
    Perubahan dari Model 1:
      1. BatchNorm2d setelah setiap Conv  -> training lebih stabil, efek regularisasi ringan
      2. Dropout2d di blok konvolusi       -> menjatuhkan seluruh feature map secara acak
      3. Dropout sebelum Linear            -> mencegah fc menghafal data latih
    Bentuk tensor sama seperti Model 1 (input 100x100 -> output 6 kelas).
    """

    def __init__(self, input_c=3, output=6, dropout_conv=0.1, dropout_fc=0.5):
        super().__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(input_c, 20, kernel_size=5),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout2d(dropout_conv),
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(20, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout2d(dropout_conv),
        )

        self.dropout = nn.Dropout(dropout_fc)
        self.fc = nn.Linear(50 * 9 * 9, output)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.convBlock1(x)            # (B, 20, 32, 32)
        x = self.convBlock2(x)            # (B, 50, 9, 9)
        x = self.dropout(torch.flatten(x, 1))
        x = self.fc(x)
        return self.logSoftmax(x)


# ---------------------------------------------------------------------------
# Pelengkap training (bagian terpenting untuk menekan overfitting)
# ---------------------------------------------------------------------------
from torchvision import transforms

# 1. Data augmentation: HANYA untuk data latih, bukan validasi/test
train_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])
eval_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


class EarlyStopping:
    """Hentikan training jika val_loss tidak membaik, simpan bobot terbaik."""

    def __init__(self, patience=7, path="best_model2.pt"):
        self.patience, self.path = patience, path
        self.best, self.counter = float("inf"), 0

    def step(self, val_loss, model):
        if val_loss < self.best:
            self.best, self.counter = val_loss, 0
            torch.save(model.state_dict(), self.path)
            return False
        self.counter += 1
        return self.counter >= self.patience


def build_training(model, lr=1e-3):
    # 2. Weight decay (L2) lewat AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # 3. Turunkan learning rate saat val_loss stagnan
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.NLLLoss()  # cocok dengan LogSoftmax
    return optimizer, scheduler, criterion


# Pola loop training:
#   model.train() -> hitung loss latih
#   model.eval() + torch.no_grad() -> hitung val_loss
#   scheduler.step(val_loss)
#   if early_stopper.step(val_loss, model): break
#   Setelah selesai: model.load_state_dict(torch.load("best_model2.pt"))


if __name__ == "__main__":
    model = SimpleCNNReg()
    dummy = torch.rand(2, 3, 100, 100)
    out = model(dummy)
    print("Bentuk output :", tuple(out.shape))                      # (2, 6)
    print("Jumlah peluang:", out.exp().sum(dim=1))                  # ~1.0
    print("Jumlah parameter:", sum(p.numel() for p in model.parameters()))