import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, input_c=3, output=6, image_size=100, **kwargs):
        # **kwargs menampung argumen lama (input_dim, hidden_dim, dropout, device)
        # supaya tidak error kalau train.py masih mengirimnya.
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_c, 20, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3),

            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3),
        )

        # Ukuran flatten dihitung otomatis (100x100 -> 50*9*9 = 4050)
        with torch.no_grad():
            flat_size = self.features(
                torch.zeros(1, input_c, image_size, image_size)
            ).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, output),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = SimpleCNN()
    dummy = torch.rand(1, 3, 100, 100)
    out = model(dummy)
    print("Bentuk input  :", tuple(dummy.shape))
    print("Bentuk output :", tuple(out.shape))                         # (1, 6)
    print("Jumlah peluang:", round(out.exp().sum().item(), 4))         # 1.0
    print("Jumlah parameter:", sum(p.numel() for p in model.parameters()))  # 50876