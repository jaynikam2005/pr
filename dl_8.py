
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 1️⃣ Load MNIST
transform = transforms.ToTensor()
train = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test = DataLoader(datasets.MNIST('.', train=False, transform=transform), batch_size=64)

# 2️⃣ Define simple MLP
class MLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden), nn.ReLU(),
            nn.Linear(hidden, 10)
        )
    def forward(self, x): return self.net(x)

# 3️⃣ Try different hyperparameters
configs = [
    {"lr": 0.01, "hidden": 64, "batch": 64},
    {"lr": 0.001, "hidden": 128, "batch": 128},
]

for i, cfg in enumerate(configs):
    model = MLP(cfg["hidden"])
    opt = optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(f"runs/config_{i}")

    # 4️⃣ Train small epochs
    for epoch in range(2):
        total, correct = 0, 0
        for x, y in train:
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward(); opt.step()
            pred = out.argmax(1)
            correct += (pred == y).sum().item(); total += y.size(0)
        acc = 100 * correct / total
        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)
        print(f"Config {i}, Epoch {epoch+1}: Acc={acc:.2f}%")

writer.close()

