

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load data
train = datasets.FashionMNIST('.', train=True, download=True, transform=transforms.ToTensor())
test = datasets.FashionMNIST('.', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=1000)

# Simple MLP
class MLP(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            act,
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x.view(-1, 784))

# Train function
def train_model(act):
    model = MLP(act)
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    losses, accs = [], []
    for epoch in range(3):   # few epochs for speed
        for X, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
        losses.append(loss.item())
        correct = sum((model(X).argmax(1)==y).sum().item() for X,y in test_loader)
        accs.append(100*correct/len(test))
    return losses, accs

# Compare activations
acts = {'Sigmoid': nn.Sigmoid(), 'Tanh': nn.Tanh(), 'ReLU': nn.ReLU()}
results = {n: train_model(a) for n,a in acts.items()}

# Plot
for name, (loss, acc) in results.items():
    plt.plot(acc, label=name)
plt.title("Test Accuracy Comparison")
plt.xlabel("Epochs"); plt.ylabel("Accuracy (%)")
plt.legend(); plt.show()

