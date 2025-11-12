

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load data
train = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,
    transform=transforms.ToTensor()), batch_size=64, shuffle=True)
test = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False,
    transform=transforms.ToTensor()), batch_size=64)

# Simple MLP
class Net(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.drop = nn.Dropout(0.5) if dropout else nn.Identity()
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

# Train & test function
def run(model, l2=0.0, label="Model"):
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2)
    loss_fn = nn.CrossEntropyLoss()
    losses, accs = [], []
    for epoch in range(3):
        model.train()
        total_loss = 0
        for x, y in train:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        losses.append(total_loss / len(train))

        # Evaluate
        correct = 0
        with torch.no_grad():
            for x, y in test:
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
        acc = 100 * correct / len(test.dataset)
        accs.append(acc)
        print(f"{label} - Epoch {epoch+1}: Loss={losses[-1]:.4f}, Acc={acc:.2f}%")
    return losses, accs

# Run models
l1, a1 = run(Net(), label="No Reg")
l2, a2 = run(Net(), l2=0.001, label="L2 Reg")
l3, a3 = run(Net(dropout=True), label="Dropout")

# Plot
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(l1, label='No Reg')
plt.plot(l2, label='L2 Reg')
plt.plot(l3, label='Dropout')
plt.title('Training Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(a1, label='No Reg')
plt.plot(a2, label='L2 Reg')
plt.plot(a3, label='Dropout')
plt.title('Accuracy')
plt.legend()
plt.show()

