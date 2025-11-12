

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1️⃣ Load MNIST data
transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# 2️⃣ Build the Neural Network
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x): return self.model(x)

net = MLP()
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), lr=0.001)

# 3️⃣ Train the Network
epochs = 5
losses, accs = [], []

for e in range(epochs):
    net.train()
    total_loss = 0
    for X, y in trainloader:
        opt.zero_grad()
        out = net(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    losses.append(total_loss / len(trainloader))

    # test accuracy
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in testloader:
            preds = net(X).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total
    accs.append(acc)
    print("Epoch:", e+1, "| Loss:", round(losses[-1], 4), "| Accuracy:", round(acc, 2), "%")

# 4️⃣ Plot loss and accuracy
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(losses, 'r', label='Loss'); plt.title('Training Loss'); plt.legend()
plt.subplot(1,2,2)
plt.plot(accs, 'b', label='Accuracy'); plt.title('Test Accuracy'); plt.legend()
plt.show()

#Fahion
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1️⃣ Load MNIST data
transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# 2️⃣ Build the Neural Network
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x): return self.model(x)

net = MLP()
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), lr=0.001)

# 3️⃣ Train the Network
epochs = 5
losses, accs = [], []

for e in range(epochs):
    net.train()
    total_loss = 0
    for X, y in trainloader:
        opt.zero_grad()
        out = net(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    losses.append(total_loss / len(trainloader))

    # test accuracy
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in testloader:
            preds = net(X).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total
    accs.append(acc)
    print("Epoch:", e+1, "| Loss:", round(losses[-1], 4), "| Accuracy:", round(acc, 2), "%")

# 4️⃣ Plot loss and accuracy
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(losses, 'r', label='Loss'); plt.title('Training Loss'); plt.legend()
plt.subplot(1,2,2)
plt.plot(accs, 'b', label='Accuracy'); plt.title('Test Accuracy'); plt.legend()
plt.show()

