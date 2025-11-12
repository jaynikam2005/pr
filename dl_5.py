
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms

# Data (with simple augmentation)
tfm = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])
train = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=tfm), batch_size=64, shuffle=True)
test = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor()), batch_size=64)

# MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(32*32*3,256), nn.Linear(256,10)
    def forward(self,x):
        x=x.view(-1,32*32*3)
        return self.fc2(torch.relu(self.fc1(x)))

# CNN model (2 conv + FC)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,16,3,padding=1)
        self.conv2=nn.Conv2d(16,32,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc=nn.Linear(32*8*8,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=x.view(-1,32*8*8)
        return self.fc(x)

# Train + test
def train_test(model,name):
    opt=optim.Adam(model.parameters(),lr=0.001)
    loss_fn=nn.CrossEntropyLoss()
    for e in range(2): # 2 epochs = fast
        for x,y in train:
            opt.zero_grad(); loss=loss_fn(model(x),y)
            loss.backward(); opt.step()
    correct=0
    with torch.no_grad():
        for x,y in test:
            correct+=(model(x).argmax(1)==y).sum().item()
    print(f"{name} Accuracy: {100*correct/len(test.dataset):.2f}%")

# Run both
train_test(MLP(), "MLP")
train_test(CNN(), "CNN")

