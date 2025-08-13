# Example: Local (non-federated) training of ROLANN using synthetic data and ResNet feature extraction.
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor
from federated_rolann.core import ROLANN
from torchvision.models import resnet18, ResNet18_Weights

# Synthetic dataset
ds_train = FakeData(size=32, image_size=(3, 32, 32), num_classes=3, transform=ToTensor())
ds_test = FakeData(size=16, image_size=(3, 32, 32), num_classes=3, transform=ToTensor())

loader_train = DataLoader(ds_train, batch_size=8)
loader_test = DataLoader(ds_test, batch_size=8)

# Feature extractor model
resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet.eval()

rolann = ROLANN(num_classes=3, encrypted=False)
rolann.eval()

# Local training
for x, y in loader_train:
    with torch.no_grad():
        features = resnet(x)
    label = (torch.nn.functional.one_hot(y, num_classes=3) * 0.9 + 0.05)
    rolann.aggregate_update(features, label)

# Evaluation
def evaluate(model, resnet, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            features = resnet(x)
            preds = model(features)
            correct += (preds.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total

print("Acc train:", evaluate(rolann, resnet, loader_train))
print("Acc test:", evaluate(rolann, resnet, loader_test))
