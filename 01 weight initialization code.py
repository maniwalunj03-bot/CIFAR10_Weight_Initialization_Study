# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 11:53:37 2025

@author: Manisha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time, json, os
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# 12 layer deep convolution layers without batch norm or skip connections
class DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        layers = []
        in_chanels = 3
        for _ in range(6):   # 6 bolck * 2 conv2d layer = 12 layers
            layers.append(nn.Conv2d(in_chanels, 64, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_chanels = 64
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(64*32*23, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        return x

# Weight Initialization method
def init_weights(model, mode='xeviar'):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            
            if mode == 'zero':
                nn.init.zeros_(m.weight)
            
            elif mode == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            
            elif mode == 'xavier':
                nn.init.xavier_normal_(m.weight)
            
            elif mode == 'kaiming':
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            
            elif mode == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# Training function
def train_model(model, train_loader, test_loader, device, epochs=10):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    history = {"train_loss": [], "test_loss": [], "test_accs": []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        # test
        model.eval()
        correct, total, test_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
        train_loss = running_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)
        test_accs = 100 * correct / total
        
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_accs"].append(test_accs)

        print(f"Epoch {epoch+1:2d}/10 | Train loss: {train_loss:.4f} | Test "
              f"loss: {test_loss:.4f} | Test accuracy: {test_accs:.4f}")
    return history

# Main experiment loop

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # CIFAR 10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])           
    
    train_set = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
        )
    test_set = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
        )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, 
                                              shuffle=False)
    
    INIT_METHODS = ["zero", "normal", "xevier", "kaiming", "orthogonal"]
    results = {}
    
    for mode in INIT_METHODS:
        print("\n" + "=" * 60)
        print(f"Training with initialization : {mode.upper()}")
        print("=" * 60)
        
        model = DeepCNN()
        init_weights(model, mode)
        
        start_time = time.time()
        history = train_model(model, train_loader, test_loader, device)
        end_time = time.time()
        
        history["time_sec"] = end_time - start_time
        results[mode] = history
    
        torch.save(model.state_dict(), f"Model {mode}.pth")
        
    # Save all results
    with open("init_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nTraining completed for all initialization.")
    print("Results saved to: init_results.json")
    print("Model saved as: model_<init>.pth")
    
# Load results

with open("init_results.json", "r") as f:
    results = json.load(f)
    
init_methods = list(results.keys())
epochs = len(next(iter(results.values()))["train_loss"])

# plot train VS test loss (Per Initialization)
plt.figure(figsize=(12, 5))
for mode in init_methods:
    plt.plot(results[mode]["train_loss"], label=f"{mode} Train", linestyle="--")
    plt.plot(results[mode]["test_loss"], label=f"{mode} Test")
plt.title("Loss curve for different weight initialization (DeepCNN")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve_initialization.png", dpi=300)
plt.show()

# Plot accuracy comparison
final_acc = [results[m]["test_accs"][-1] for m in init_methods]
plt.figure(figsize=(12, 7))
sns.barplot(x=init_methods, y=final_acc, palette="viridis")
plt.title("Final test accuracy per initialization")
plt.xlabel("Initializatin Method")
plt.ylabel("Accuracy %")
for i, v in enumerate(final_acc):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("accuracy_comparision_initialization.png", dpi=300)
plt.show()   

# Heatmap of training stability
acc_matrix = [results[m]["test_accs"] for m in init_methods]

plt.figure(figsize=(8, 6))
sns.heatmap(acc_matrix, annot=True, fmt=".1f", cmap="mako",
            xticklabels=[f"E{i+1}" for i in range (epochs)],
            yticklabels=init_methods)
plt.title("Accuracy Progression Heatmap (Epochs X Initialization)")
plt.xlabel("Epochs")
plt.ylabel("Initialization")
plt.tight_layout()
plt.savefig("accuracy_heatmap_initialization.png", dpi=300)
plt.show()

# print summary table
print("\nFinal summary table")
print("-" * 40)

for m in init_methods:
    best_acc = max([results[m]["test_accs"]])
    time_min = results[m]["time_sec"] / 60
    print(f"{m.upper():<12} | Final acc: {results[m]['test_accs'][-1]:.2f}%"
          f"| Best: {best_acc:.2f} | Time: {time_min:.1f} min")