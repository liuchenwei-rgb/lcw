import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
DATA_RATIO = 0.3
MODEL_TYPE = "resnet"
SAVE_PATH = "./fashion_mnist_results"

os.makedirs(SAVE_PATH, exist_ok=True)

if MODEL_TYPE == "inceptionv3":
    input_size = 299
else:
    input_size = 224

transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

train_indices = np.random.choice(
    len(train_dataset), int(len(train_dataset)*DATA_RATIO), replace=False
)
test_indices = np.random.choice(
    len(test_dataset), int(len(test_dataset)*DATA_RATIO), replace=False
)
train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

train_loader = DataLoader(
    train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def build_model(model_type):
    if model_type == "inceptionv3":
        model = models.inception_v3(pretrained=True)
        model.aux_logits = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
    
    elif model_type == "resnet":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
    
    model = model.to(device)
    return model

model = build_model(MODEL_TYPE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "params": {
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "epochs": epochs,
            "data_ratio": DATA_RATIO,
            "model_type": MODEL_TYPE
        }
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            train_total += labels.size(0)
            
            train_bar.set_postfix(loss=loss.item(), acc=train_correct.item()/train_total)
        
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct.double() / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc.item())
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(test_loader, desc=f"Validating Epoch {epoch+1}")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)
                
                val_bar.set_postfix(loss=loss.item(), acc=val_correct.item()/val_total)
        
        epoch_val_loss = val_loss / len(test_loader.dataset)
        epoch_val_acc = val_correct.double() / len(test_loader.dataset)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc.item())
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
    
    return model, history

model, history = train_model(
    model, train_loader, test_loader, criterion, optimizer, EPOCHS
)

model_path = os.path.join(SAVE_PATH, f"{MODEL_TYPE}_fashion_mnist.pth")
torch.save(model.state_dict(), model_path)

history_path = os.path.join(SAVE_PATH, f"{MODEL_TYPE}_history.json")
with open(history_path, 'w') as f:
    json.dump(history, f, indent=4)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label='Train Loss')
plt.plot(history["val_loss"], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label='Train Acc')
plt.plot(history["val_acc"], label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig(os.path.join(SAVE_PATH, f"{MODEL_TYPE}_training_curve.png"))
plt.show()

print("\n================ 实验结果 ================")
print(f"模型类型: {MODEL_TYPE}")
print(f"最终训练准确率: {history['train_acc'][-1]:.4f}")
print(f"最终验证准确率: {history['val_acc'][-1]:.4f}")
print(f"结果保存路径: {SAVE_PATH}")
