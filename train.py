import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import CatsDogsDataset
from model import SimpleCNN
from tqdm import tqdm

# ===== 设置设备 =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"训练将使用设备：{device}")

# ===== 数据集准备 =====
dataset = CatsDogsDataset('./data/train')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# ===== 模型、损失、优化器 =====
model = SimpleCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ===== 准确率函数 =====
def calc_accuracy(preds, labels):
    preds = torch.sigmoid(preds)  # 将输出映射到 [0, 1]
    preds = (preds > 0.5).float()  # 二值化
    return (preds == labels).float().mean().item()

# ===== 训练循环 =====
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_acc = 0
    for imgs, labels in tqdm(train_loader, desc=f"[训练] Epoch {epoch+1}"):
        imgs, labels = imgs.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += calc_accuracy(outputs, labels)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # ===== 验证 =====
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_acc += calc_accuracy(outputs, labels)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    print(f"[Epoch {epoch+1}] 训练 Loss: {train_loss:.4f}, 准确率: {train_acc:.4f} | 验证 Loss: {val_loss:.4f}, 准确率: {val_acc:.4f}")

# ===== 模型保存 =====
torch.save(model.state_dict(), "model.pth")
print("✅ 模型已保存为 model.pth")
