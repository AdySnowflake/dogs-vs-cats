import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from torchvision import transforms
from dataset import CatsDogsDataset
from model import SimpleCNN
from tqdm import tqdm


def main():
    # ===== 设置设备 =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练将使用设备：{device}")

    # ===== 数据集准备 =====
    dataset = CatsDogsDataset('./data/train')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # 优化：多进程加载 + 锁页内存
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=64,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # ===== 模型、损失、优化器 =====
    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 混合精度训练（仅 CUDA 时启用）
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

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
            # 混合精度前向传播
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
                with autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += calc_accuracy(outputs, labels)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"[Epoch {epoch+1}] 训练 Loss: {train_loss:.4f}, 准确率: {train_acc:.4f} | 验证 Loss: {val_loss:.4f}, 准确率: {val_acc:.4f}")

    # ===== 模型保存 =====
    torch.save(model.state_dict(), "model2.pth")
    print("✅ 模型已保存为 model2.pth")


if __name__ == '__main__':
    main()
