import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from model import SimpleCNN  # 确保与模型结构一致

# ===== 配置 =====
MODEL_PATH = "model.pth"
IMAGE_DIR = "./predict"  # 要预测的图片所在目录

# ===== 图像预处理 =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ===== 加载模型 =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ===== 批量读取并预测 =====
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print(f"未找到图片，确保 {IMAGE_DIR} 目录中存在 .jpg/.png 文件")
    exit()

for filename in sorted(image_files):
    path = os.path.join(IMAGE_DIR, filename)
    try:
        image = Image.open(path).convert('RGB')
    except Exception as e:
        print(f"无法打开图片 {filename}: {e}")
        continue

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob > 0.5 else 0
        label = "狗 🐶" if pred == 1 else "猫 🐱"
        print(f"[{filename}] → {label} （置信度: {prob:.4f}）")
