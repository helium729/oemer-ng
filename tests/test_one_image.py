import torch
from oemer_ng.models.omr_model import OMRModel
from PIL import Image
import numpy as np

# 加载模型
model = OMRModel(n_channels=1, num_classes=3, mode='segmentation')

# 加载检查点
checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu', weights_only=True) # map_location='cuda' to use GPU

# 提取模型状态字典（处理不同格式）
if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

# 加载测试图像
img = Image.open('path/to/test_image.png').convert('L')  # 灰度图
img_array = np.array(img) / 255.0
img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

# 预测
with torch.no_grad():
    output = model(img_tensor)  # (1, 3, H, W)
    prediction = torch.argmax(output, dim=1).squeeze(0).numpy()  # (H, W)

print(f"预测结果形状: {prediction.shape}")
print(f"唯一值: {np.unique(prediction)}")  # 应该是 [0, 1, 2]
