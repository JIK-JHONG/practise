import os
import cv2  # 改用 OpenCV
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# 超參數設定
batch_size = 16
epochs = 300
learning_rate = 0.001
latent_dim = 64
image_size = 64  # 圖片調整為 64x64

# 自定義 Dataset，使用 OpenCV 讀取圖片
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(subdir, file))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 用 OpenCV 讀取圖片
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 轉為 RGB 格式
        if self.transform:
            image = self.transform(image)
        return image

# 圖片轉換（保持與 PyTorch 相容）
transform = transforms.Compose([
    transforms.ToPILImage(),  # OpenCV 讀取的圖片需轉為 PIL 格式
    transforms.Resize((image_size, image_size)),  # 調整圖片大小
    transforms.ToTensor()  # 轉為 Tensor 並正規化至 [0, 1]
])

# 載入資料
data_dir = "images"  # 替換為你的資料夾路徑
dataset = ImageFolderDataset(root_dir=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定義 Autoencoder 模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size * image_size * 3, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, image_size * image_size * 3),
            nn.Sigmoid(),
            nn.Unflatten(1, (3, image_size, image_size))
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# 建立模型、損失函數與優化器
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# TensorBoard 寫入器
writer = SummaryWriter("log/autoencoder_opencv_example")

# 訓練模型
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images in dataloader:
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # 將訓練損失寫入 TensorBoard
    writer.add_scalar("Training Loss", avg_loss, epoch + 1)

    # 在 TensorBoard 中可視化輸入與輸出圖片
    if (epoch + 1) % 2 == 0:  # 每 2 個 epoch 可視化一次
        model.eval()
        with torch.no_grad():
            sample_images = images[:8]  # 取一個批次的前 8 張圖片
            reconstructions = model(sample_images)

            # 將圖片轉為可視化格式
            inputs = sample_images.cpu().numpy()
            outputs = reconstructions.cpu().numpy()

            writer.add_images("Input Images", inputs, global_step=epoch + 1)
            writer.add_images("Reconstructed Images", outputs, global_step=epoch + 1)

# 儲存模型
torch.save(model.state_dict(), "autoencoder_opencv.pth")
writer.close()

# 使用 TensorBoard 檢視結果
# 在終端機執行以下指令：
# tensorboard --logdir=log/autoencoder_opencv_example