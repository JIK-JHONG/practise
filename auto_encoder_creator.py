import os
import cv2  # 使用 OpenCV
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 設定超參數
image_size = 64  # 圖片大小
latent_dim = 64  # 與訓練時保持一致
output_dir = "output_images"  # 輸出圖片的資料夾
os.makedirs(output_dir, exist_ok=True)

# 定義 Autoencoder 模型（需與訓練時一致）
class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(image_size * image_size * 3, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, latent_dim),
            torch.nn.ReLU()
        )
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, image_size * image_size * 3),
            torch.nn.Sigmoid(),
            torch.nn.Unflatten(1, (3, image_size, image_size))
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# 自定義 Dataset 使用 OpenCV 讀取圖片
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # self.image_paths = [
        #     os.path.join(root_dir, file)
        #     for file in os.listdir(root_dir)
        #     if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        # ]
        self.image_paths = []
        for file in os.listdir(root_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                self.image_paths.append(os.path.join(root_dir, file))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 用 OpenCV 讀取圖片
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 轉為 RGB
        if self.transform:
            image = self.transform(image)
        return image, img_path

# 圖片轉換
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

# 載入資料集
test_data_dir = "images"  # 替換為你的資料夾路徑
dataset = ImageFolderDataset(root_dir=test_data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # 一次處理一張圖片

# 載入模型
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load("autoencoder_opencv.pth"))
model.eval()

# 對資料進行推論並儲存輸出
with torch.no_grad():
    for images, paths in dataloader:
        images = images.to(device)
        reconstructions = model(images)

        # 轉回 CPU 並轉換為可視化格式
        output_image = reconstructions.squeeze(0).cpu().numpy()  # 移除批次維度
        output_image = np.transpose(output_image, (1, 2, 0))  # CHW -> HWC
        output_image = (output_image * 255).astype(np.uint8)  # 恢復像素範圍 [0, 255]

        # 保存輸出圖片
        original_filename = os.path.basename(paths[0])
        output_path = os.path.join(output_dir, f"reconstructed_{original_filename}")
        cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

        print(f"Saved reconstructed image to {output_path}")