import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter
import os
import shutil


__ASCII_GREEN__ = "\033[32m"
__ASCII_YELLOW__ = "\033[33m"
__ASCII_RESET__ = "\033[0m" 

# import torch.optim as optim

def check_and_delete_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):  # 檢查資料夾是否存在
        print(f"資料夾 '{folder_path}' 存在，正在刪除...")
        shutil.rmtree(folder_path)  # 刪除資料夾及其中的所有內容
        print("資料夾已成功刪除。")
    else:
        print(f"資料夾 '{folder_path}' 不存在，無需刪除。")
class L1L2_Models:
    def __init__(self, l1_lambda=0.0, l2_lambda=0.0):
        """
        初始化 L1 和 L2 正則化強度
        """
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def __L1__(self, model):
        """
        計算模型參數的 L1 正則化項
        """
        l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
        return self.l1_lambda * l1_norm

    def __L2__(self, model):
        """
        計算模型參數的 L2 正則化項
        """
        l2_norm = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)
        return self.l2_lambda * l2_norm

    def __L1L2__(self, model):
        """
        計算 L1 和 L2 正則化的總和
        """
        return self.__L1__(model) + self.__L2__(model)

    def apply_regularization(self, loss, model):
        """
        將正則化項添加到損失函數中
        """
        regularization_term = self.__L1L2__(model)
        return loss + regularization_term



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience  # 容忍多少個epoh沒有改善
        self.min_delta = min_delta  # 改善的最小幅度
        self.counter = 0  # 用於計數多少輪沒有改善
        self.best_loss = np.inf  # 儲存最佳驗證損失
        self.early_stop = False  # 停止訓練的標誌

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # 如果有改善，重置計數器
        else:
            self.counter += 1  # 如果沒有改善，計數器+1

        if self.counter >= self.patience:
            self.early_stop = True  # 如果超過耐心次數，停止訓練

	# •	patience 參數設定在多少個 epoch 內，如果驗證損失沒有顯著改善（小於 min_delta），則停止訓練。
	# •	min_delta 參數設置了什麼算作“顯著的改善”，即如果驗證損失下降小於此值，也不算改善。
	# •	counter 用來計數連續多少個 epoch 沒有改善，一旦超過 patience 就會停止訓練。
# 生成數據集
def function_generator(x,amp):
    return np.array(amp * np.exp(-x**2))

class GaussianDataset(Dataset):
    def __init__(self, size=256, num_samples=1000):
        self.size = size
        self.num_samples = num_samples
        self.x = np.linspace(-10, 10, size)
        self.amp = np.linspace(1,2,num_samples)
        # self.amp = np.random.uniform(1, 2, num_samples)
        self.data = np.array([amp * np.exp(-self.x**2) for amp in self.amp], dtype=np.float32)
        self.labels = np.array(self.amp, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

# 定義模型
class MyModel(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            # nn.Linear(256, 512),
            # nn.ReLU(),
            nn.Linear(512, 1),
            # nn.ReLU()  # 輸出範圍 [0, oo]
        )

    def forward(self, x):
        return self.model(x)

# 訓練模型
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=5):
    criterion = nn.MSELoss()    #   lose function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 定義學習率：對於權重 \(w\) 使用 0.001，對於偏差 \(b\) 使用 0.01
    # optimizer = optim.Adam([
    #     {'params': model.model[0].weight, 'lr': lr},  # 第一層權重
    #     {'params': model.model[0].bias, 'lr': lr},    # 第一層偏差
    #     {'params': model.model[2].weight, 'lr': lr},  # 第二層權重
    #     {'params': model.model[2].bias, 'lr': lr},    # 第二層偏差
    #     {'params': model.model[4].weight, 'lr': lr},  # 第三層權重
    #     {'params': model.model[4].bias, 'lr': lr}     # 第三層偏差
    # ])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # 每10個epoch將lr縮小為原來的0.1
    train_losses, val_losses = [], []
    early_stopping = EarlyStopping(patience=patience)

    # 記錄模型計算圖 (Graph)
    example_input = torch.randn(batch_size, size).to(device)  # 假設 num_samples 是你的輸入維度
    writer.add_graph(model, input_to_model=example_input)
    writer.add_text("Profile to Amplitude", str(model))
    
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, labels) in enumerate(train_loader):
            # 需注意形狀，data(batchSize,num_samples) , labels(batchSize) 需要再次拓展 1 >> labels(batchSize,1)
            data, labels = data.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 記錄每個 Batch 的損失值到 TensorBoard
            
            writer.add_scalar("Train/Batch_Loss", loss.item(), epoch * len(train_loader) + batch_idx)

        train_losses.append(train_loss / len(train_loader))

        # 驗證
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                # 需注意形狀，data(batchSize,num_samples) , labels(batchSize) 需要再次拓展 1 >> labels(batchSize,1)
                data, labels = data.to(device), labels.to(device).unsqueeze(1)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))
        # 記錄 Epoch 平均損失值到 TensorBoard
        
        writer.add_scalar("Train/Epoch_Loss", train_losses[-1], epoch)
        writer.add_scalar("Val/Epoch_Loss", val_losses[-1], epoch)
        
        lr = optimizer.param_groups[0]['lr']  # Adam 的學習率位於 'param_groups' 中
        writer.add_scalar("Learning Rate", lr, epoch)

        # 記錄模型參數直方圖
        for name, param in model.named_parameters():
            writer.add_histogram(f"Parameters/{name}", param, epoch)

        # 選擇一個樣本輸入並產生輸出波形
        example_data, _ = next(iter(val_loader))
        example_data = example_data[:1].to(device)  # 取出一個樣本
        output_waveform = model(example_data).cpu().detach().numpy()
        # output = model(example_input)
        # writer.add_embedding(output, metadata=None, label_img=None, global_step=epoch)
        # 你可以使用 t-SNE 或其他方法來將高維數據降維進行可視化，這裡我們直接輸出投影結果
        

        # 使用 matplotlib 畫出輸入和輸出波形圖
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # 輸入波形
        
        ax.plot(example_data[0].cpu().numpy().flatten(), label="Input Waveform", color='b')
        # ax.plot(output_waveform.flatten(), label="Input Waveform", color='b')
        ax.set_title("Input Waveform")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        ax.legend()


        plt.tight_layout()

        # 記錄波形圖到 TensorBoard
        writer.add_figure("Waveforms/Input_and_Output", fig, epoch)


        # 調整學習率
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.5f}, Val Loss: {val_losses[-1]:.5f}")
        early_stopping(val_losses[-1])
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    # 繪製 loss
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    # plt.show()
    plt.tight_layout()

    # 記錄波形圖到 TensorBoard
    writer.add_figure("Train / Val Loss ", fig)

    return model

# 儲存與讀取模型
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# 預測函數
def predict(model, x,amp, size=256 ):
    x = np.linspace(-10, 10, size)
    data = np.exp(-x**2) * amp
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        amp_pred = model(data_tensor).item()
    return amp_pred

# 主流程
if __name__ == "__main__":
    start_time = time.time()
    # 設定參數
    size = 256  #   多少點
    batch_size = 64  #   每次取樣多少
    epochs = 256
    lr = 0.001 * 10 
    num_samples = 10 #   多少資料組
    patience = 20 

    device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    # 建立數據集與分割
    dataset = GaussianDataset(size=size, num_samples=num_samples)
    # 繪製前幾個樣本
    x = np.linspace(-10, 10, size)

    plt.figure(figsize=(10, 6))
    for i in range(num_samples):
        data, label = dataset[i]
        plt.plot(x, data.numpy(), label=f"Amp: {label:.2f}")

    plt.title("Gaussian Dataset Samples")
    plt.xlabel("x")
    plt.ylabel("Amplitude * exp(-x^2)")
    plt.legend()
    plt.grid(True)
    plt.show()

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    # 指定存放日誌的路徑
    log_dir = "./log_dir/"
    check_and_delete_folder(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 初始化模型
    model = MyModel(size=size).to(device)

    # 訓練模型
    model = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, patience=patience)
    
    # for p in model.parameters():
    #     print(p)
    



    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"訓練耗時: {round(elapsed_time, 2)} 秒")
    # 儲存模型
    save_model(model, "gaussian_model.pth")

    # 載入模型並推測
    model = load_model(MyModel(size=size).to(device), "gaussian_model.pth")

    # 輸入預測結果序列
    amp_pre = np.linspace(1,2,num_samples)
    x_pre = np.linspace(-10, 10, size)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for amp_pre_set in amp_pre:
        test_amp = predict(model, x_pre ,amp_pre_set, size)    
        # ax.plot(amp_pre_set, test_amp, label=f"Predicted = {amp_pre_set}" , marker='o')
        ax.plot(amp_pre_set, test_amp, label=None , marker='o', color='orange')
        print(f"Predicted Amplitude: {__ASCII_GREEN__}{test_amp:.4f}{__ASCII_RESET__} {__ASCII_YELLOW__}[{amp_pre_set:.4f}({(amp_pre_set-test_amp)/amp_pre_set*100:.2f}%{__ASCII_RESET__})]")
    ax.plot(amp_pre,amp_pre ,linestyle="dashed")
    ax.set_title("Result")
    ax.set_xlabel("True Amplitude")
    ax.set_ylabel("Predicted Amplitude")
    ax.grid(True)
    # ax.legend()
    plt.tight_layout()

    # 記錄波形圖到 TensorBoard
    writer.add_figure("Result", fig)

