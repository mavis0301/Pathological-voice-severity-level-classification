# %%
import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import audioflux as af
import parselmouth
from parselmouth.praat import call

# 檢查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_loss = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.best_loss = val_loss
        torch.save(model.state_dict(), 'checkpoint.pth')


# 動態調整 batch size 函數
def adjust_batch_size(feature_shape, available_memory, safety_factor=0.8):
    """
    根據可用的 GPU 顯存動態調整 batch size
    :param feature_shape: 每個特徵的形狀 (e.g., (channels, height, width))
    :param available_memory: 可用的顯存大小 (以 bytes 為單位)
    :param safety_factor: 安全係數，防止超過可用顯存，默認為0.8
    :return: 動態調整後的 batch size
    """
    feature_size = torch.prod(torch.tensor(feature_shape)).item() * 4  # 每個浮點數佔用4 bytes
    max_batch_size = int(safety_factor * available_memory / feature_size)

    # 確保 batch size 合理
    batch_size = max(1, max_batch_size)
    return batch_size

# 動態獲取可用顯存並計算 batch size
def get_dynamic_batch_size(sample_feature):
    feature_shape = sample_feature.shape  # 特徵的形狀

    # 獲取可用 GPU 顯存
    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()
    else:
        # 如果沒有 GPU，可用內存設置為一個合理的值（例如 4GB）
        available_memory = 4 * 1024 ** 3  # 4GB

    # 計算 batch size
    batch_size = adjust_batch_size(feature_shape, available_memory)

    print(f"動態調整後的 batch size: {batch_size}")
    return batch_size

# %% Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# %% Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# %% CRNN 模型
class CRNN(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5, lstm_hidden_size=128, lstm_num_layers=2):
        super(CRNN, self).__init__()
        # 卷積部分
        self.conv_layers = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # LSTM部分
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)

        # 全連接層分類
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 卷積層
        x = self.conv_layers(x)
        # 將卷積層輸出 reshape 成 (batch_size, 序列長度, 特徵數)
        x = x.view(x.size(0), -1, x.size(1))
        # LSTM
        x, _ = self.lstm(x)
        # 使用最後一個時間步的輸出作為分類的輸入
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class CustomAudioDataset(Dataset):
    def __init__(self, file_list, labels, sample_rate, max_height, max_width, augment=False):
        self.file_list = file_list
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_height = max_height
        self.max_width = max_width
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]
        features = extract_features(file_path, self.sample_rate, augment=self.augment)
        features = pad_features(features, self.max_height, self.max_width)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return features, label

# %% 提取特徵時計算最大高度和寬度
def calculate_max_dimensions(file_list, sample_rate):
    max_height, max_width = 0, 0
    for file in tqdm(file_list, desc="Calculating max dimensions"):
        features = extract_features(file, sample_rate)
        max_height = max(max_height, features.shape[1])  # H
        max_width = max(max_width, features.shape[2])    # W
    return max_height, max_width

# %% 填充特徵函數
def pad_features(features, max_height, max_width):
    # 填充高度和寬度到最大值
    height_padding = max_height - features.shape[1]
    width_padding = max_width - features.shape[2]

    padded_features = []
    for feature in features:
        padded_feature = np.pad(
            feature,
            ((0, 0), (0, height_padding), (0, width_padding)),
            mode='wrap'  # 使用數據循環填充
        )
        padded_features.append(padded_feature)
    return np.array(padded_features)


# %% 動態調整 Dropout
class DynamicDropout(nn.Module):
    def __init__(self, initial_p=0.5):
        super(DynamicDropout, self).__init__()
        self.dropout_rate = initial_p

    def forward(self, x):
        return nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

    def update_dropout_rate(self, epoch, decay_factor=0.99):
        self.dropout_rate = max(0.1, self.dropout_rate * decay_factor ** epoch)


# %% 數據增強函數
def time_stretch(data, rate=1.0):
    return librosa.effects.time_stretch(data, rate=rate)

def pitch_shift(data, sr, n_steps=0):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)

def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def time_shift(data, shift_max=0.2):
    shift = np.random.randint(len(data) * shift_max)
    augmented_data = np.roll(data, shift)
    return augmented_data

def apply_convolution(data, ir_signal):
    return np.convolve(data, ir_signal, mode='same')

def spec_augment(spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
    spec = spec.copy()
    num_mel_channels = spec.shape[0]
    num_time_steps = spec.shape[1]

    for _ in range(num_mask):
        # 頻率遮擋
        freq_percentage = np.random.uniform(0.0, freq_masking_max_percentage)
        num_freqs_to_mask = int(freq_percentage * num_mel_channels)
        f0 = np.random.randint(0, num_mel_channels - num_freqs_to_mask)
        spec[f0:f0 + num_freqs_to_mask, :] = 0

        # 時間遮擋
        time_percentage = np.random.uniform(0.0, time_masking_max_percentage)
        num_times_to_mask = int(time_percentage * num_time_steps)
        t0 = np.random.randint(0, num_time_steps - num_times_to_mask)
        spec[:, t0:t0 + num_times_to_mask] = 0

    return spec


# %% 更新 extract_features 函數，動態調整 n_fft
def extract_features(file_path, sample_rate, augment=False):
    data, _ = librosa.load(file_path, sr=sample_rate)
    data = librosa.util.normalize(data)

    if augment:
        # 隨機選擇應用增強技術
        if np.random.rand() < 0.5:
            data = time_stretch(data, rate=np.random.uniform(0.8, 1.2))
        if np.random.rand() < 0.5:
            data = pitch_shift(data, sample_rate, n_steps=np.random.randint(-5, 5))
        if np.random.rand() < 0.5:
            data = add_noise(data)
        if np.random.rand() < 0.5:
            data = time_shift(data)
        if np.random.rand() < 0.5:
            ir_signal = np.random.randn(256)  # 用隨機信號進行卷積
            data = apply_convolution(data, ir_signal)

    # 動態調整 n_fft
    n_fft = min(2048, len(data))  # n_fft 不得超過音頻長度

    # Mel 頻譜特徵
    mel_spec_obj = af.MelSpectrogram(num=184, samplate=sample_rate, radix2_exp=10)
    mel_spec_arr = mel_spec_obj.spectrogram(data)
    mel_spec_dB_arr = af.utils.power_to_db(mel_spec_arr)

    # SpecAugment
    if augment and np.random.rand() < 0.5:
        mel_spec_dB_arr = spec_augment(mel_spec_dB_arr)

    # MFCC 特徵
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13, n_fft=n_fft)

    # Chroma 特徵
    chroma = librosa.feature.chroma_stft(y=data, sr=sample_rate, n_fft=n_fft)

    # Spectral Contrast 特徵
    spectral_contrast = librosa.feature.spectral_contrast(y=data, sr=sample_rate, n_fft=n_fft)

    # Zero-Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=data)

    # RMSE
    rmse = librosa.feature.rms(y=data)

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=data, sr=sample_rate)

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=sample_rate)

    # Spectral Flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=data)

    features = [
        mel_spec_dB_arr,
        mfccs,
        chroma,
        spectral_contrast,
        zero_crossing_rate,
        rmse,
        tonnetz,
        spectral_bandwidth,
        spectral_flatness
    ]

    # 獲取最大高度和寬度
    max_height = max(f.shape[0] for f in features)
    max_width = max(f.shape[1] for f in features)

    padded_features = []
    for feature in features:
        # 填充高度和寬度
        height_padding = max_height - feature.shape[0]
        width_padding = max_width - feature.shape[1]

        # 循環填充特徵，保證不使用0或null進行填充，而是用自身數據
        padded_feature = np.pad(
            feature,
            ((0, height_padding), (0, width_padding)),
            mode='wrap'  # 使用數據循環填充
        )
        padded_features.append(padded_feature)

    return np.array(padded_features)


# %% 處理音訊
def process_audio(fileList, sample_rate):
    audio_features = []

    max_height, max_width = 0, 0

    # 首先找到所有特征中的最大高度和宽度
    for file in fileList:
        features = extract_features(file, sample_rate)
        max_height = max(max_height, features.shape[1])  # H
        max_width = max(max_width, features.shape[2])    # W
        audio_features.append(features)

    # 对每个特征进行统一的填充
    padded_audio_features = []
    for features in audio_features:
        # 填充高度和宽度到最大值
        height_padding = max_height - features.shape[1]
        width_padding = max_width - features.shape[2]

        padded_feature = np.pad(
            features,
            ((0, 0), (0, height_padding), (0, width_padding)),
            mode='wrap'  # 使用數據循環填充，而不是常量填充
        )
        padded_audio_features.append(padded_feature)

    return np.array(padded_audio_features)



# %% 訓練與測試
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, patience=5):
    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        scheduler.step()

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Training complete")

# %% 評估模型
def evaluate_model(model, test_loader, criterion=None):
    model.eval()
    y_true, y_pred = [], []
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if criterion:
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    if criterion:
        val_loss /= len(test_loader.dataset)
        return val_loss, accuracy
    else:
        return y_true, y_pred


# %% 主程序
if __name__ == "__main__":
    nowDate = '0910'
    sample_rate = 16000
    wav_path = "training_data/"
    train_df = pd.read_csv('2_fold_data_set/train.csv', dtype={'ID': str})
    val_df = pd.read_csv('2_fold_data_set/val.csv', dtype={'ID': str})
    train_df['ID'] = train_df['ID'].astype(str).str.zfill(4)
    val_df['ID'] = val_df['ID'].astype(str).str.zfill(4)

    X_train_filename = train_df['ID'].apply(lambda x: os.path.join(wav_path, f"{x}.wav")).tolist()
    y_train_label = np.array(train_df['E'])
    X_val_filename = val_df['ID'].apply(lambda x: os.path.join(wav_path, f"{x}.wav")).tolist()
    y_val_label = np.array(val_df['E'])

    # 計算最大高度和寬度（為了填充特徵）
    max_height, max_width = calculate_max_dimensions(X_train_filename + X_val_filename, sample_rate)

    # 創建 Dataset 和 DataLoader
    train_dataset = CustomAudioDataset(X_train_filename, y_train_label, sample_rate, max_height, max_width, augment=True)
    val_dataset = CustomAudioDataset(X_val_filename, y_val_label, sample_rate, max_height, max_width, augment=False)

    # 從一個樣本中獲取特徵，用於計算 batch size
    sample_feature, _ = train_dataset[0]

    # 動態計算 batch size
    batch_size = get_dynamic_batch_size(sample_feature)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化模型、損失函數、優化器等
    model = CRNN(num_classes=5, dropout_rate=0.5, lstm_hidden_size=32, lstm_num_layers=1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 訓練模型
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=200, patience=40)

    # 評估模型
    y_true, y_pred = evaluate_model(model, val_loader)
    report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'], digits=4)
    print(report)
    conf = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.savefig("Confusion_matrix.png")
    torch.save(model.state_dict(), f'{nowDate}_model.pth')
    print(f"Model saved as {nowDate}_model.pth")