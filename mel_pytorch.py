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

# Early stopping class
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

# %% EfficientNetV2
class MultiModalEfficientNetV2WithAttention(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super(MultiModalEfficientNetV2WithAttention, self).__init__()
        self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)  # 修改輸出層

        # 注意力模塊
        self.ca1 = ChannelAttention(256)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(512)
        self.sa2 = SpatialAttention()

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        return x

# %% 動態調整 Dropout
class DynamicDropout(nn.Module):
    def __init__(self, initial_p=0.5):
        super(DynamicDropout, self).__init__()
        self.dropout_rate = initial_p

    def forward(self, x):
        return nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

    def update_dropout_rate(self, epoch, decay_factor=0.99):
        self.dropout_rate = max(0.1, self.dropout_rate * decay_factor ** epoch)

# %% 特徵處理：Mel 頻譜、MFCC、waveform
def extract_features(file_path, sample_rate):
    data, _ = librosa.load(file_path, sr=sample_rate)
    data = librosa.util.normalize(data)

    # Mel 頻譜特徵
    mel_spec_obj = af.MelSpectrogram(num=184, samplate=sample_rate, radix2_exp=10)
    mel_spec_arr = mel_spec_obj.spectrogram(data)
    mel_spec_dB_arr = af.utils.power_to_db(mel_spec_arr)
    mel_shape = mel_spec_dB_arr.shape  # 保存 Mel 頻譜的形狀 (1, H, W)

    # MFCC 特徵
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
    mfccs_resized = np.pad(mfccs, ((0, 0), (0, mel_shape[1] - mfccs.shape[1])), mode='constant')

    # Waveform 特徵
    waveform = np.expand_dims(data[:mel_shape[1]], axis=0)
    waveform_resized = np.pad(waveform, ((0, 0), (0, mel_shape[1] - waveform.shape[1])), mode='constant')

    # 返回調整後的特徵
    return mel_spec_dB_arr, mfccs_resized, waveform_resized

# %% 處理音訊
def process_audio(fileList, sample_rate):
    audio_features = []
    max_height, max_width = 0, 0

    # 確定所有樣本中的最大高度和寬度
    for file in fileList:
        mel_spec, mfccs, waveform = extract_features(file, sample_rate)
        max_height = max(max_height, mel_spec.shape[0], mfccs.shape[0], waveform.shape[0])
        max_width = max(max_width, mel_spec.shape[1], mfccs.shape[1], waveform.shape[1])

    # 將所有特徵統一填充到最大長寬
    for file in fileList:
        mel_spec, mfccs, waveform = extract_features(file, sample_rate)

        mel_spec_padded = np.pad(mel_spec, ((0, max_height - mel_spec.shape[0]), (0, max_width - mel_spec.shape[1])),
                                 mode='wrap')
        mfccs_padded = np.pad(mfccs, ((0, max_height - mfccs.shape[0]), (0, max_width - mfccs.shape[1])),
                              mode='wrap')
        waveform_padded = np.pad(waveform, ((0, max_height - waveform.shape[0]), (0, max_width - waveform.shape[1])),
                                 mode='wrap')

        print(f"mel_spec_padded shape: {mel_spec_padded.shape}, mfccs_padded shape: {mfccs_padded.shape}, waveform_padded shape: {waveform_padded.shape}")

        combined_features = np.stack([mel_spec_padded, mfccs_padded, waveform_padded], axis=0)
        audio_features.append(combined_features)

    return np.array(audio_features)

# %% 訓練與測試
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, patience=5):
    model = model.to(device)
    dynamic_dropout = DynamicDropout(0.5)
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        dynamic_dropout.update_dropout_rate(epoch)

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

# %%
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

# %% MAIN
if __name__ == "__main__":
    nowDate = '0910'
    sample_rate = 16000
    wav_path = "E13_0705data/"
    train_df = pd.read_csv('2_fold_data_set/two_fold_1.csv', dtype={'ID': str})
    val_df = pd.read_csv('2_fold_data_set/two_fold_2.csv', dtype={'ID': str})
    train_df['ID'] = train_df['ID'].astype(str).str.zfill(4)
    val_df['ID'] = val_df['ID'].astype(str).str.zfill(4)

    X_train_filename = train_df['ID'].apply(lambda x: os.path.join(wav_path, f"{x}.wav")).tolist()
    y_train_label = np.array(train_df['E'])
    X_val_filename = val_df['ID'].apply(lambda x: os.path.join(wav_path, f"{x}.wav")).tolist()
    y_val_label = np.array(val_df['E'])

    # 處理音訊，提取多模態特徵
    train_audio = process_audio(X_train_filename, sample_rate)
    val_audio = process_audio(X_val_filename, sample_rate)

    # 轉為 PyTorch tensor
    train_data = torch.tensor(train_audio, dtype=torch.float32)
    train_labels = torch.tensor(y_train_label, dtype=torch.long)
    val_data = torch.tensor(val_audio, dtype=torch.float32)
    val_labels = torch.tensor(y_val_label, dtype=torch.long)

    # DataLoader
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 帶注意力機制和多模態輸入的 EfficientNetV2
    model = MultiModalEfficientNetV2WithAttention(num_classes=5, dropout_rate=0.5)

    # 損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00003, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 訓練包括early stop的耐心
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=200, patience=5)

    # 評估
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
