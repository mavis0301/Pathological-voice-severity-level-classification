#%% md
# 本檔案使用資料處理後的 Mel 頻譜圖進行訓練與預測

#%% Import 所有必要的庫
import os
import math
import numpy as np
import pandas as pd
import librosa
import scipy.signal
import scipy.io.wavfile
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Conv2D, ReLU, Add, MaxPooling2D, Activation, Multiply, Concatenate
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import audioflux as af
import parselmouth
from parselmouth.praat import call
from focal_loss import SparseCategoricalFocalLoss

#%% HNR副程式
def getHnr(filename, unit="Hertz"):
    sound = parselmouth.Sound(filename)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 60, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    return hnr

#%% 計算CPP參數
def cpp(x, fs, pitch_range):
    frameLen = len(x)
    NFFT = 2 ** (math.ceil(np.log2(frameLen)))
    quef_seq = range(int(np.round_(fs / pitch_range[1])) - 1, int(np.round_(fs / pitch_range[0])))

    HPfilt_b = [1 - 0.97]
    x = scipy.signal.lfilter(HPfilt_b, 1, x)

    SpecdB = 20 * np.log10(np.abs(np.fft.fft(x)))
    ceps = 20 * np.log10(np.abs(np.fft.fft(SpecdB)))

    ceps_lim = ceps[quef_seq]
    ceps_max = np.max(ceps_lim)
    max_index = np.argmax(ceps_lim)
    p = np.polyfit(quef_seq, ceps_lim, 1)
    ceps_norm = np.polyval(p, quef_seq[max_index])

    return ceps_max - ceps_norm

#%% 聲壓計算 (SPL)
class soundBase:
    def __init__(self, path):
        self.path = path

    def audioread(self, formater='sample'):
        fs, data = scipy.io.wavfile.read(self.path)
        if formater == 'sample':
            data = data / (2 ** (16 - 1))
        return data, fs

    def SPL(self, data, fs, frameLen=50, isplot=False):
        M = fs * frameLen // 1000
        frame_count = len(data) // M
        frames = np.reshape(data[:frame_count * M], (frame_count, M))
        pa = np.sqrt(np.mean(frames ** 2, axis=1))
        p0 = 2e-5
        spls = 20 * np.log10(pa / p0)
        if isplot:
            plt.plot(spls)
            plt.show()
        return np.sum(spls > (0.8 * np.max(spls))) * (frameLen / 1000.0)

#%% Spatial Attention Block
def spatialAttention_eff(input_feature, kernel_size=7, name=""):
    avg_pool = tf.reduce_mean(input_feature, axis=3, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=3, keepdims=True)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=112, kernel_size=kernel_size, strides=1, padding='same', use_bias=False, dilation_rate=2)(concat)
    cbam_feature = Activation('sigmoid')(cbam_feature)
    return Multiply()([input_feature, cbam_feature])

#%% 殘差區塊 (Residual Block)
def residual_block_eff(x, filters, conv_num=3, activation="relu"):
    s = Conv2D(filters, 1, padding="same", dilation_rate=12)(x)
    for _ in range(conv_num - 1):
        x = Conv2D(filters, 3, padding="same", dilation_rate=12)(x)
        x = ReLU()(x)
        x = Conv2D(filters, 3, padding="same", dilation_rate=12)(x)
        x = Add()([x, s])
        x = ReLU()(x)
    return MaxPooling2D(pool_size=3, strides=2)(x)

#%% 取得檔案名稱
def get_full_filenames(df, folder_path):
    full_filenames = []
    for short_filename in df['ID']:
        full_path = os.path.join(folder_path, f"{short_filename}.wav")
        if os.path.exists(full_path):
            full_filenames.append(full_path)
        else:
            print(f"Warning: {full_path} does not exist.")
    return np.array(full_filenames)

#%% Rule-based 篩選
def rule3(fileList, labels, nowDate, fold, hnr_threshold=10, cpp_threshold=10, total_threshold=50, verbose=True):
    rule = []
    for file, i in tqdm(zip(fileList, labels), total=len(fileList)):
        hnr_ = getHnr(file)
        fs, signal = scipy.io.wavfile.read(file)
        cpp_ = cpp(x=signal, fs=fs, pitch_range=[60, 333.3])
        sb = soundBase(file)
        data, fs = sb.audioread()
        total_ = sb.SPL(data, fs)

        if hnr_ < hnr_threshold or cpp_ < cpp_threshold or total_ < total_threshold:
            rule.append(file)
    return np.array(rule)

#%% 處理音訊
def process_audio(fileList, sample_rate):
    audio = []
    for file in fileList:
        data, _ = librosa.load(file, sr=sample_rate)
        data = librosa.util.normalize(data)
        if data.shape[0] <= 144000:
            data = np.pad(data, (0, 144000 - data.shape[0]), mode='constant')
        else:
            data = data[:144000]
        audio.append(data)
    return np.array(audio)

#%% 建立EfficientNetB4 模型
def create_model():
    model_mels = EfficientNetB4(include_top=False, weights='imagenet', input_shape=(184, 559, 3), pooling='max')
    layer = tf.keras.models.Model(inputs=model_mels.input, outputs=model_mels.get_layer('block4c_project_conv').output)
    x = layer.output
    x = Dropout(0.25)(x)
    x_sp = spatialAttention_eff(x, kernel_size=4, name="")
    x = residual_block_eff(x_sp, 8, 3)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(1280, kernel_regularizer=regularizers.l2(0.001))(x)
    x = gelu(x)
    x = Dropout(0.1)(x)
    x = Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)
    x = gelu(x)
    output = Dense(units=5, activation='softmax')(x)
    return tf.keras.models.Model(inputs=model_mels.input, outputs=output)

#%% GELU激活函數
def gelu(x):
    return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))

#%% 主程式
if __name__ == "__main__":
    nowDate = '0910'
    sample_rate = 16000
    wav_path = "E13_0705data/"
    train_df = pd.read_csv('2_fold_data_set/two_fold_1.csv', dtype={'ID': str})
    test_df = pd.read_csv('2_fold_data_set/two_fold_2.csv', dtype={'ID': str})
    train_df['ID'] = train_df['ID'].astype(str).str.zfill(4)
    test_df['ID'] = test_df['ID'].astype(str).str.zfill(4)

    X_train_filename = get_full_filenames(train_df, wav_path)
    y_train_label = np.array(train_df['E'])
    X_test_filename = get_full_filenames(test_df, wav_path)
    y_test_label = np.array(test_df['E'])

    # 規則篩選並處理音訊
    rule3TrainList = rule3(X_train_filename, y_train_label, nowDate+'_train', fold=0)
    rule3TestList = rule3(X_test_filename, y_test_label, nowDate+'_test', fold=0)
    train_audio = process_audio(X_train_filename, sample_rate)
    test_audio = process_audio(X_test_filename, sample_rate)

    # 訓練 Mel 頻譜
    X = []
    X1 = []
    for i in range(train_audio.shape[0]):
        spec_obj = af.MelSpectrogram(num=184, samplate=sample_rate, radix2_exp=10)
        spec_arr = spec_obj.spectrogram(train_audio[i])
        spec_dB_arr = af.utils.power_to_db(spec_arr)
        X.append(spec_dB_arr)
    for i in range(test_audio.shape[0]):
        spec_obj = af.MelSpectrogram(num=184, samplate=sample_rate, radix2_exp=10)
        spec_arr = spec_obj.spectrogram(test_audio[i])
        spec_dB_arr = af.utils.power_to_db(spec_arr)
        X1.append(spec_dB_arr)

    MelS_x_train = np.stack((X, X, X), axis=3)
    MelS_x_val = np.stack((X1, X1, X1), axis=3)

    # 創建模型並訓練
    model_mels = create_model()
    lr = 0.00003
    class_weight = (0.3, 0.3, 0.3, 0.1)
    model_mels.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                       loss=SparseCategoricalFocalLoss(gamma=2, class_weight=class_weight),
                       metrics=['accuracy'])
    batch_size = 16
    epoch = 60

    # 訓練模型
    model_history = model_mels.fit(MelS_x_train, y_train_label,
                                   validation_data=(MelS_x_val, y_test_label),
                                   batch_size=batch_size, epochs=epoch)

    # 預測並顯示結果
    mfcc_predict = model_mels.predict(MelS_x_val)
    mfcc_pred = np.argmax(mfcc_predict, axis=-1)
    report = classification_report(y_test_label, mfcc_pred)
    print(report)

    # 繪製混淆矩陣
    conf = confusion_matrix(y_test_label, mfcc_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.show()
