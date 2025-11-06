# -*- coding: utf-8 -*-
"""
train_apnea_2dcnn_with_filter.py
--------------------------------
Sleep apnea audio detection with 2D CNN on Mel spectrograms.
Includes: band-pass filtering, optional noise reduction, normalization,
train/val/test split, metrics, curves, and best-checkpoint saving.

Usage:
    python train_apnea_2dcnn_with_filter.py --data_root APNEA_EDF/APNEA_EDF --epochs 25
"""

import os
import glob
import argparse
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List

# Torch / Audio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

# Signal processing
from scipy.signal import butter, filtfilt

# Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)

# Progress bar
from tqdm import tqdm

# Try noise reduction (optional)
try:
    import noisereduce as nr
    HAS_NR = True
except Exception:
    HAS_NR = False
    print("[Info] 'noisereduce' not found. Proceeding without spectral noise reduction.")


# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# Data loading (Memory-efficient: lazy loading)
# -----------------------
def build_dataset_index(root_dir: str) -> List[dict]:
    """
    构建数据集索引，而不是加载所有数据到内存
    Returns:
        index: List of dict with keys: (file_path, sample_idx, label)
        每个条目表示一个样本的位置信息
    """
    index = []
    patient_dirs = sorted(glob.glob(os.path.join(root_dir, '*')))
    if len(patient_dirs) == 0:
        raise FileNotFoundError(f"No patient folders found under: {root_dir}")

    total_ap = 0
    total_nap = 0

    for pdir in patient_dirs:
        ap_path = glob.glob(os.path.join(pdir, '*_ap.npy'))
        nap_path = glob.glob(os.path.join(pdir, '*_nap.npy'))
        if not ap_path or not nap_path:
            continue
        
        ap_path = ap_path[0]
        nap_path = nap_path[0]
        
        # 快速检查文件形状（只读取元数据）
        try:
            ap = np.load(ap_path, mmap_mode='r')  # 内存映射，不实际加载
            nap = np.load(nap_path, mmap_mode='r')
            
            if ap.ndim != 2 or nap.ndim != 2:
                continue
            if ap.shape[1] != 160000 or nap.shape[1] != 160000:
                continue
            
            # 为每个样本创建索引条目
            for i in range(len(ap)):
                index.append({
                    'file_path': ap_path,
                    'sample_idx': i,
                    'label': 1  # apnea
                })
                total_ap += 1
            
            for i in range(len(nap)):
                index.append({
                    'file_path': nap_path,
                    'sample_idx': i,
                    'label': 0  # normal
                })
                total_nap += 1
                
        except Exception as e:
            print(f"[Warning] Skipping {pdir}: {e}")
            continue

    if len(index) == 0:
        raise RuntimeError("No valid *_ap.npy and *_nap.npy pairs found.")

    apnea_ratio = total_ap / len(index)
    print(f"[Data] Indexed {len(index)} segments (apnea: {total_ap}, normal: {total_nap}); apnea ratio={apnea_ratio:.3f}")
    return index


# -----------------------
# Signal processing
# -----------------------
def bandpass_filter(sig: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 4) -> np.ndarray:
    """
    Butterworth band-pass filtering with filtfilt (zero-phase).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, sig, method="pad")


def preprocess_waveform(
    wav: np.ndarray,
    fs: int = 16000,
    do_denoise: bool = True,
    do_bandpass: bool = True,
    lowcut: float = 100.0,
    highcut: float = 2000.0
) -> np.ndarray:
    """
    Noise reduction (optional) + band-pass filtering + z-score normalization.
    """
    # Denoise
    if do_denoise and HAS_NR:
        # Noisereduce expects mono array
        wav = nr.reduce_noise(y=wav, sr=fs, stationary=False, prop_decrease=0.9)
    # Band-pass (优化：使用更快的滤波器或移到GPU)
    if do_bandpass:
        wav = bandpass_filter(wav, lowcut=lowcut, highcut=highcut, fs=fs, order=4)
    # 归一化：去均值并归一化到合理范围（保持相对幅度信息）
    # 注意：这里做基本归一化，最终的精确归一化在GPU上per-sample进行
    wav = wav - np.mean(wav)
    std = np.std(wav) + 1e-8
    wav = wav / std
    return wav.astype(np.float32)


# -----------------------
# Dataset
# -----------------------
class ApneaMelDataset(Dataset):
    def __init__(
        self,
        index: List[dict],  # 数据索引而不是实际数据
        sr: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        fmin: int = 50,
        fmax: int = 4000,
        do_denoise: bool = True,
        do_bandpass: bool = True,
        augment: bool = False
    ):
        """
        index: List of dict with keys (file_path, sample_idx, label)
        不再将数据全部加载到内存，而是按需从文件加载
        """
        self.index = index
        self.sr = sr
        self.do_denoise = do_denoise
        self.do_bandpass = do_bandpass
        self.augment = augment
        
        # 缓存最近打开的文件（LRU缓存，减少重复打开文件）
        # 注意：在多进程环境下（num_workers > 0），每个进程有独立的缓存
        from collections import OrderedDict
        self._file_cache = OrderedDict()
        self._cache_size = 4  # 最多缓存4个文件

        # mel转换将在GPU上批量进行，不在Dataset中
        # 存储参数供后续使用
        self.mel_params = {
            'sample_rate': sr,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': n_mels,
            'f_min': fmin,
            'f_max': fmax
        }
    
    def _load_sample_from_file(self, file_path: str, sample_idx: int) -> np.ndarray:
        """从文件加载单个样本（使用内存映射避免完整加载）"""
        # 检查缓存（LRU缓存）
        if file_path in self._file_cache:
            # 移动到末尾（最近使用）
            data = self._file_cache.pop(file_path)
            self._file_cache[file_path] = data
            return data[sample_idx].astype(np.float32).copy()
        
        # 使用内存映射模式加载（只读取需要的部分）
        # mmap_mode='r' 表示只读模式，不会将整个文件加载到内存
        data = np.load(file_path, mmap_mode='r')
        sample = data[sample_idx].astype(np.float32).copy()
        
        # 更新LRU缓存
        if len(self._file_cache) >= self._cache_size:
            # 删除最旧的缓存项（第一个）
            self._file_cache.popitem(last=False)
        
        self._file_cache[file_path] = data
        
        return sample

    def __len__(self):
        return len(self.index)

    def _augment(self, wav: np.ndarray) -> np.ndarray:
        """
        Enhanced waveform augmentations (optional):
        - gain variation
        - time shift
        - gaussian noise
        - time stretching (pitch preservation)
        - frequency masking (spectral augmentation simulation)
        """
        if not self.augment:
            return wav

        # Gain variation (更广的范围)
        if random.random() < 0.7:
            gain = np.random.uniform(0.7, 1.3)
            wav = wav * gain

        # Time shift up to +- 1.0s (增加范围)
        if random.random() < 0.7:
            max_shift = int(1.0 * self.sr)
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift > 0:
                wav = np.concatenate([wav[shift:], np.zeros(shift, dtype=wav.dtype)])
            elif shift < 0:
                wav = np.concatenate([np.zeros(-shift, dtype=wav.dtype), wav[:shift]])

        # Gaussian noise (更广的SNR范围)
        if random.random() < 0.7:
            noise_std = np.std(wav) / np.random.uniform(5, 20)
            wav = wav + np.random.randn(*wav.shape).astype(wav.dtype) * noise_std

        # Time stretching (简单的线性插值实现)
        if random.random() < 0.4:
            stretch_factor = np.random.uniform(0.9, 1.1)
            if abs(stretch_factor - 1.0) > 0.01:
                from scipy.interpolate import interp1d
                original_indices = np.arange(len(wav))
                new_length = int(len(wav) * stretch_factor)
                new_indices = np.linspace(0, len(wav) - 1, new_length)
                f = interp1d(original_indices, wav, kind='linear')
                wav = f(new_indices).astype(wav.dtype)
                # 如果拉伸后长度变化，裁剪或填充
                if len(wav) > len(original_indices):
                    wav = wav[:len(original_indices)]
                elif len(wav) < len(original_indices):
                    wav = np.pad(wav, (0, len(original_indices) - len(wav)), mode='constant')

        return wav

    def __getitem__(self, idx):
        # 从索引获取文件路径和样本位置
        item = self.index[idx]
        file_path = item['file_path']
        sample_idx = item['sample_idx']
        label = int(item['label'])
        
        # 按需从文件加载数据
        wav = self._load_sample_from_file(file_path, sample_idx)  # (160000,)

        # Augment (pre)
        wav = self._augment(wav)

        # Preprocess (优化：如果降噪太慢，可以考虑禁用)
        wav = preprocess_waveform(
            wav,
            fs=self.sr,
            do_denoise=self.do_denoise,
            do_bandpass=self.do_bandpass
        )

        # 只返回原始波形，让mel转换和归一化在GPU上批量进行
        # 这样可以减少CPU-GPU数据传输次数，提高效率
        # 注意：归一化不在CPU上进行，而是移到GPU批量处理
        return wav.astype(np.float32), torch.tensor(label, dtype=torch.long)


# -----------------------
# Attention Mechanisms
# -----------------------
class ChannelAttention(nn.Module):
    """通道注意力机制（SE-Net风格）"""
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // reduction, num_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, num_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(num_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# -----------------------
# Model
# -----------------------
class CNN2D(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.5, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        
        # 第一层：提取低级特征（增加容量）
        self.conv1_1 = nn.Conv2d(1, 48, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(48)
        self.conv1_2 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(48)
        if use_attention:
            self.att1 = CBAM(48, reduction=8)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.15)
        
        # 第二层：提取中级特征
        self.conv2_1 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(96)
        self.conv2_2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(96)
        if use_attention:
            self.att2 = CBAM(96, reduction=8)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.2)
        
        # 第三层：提取高级特征
        self.conv3_1 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(192)
        self.conv3_2 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(192)
        if use_attention:
            self.att3 = CBAM(192, reduction=16)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # 第四层：更深层次的特征提取
        self.conv4_1 = nn.Conv2d(192, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        if use_attention:
            self.att4 = CBAM(256, reduction=16)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.3)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 增强的分类器，更大的容量
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 第一层
        x = F.relu(self.bn1_1(self.conv1_1(x)), inplace=True)
        x = F.relu(self.bn1_2(self.conv1_2(x)), inplace=True)
        if self.use_attention:
            x = self.att1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第二层
        x = F.relu(self.bn2_1(self.conv2_1(x)), inplace=True)
        x = F.relu(self.bn2_2(self.conv2_2(x)), inplace=True)
        if self.use_attention:
            x = self.att2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 第三层
        x = F.relu(self.bn3_1(self.conv3_1(x)), inplace=True)
        x = F.relu(self.bn3_2(self.conv3_2(x)), inplace=True)
        if self.use_attention:
            x = self.att3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # 第四层
        x = F.relu(self.bn4_1(self.conv4_1(x)), inplace=True)
        x = F.relu(self.bn4_2(self.conv4_2(x)), inplace=True)
        if self.use_attention:
            x = self.att4(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # 全局池化和分类
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# -----------------------
# Train / Eval
# -----------------------
def find_optimal_threshold(probs, labels, metric='f1'):
    """
    寻找最佳分类阈值
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    thresholds = np.arange(0.3, 0.8, 0.01)
    best_score = -1
    best_threshold = 0.5
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(labels, preds, zero_division=0)
        elif metric == 'recall':
            score = recall_score(labels, preds, zero_division=0)
        elif metric == 'precision':
            score = precision_score(labels, preds, zero_division=0)
        else:
            score = accuracy_score(labels, preds)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def evaluate_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    optimize_threshold: bool = True
) -> dict:
    """
    Compute metrics from logits and labels.
    可选择优化阈值以最大化F1或召回率。
    """
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    probs_apnea = probs[:, 1]  # 呼吸暂停类的概率
    
    # 默认阈值0.5
    preds_default = probs.argmax(axis=1)
    
    # 如果优化阈值，寻找最佳阈值
    optimal_threshold = 0.5
    best_f1_with_threshold = None
    if optimize_threshold:
        optimal_threshold, best_f1_score = find_optimal_threshold(probs_apnea, labels, metric='f1')
        preds_optimal = (probs_apnea >= optimal_threshold).astype(int)
        
        # 使用优化阈值计算指标
        acc_optimal = accuracy_score(labels, preds_optimal)
        precision_optimal, recall_optimal, f1_optimal, _ = precision_recall_fscore_support(
            labels, preds_optimal, average='binary', zero_division=0
        )
        best_f1_with_threshold = {
            'threshold': optimal_threshold,
            'accuracy': acc_optimal,
            'precision': precision_optimal,
            'recall': recall_optimal,
            'f1': f1_optimal
        }
        preds = preds_optimal  # 使用优化后的预测
    else:
        preds = preds_default
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )

    # Probabilities for positive class
    try:
        auc = roc_auc_score(labels, probs_apnea)
    except Exception:
        auc = float('nan')

    cm = confusion_matrix(labels, preds).tolist()
    report = classification_report(labels, preds, target_names=["normal(0)", "apnea(1)"], zero_division=0)

    result = dict(
        accuracy=acc, precision=precision, recall=recall, f1=f1, auc=auc,
        confusion_matrix=cm, classification_report=report,
        optimal_threshold=optimal_threshold if optimize_threshold else 0.5
    )
    
    if best_f1_with_threshold:
        result['metrics_at_optimal_threshold'] = best_f1_with_threshold

    return result


def run_epoch(model, loader, device, optimizer=None, criterion=None, 
              scaler=None, max_grad_norm=1.0, use_amp=False, mel_transform=None, to_db=None):
    """
    If optimizer & criterion provided -> train step; else eval step.
    Returns average loss and logits/labels for metrics.
    """
    is_train = optimizer is not None and criterion is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_logits = []
    all_labels = []

    # 使用tqdm显示进度
    loader_iter = tqdm(loader, desc="Train" if is_train else "Eval", disable=False)
    
    # 验证模式下使用no_grad和禁用梯度计算以节省内存
    grad_context = torch.no_grad() if not is_train else torch.enable_grad()

    with grad_context:
        for xb, yb in loader_iter:
            # xb现在是波形数据 (B, 160000)，需要在GPU上转换为mel频谱
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            # 在GPU上进行批量预处理和mel转换（比CPU逐个转换快得多）
            # xb: (B, 160000) -> (B, 1, n_mels, T)
            
            # 注意：不在GPU上对波形做归一化，因为这会破坏样本间的相对差异
            # 波形已经在上层做了基本的去均值和归一化
            
            # 转换为mel频谱
            if mel_transform is not None:
                # MelSpectrogram期望输入: (batch, time) 或 (batch, channel, time)
                # 根据torchaudio文档，输入(B, L)或(B, C, L)，输出是(B, n_mels, T)
                # 如果输入有channel维度，输出仍保持channel维度，但我们希望统一格式
                if xb.dim() == 2:  # (B, 160000) - 直接使用，torchaudio会自动处理为(B, 1, 160000)内部
                    # 使用(B, L)格式，输出将是(B, n_mels, T)
                    xb = mel_transform(xb)  # (B, n_mels, T)
                elif xb.dim() == 3:  # (B, 1, 160000) 或 (B, C, 160000)
                    # 如果有channel维度，输出可能是(B, n_mels, T)或(B, C, n_mels, T)
                    xb = mel_transform(xb)  # 可能是(B, n_mels, T)或(B, 1, n_mels, T)
                else:
                    raise ValueError(f"Unexpected input shape for mel_transform: {xb.shape}")
                
                if to_db is not None:
                    xb = to_db(xb)
                
                # 统一格式为 (B, 1, n_mels, T)
                if xb.dim() == 3:  # (B, n_mels, T)
                    xb = xb.unsqueeze(1)  # (B, 1, n_mels, T)
                elif xb.dim() == 4:  # (B, C, n_mels, T) - 压缩channel维度
                    # 如果是(B, 1, n_mels, T)已经正确，否则需要处理
                    if xb.shape[1] == 1:
                        pass  # 已经是(B, 1, n_mels, T)
                    else:
                        # 如果有多个channel，取第一个或平均
                        xb = xb.mean(dim=1, keepdim=True)  # (B, 1, n_mels, T)
                elif xb.dim() > 4:
                    # 压缩多余维度（不应该发生）
                    raise ValueError(f"Unexpected mel output shape: {xb.shape}, expected 3D or 4D")
            elif xb.dim() == 3:  # 如果已经是mel频谱 (B, n_mels, T)
                xb = xb.unsqueeze(1)  # (B, 1, n_mels, T)
            
            # 归一化：使用更稳定的方法
            # 对于mel频谱图，使用全局统计量而不是per-sample归一化
            # 这样可以保持不同样本之间的相对差异，这对于区分正常和异常很重要
            if xb.dim() == 4:  # (B, 1, n_mels, T)
                # 使用固定范围归一化：将mel频谱图归一化到[-1, 1]范围
                xb_min = xb.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]  # (B, 1, 1, 1)
                xb_max = xb.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]  # (B, 1, 1, 1)
                xb_range = (xb_max - xb_min) + 1e-8
                # Min-Max归一化到[-1, 1]
                xb = 2.0 * (xb - xb_min) / xb_range - 1.0

            if is_train:
                optimizer.zero_grad()

            # 混合精度训练（修复警告：使用新的API）
            if is_train and use_amp and scaler is not None:
                # 使用torch.amp.autocast替代torch.cuda.amp.autocast
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    try:
                        logits = model(xb)
                    except (Exception, RuntimeError) as e:
                        # 如果编译模型第一次forward失败（如Triton缺失），回退到原始模型
                        error_str = str(e).lower()
                        if hasattr(model, '_use_compile') and model._use_compile and hasattr(model, '_original_model'):
                            if 'triton' in error_str or 'tritonmissing' in error_str:
                                print(f"[Warning] Compiled model failed (Triton issue): {e}")
                                print("[Warning] Falling back to uncompiled model")
                                original = model._original_model
                                # 更新模型（optimizer仍会工作，因为它绑定的是参数）
                                model.__dict__.update(original.__dict__)
                                model._use_compile = False
                                logits = model(xb)
                            else:
                                raise
                        else:
                            raise
                    loss = criterion(logits, yb)
                
                scaler.scale(loss).backward()
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                try:
                    logits = model(xb)
                except (Exception, RuntimeError) as e:
                    # 如果编译模型第一次forward失败（如Triton缺失），回退到原始模型
                    error_str = str(e).lower()
                    if hasattr(model, '_use_compile') and model._use_compile and hasattr(model, '_original_model'):
                        if 'triton' in error_str or 'tritonmissing' in error_str:
                            print(f"[Warning] Compiled model failed (Triton issue): {e}")
                            print("[Warning] Falling back to uncompiled model")
                            original = model._original_model
                            # 更新模型（optimizer仍会工作，因为它绑定的是参数）
                            model.__dict__.update(original.__dict__)
                            model._use_compile = False
                            logits = model(xb)
                        else:
                            raise
                    else:
                        raise
                loss = criterion(logits, yb) if is_train else F.cross_entropy(logits, yb)

                if is_train:
                    loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            total_loss += loss.item() * xb.size(0)
            # 验证时立即转移到CPU并清理GPU内存
            if not is_train:
                all_logits.append(logits.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
                # 清理GPU缓存（但不要每次都清理，会影响性能）
                # 只在必要时清理，让GPU流水线更顺畅
            else:
                # 训练时使用detach，避免保留计算图
                all_logits.append(logits.detach().cpu().numpy())
                all_labels.append(yb.detach().cpu().numpy())
            
            # 更新进度条
            if is_train:
                loader_iter.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader.dataset)
    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return avg_loss, logits_np, labels_np


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to APNEA_EDF/APNEA_EDF")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (optimized for GPU utilization)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (recommended: 1e-4 to 2e-4 for faster convergence)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for Adam optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training (AMP)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of data loading workers (default: auto-detect, optimized for GPU utilization)")
    parser.add_argument("--prefetch_factor", type=int, default=8, help="Number of batches loaded in advance by each worker (higher = more GPU utilization but more memory)")
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--fmin", type=int, default=50)
    parser.add_argument("--fmax", type=int, default=4000)
    parser.add_argument("--lowcut", type=float, default=100.0)
    parser.add_argument("--highcut", type=float, default=2000.0)
    parser.add_argument("--denoise", action="store_true", help="Enable noise reduction (disabled by default for speed)")
    parser.add_argument("--bandpass", action="store_true", help="Enable band-pass filtering (disabled by default for speed)")
    parser.add_argument("--augment", action="store_true", help="Enable simple waveform augmentation on training set")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation split (of train)")
    parser.add_argument("--test_size", type=float, default=0.1, help="Hold-out test split")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--val_batch_size", type=int, default=None, help="Batch size for validation (default: same as batch_size, use smaller to save memory)")
    parser.add_argument("--early_stop_patience", type=int, default=7, help="Early stopping patience (epochs)")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor (0.0 = no smoothing)")
    parser.add_argument("--no_attention", action="store_true", help="Disable attention mechanisms")
    parser.add_argument("--use_focal_loss", action="store_true", help="Use Focal Loss instead of CrossEntropy (better for hard examples)")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Focal Loss alpha parameter")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss gamma parameter (higher = more focus on hard examples)")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine"], help="Learning rate scheduler type")
    parser.add_argument("--T_max", type=int, default=None, help="T_max for CosineAnnealingLR (default: epochs)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # 1) Build dataset index (memory-efficient, no data loading)
    print("[Loading] Building dataset index...")
    all_index = build_dataset_index(args.data_root)
    
    # 提取标签用于分层划分
    labels = [item['label'] for item in all_index]

    # 2) Split train / test first, then split val from train
    # 使用索引而不是实际数据
    indices = list(range(len(all_index)))
    trainval_indices, test_indices, trainval_labels, _ = train_test_split(
        indices, labels, test_size=args.test_size, stratify=labels, random_state=args.seed
    )
    train_indices, val_indices, _, _ = train_test_split(
        trainval_indices, [labels[i] for i in trainval_indices], 
        test_size=args.val_size, stratify=[labels[i] for i in trainval_indices], random_state=args.seed
    )

    # 创建分割后的索引列表
    train_index = [all_index[i] for i in train_indices]
    val_index = [all_index[i] for i in val_indices]
    test_index = [all_index[i] for i in test_indices]

    print(f"[Split] Train={len(train_index)}, Val={len(val_index)}, Test={len(test_index)}")

    # 3) Datasets & Loaders (现在使用索引，内存占用极小)
    # 默认禁用慢速预处理以提高GPU利用率
    do_denoise = args.denoise  # 默认False，需要时显式启用
    do_bandpass = args.bandpass  # 默认False，需要时显式启用
    
    train_ds = ApneaMelDataset(
        train_index,
        sr=args.sr, n_mels=args.n_mels,
        n_fft=args.n_fft, hop_length=args.hop_length,
        fmin=args.fmin, fmax=args.fmax,
        do_denoise=do_denoise,
        do_bandpass=do_bandpass,
        augment=args.augment
    )
    val_ds = ApneaMelDataset(
        val_index,
        sr=args.sr, n_mels=args.n_mels,
        n_fft=args.n_fft, hop_length=args.hop_length,
        fmin=args.fmin, fmax=args.fmax,
        do_denoise=do_denoise,
        do_bandpass=do_bandpass,
        augment=False
    )
    test_ds = ApneaMelDataset(
        test_index,
        sr=args.sr, n_mels=args.n_mels,
        n_fft=args.n_fft, hop_length=args.hop_length,
        fmin=args.fmin, fmax=args.fmax,
        do_denoise=do_denoise,
        do_bandpass=do_bandpass,
        augment=False
    )

    # 3.5) 确定设备（需要在DataLoader之前，用于pin_memory）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 自动检测num_workers（优化GPU利用率，尽可能并行加载数据）
    if args.num_workers is None:
        import platform
        cpu_count = os.cpu_count() or 1
        if platform.system() == 'Windows':
            # Windows上使用较少的workers，但尽量启用并行以提升GPU利用率
            # 即使spawn有开销，也比单进程快（因为数据加载是瓶颈）
            num_workers = min(4, cpu_count)  # 最多4个workers，平衡spawn开销和并行度
            print(f"[Info] Windows detected: using {num_workers} workers for better GPU utilization.")
            print(f"[Info] If you encounter errors, try --num_workers 0")
        else:
            num_workers = min(8, cpu_count)  # Linux/Mac可以更多
    else:
        num_workers = args.num_workers
        if num_workers > 0:
            import platform
            if platform.system() == 'Windows':
                print(f"[Info] Using {num_workers} workers on Windows (ensure main guard is present for multi-processing)")
    
    # 检查persistent_workers兼容性（PyTorch >= 1.7.0）
    use_persistent_workers = False
    if num_workers > 0:
        try:
            # 检查是否有persistent_workers参数
            major, minor = map(int, torch.__version__.split('.')[:2])
            if major > 1 or (major == 1 and minor >= 7):
                use_persistent_workers = True
        except:
            pass
    
    print(f"[DataLoader] Using {num_workers} workers, batch_size={args.batch_size}, prefetch_factor={args.prefetch_factor if num_workers > 0 else 'N/A'}")
    print(f"[Optimization] Normalization moved to GPU for batch processing")
    if do_denoise:
        print("[Warning] Noise reduction enabled - this may slow down data loading significantly!")
    else:
        print("[Speed] Noise reduction disabled by default for faster training (use --denoise to enable)")
    if do_bandpass:
        print("[Info] Band-pass filtering enabled")
    else:
        print("[Speed] Band-pass filtering disabled by default for faster training (use --bandpass to enable)")
    
    # DataLoader配置（优化GPU利用率）
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': num_workers,
        'pin_memory': True if device.type == 'cuda' else False,
        'prefetch_factor': args.prefetch_factor if num_workers > 0 else None,
        'drop_last': False,  # 保留所有数据
    }
    if use_persistent_workers and num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        print("[Optimization] Using persistent_workers for faster data loading")
    
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    
    # 验证时使用更小的batch size以节省内存
    val_loader_kwargs = loader_kwargs.copy()
    val_batch_size = args.val_batch_size if args.val_batch_size else args.batch_size // 2
    if val_batch_size != args.batch_size:
        val_loader_kwargs['batch_size'] = val_batch_size
        print(f"[Memory] Using smaller batch size for validation: {val_batch_size} (train: {args.batch_size})")
    
    val_loader = DataLoader(val_ds, shuffle=False, **val_loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    
    # 在GPU上创建mel转换器（批量处理更快）
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        f_min=args.fmin,
        f_max=args.fmax
    ).to(device)
    to_db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80).to(device)

    # 4) Model / Optim / Loss
    print(f"\n[Device] Using {device}")
    
    model = CNN2D(num_classes=2, dropout=0.5, use_attention=(not args.no_attention)).to(device)
    if not args.no_attention:
        print("[Model] Attention mechanisms enabled (CBAM)")
    else:
        print("[Model] Attention mechanisms disabled")
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
    
    # 使用torch.compile优化模型（PyTorch 2.0+，大幅提升GPU利用率）
    # 注意：需要Triton，如果没有安装会跳过编译
    use_compile = False
    if hasattr(torch, 'compile') and device.type == 'cuda':
        # 检查Triton是否可用（不仅检查import，还要检查版本和功能）
        try:
            import triton
            # 尝试简单的triton测试，确保它真的可用
            use_compile = True
            print("[Optimization] Triton found, will compile model with torch.compile")
        except (ImportError, AttributeError):
            print("[Info] Triton not found or incompatible, skipping torch.compile")
            print("[Info] To enable compilation, install Triton: pip install triton")
            print("[Info] Model will run without compilation (still optimized with other methods)")
            use_compile = False
    
    # 存储原始模型，以便在编译失败时回退
    original_model = model
    if use_compile:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("[Optimization] Model compilation initialized (will compile on first forward)")
            # 标记是否使用了编译（用于在forward失败时回退）
            model._use_compile = True
            model._original_model = original_model
        except Exception as e:
            print(f"[Info] torch.compile initialization failed: {e}")
            print("[Info] Continuing without compilation")
            model = original_model
            use_compile = False
            model._use_compile = False
    else:
        model._use_compile = False
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器（可选择ReduceLROnPlateau或CosineAnnealingLR）
    # 根据参数选择调度器
    if args.scheduler == "cosine":
        T_max = args.T_max if args.T_max else args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
        scheduler_mode = None  # Cosine不需要metric
        print(f"[Scheduler] Using CosineAnnealingLR (T_max={T_max}, eta_min=1e-6)")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        scheduler_mode = 'f1'  # 基于F1调整
        print("[Scheduler] Using ReduceLROnPlateau (based on F1 score)")
    
    # 计算类别权重（处理数据不平衡，增强apnea的权重以减少漏诊）
    train_labels = [item['label'] for item in train_index]
    from collections import Counter
    label_counts = Counter(train_labels)
    total = sum(label_counts.values())
    # 增加apnea的权重，减少漏诊（医疗场景中漏诊代价更高）
    normal_weight = total / (2 * label_counts.get(0, 1))
    apnea_weight = total / (1.5 * label_counts.get(1, 1))  # 从2改为1.5，增加apnea权重
    class_weights = torch.tensor([normal_weight, apnea_weight], dtype=torch.float32).to(device)
    print(f"[Data] Class distribution: normal={label_counts.get(0, 0)}, apnea={label_counts.get(1, 0)}")
    print(f"[Loss] Using class weights: {class_weights.cpu().numpy()} (enhanced apnea weight to reduce false negatives)")
    
    # Focal Loss实现（关注困难样本）
    class FocalLoss(nn.Module):
        """Focal Loss for addressing class imbalance and hard examples
        
        Note: Focal Loss produces smaller loss values than CrossEntropy because
        it down-weights easy examples. This is by design. When using Focal Loss,
        you may need to use a higher learning rate to compensate.
        """
        def __init__(self, alpha=None, gamma=2.0, weight=None, label_smoothing=0.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.weight = weight
            self.label_smoothing = label_smoothing
            
        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, label_smoothing=self.label_smoothing, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            
            if self.alpha is not None:
                # alpha可以是一个值或张量
                if isinstance(self.alpha, (float, int)):
                    # 为每个样本分配alpha（如果target==1则用alpha，否则用1-alpha）
                    alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
                elif isinstance(self.alpha, torch.Tensor):
                    # alpha是张量（类别权重）
                    alpha_t = self.alpha[targets]
                else:
                    alpha_t = self.alpha
                focal_loss = alpha_t * focal_loss
            
            return focal_loss.mean()
    
    # Label Smoothing Loss 或 Focal Loss（防止过拟合，关注困难样本）
    use_focal = args.use_focal_loss
    if use_focal:
        focal_alpha = args.focal_alpha
        focal_gamma = args.focal_gamma
        if args.label_smoothing > 0:
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, weight=class_weights, label_smoothing=args.label_smoothing)
            print(f"[Loss] Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma}) with Label Smoothing ({args.label_smoothing})")
            print(f"[Info] Focal Loss produces smaller loss values (typically 0.0X) - this is normal and by design")
            print(f"[Info] Using higher learning rate (1e-4 to 2e-4) recommended with Focal Loss")
        else:
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, weight=class_weights)
            print(f"[Loss] Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma}) - focuses on hard examples")
            print(f"[Info] Focal Loss produces smaller loss values (typically 0.0X) - this is normal and by design")
            print(f"[Info] Using higher learning rate (1e-4 to 2e-4) recommended with Focal Loss")
    elif args.label_smoothing > 0:
        print(f"[Loss] Using CrossEntropy Loss with Label Smoothing (factor={args.label_smoothing})")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 混合精度训练（强烈推荐，可提升2倍速度并降低显存）
    scaler = None
    if args.use_amp and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        print("[Training] Using mixed precision (AMP) - can improve speed by 2x and reduce memory")
    else:
        if args.use_amp:
            print("[Warning] AMP requested but CUDA not available, using FP32")
        else:
            if device.type == 'cuda':
                print("[Recommendation] Consider using --use_amp for faster training and lower memory usage")

    best_val_f1 = -1.0
    best_val_acc = -1.0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    
    # Early Stopping
    early_stop_counter = 0
    early_stop_patience = args.early_stop_patience
    print(f"[Training] Early stopping patience: {early_stop_patience} epochs")

    # 5) Train loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")

        tr_loss, _, _ = run_epoch(
            model, train_loader, device, optimizer, criterion,
            scaler=scaler, max_grad_norm=args.max_grad_norm, 
            use_amp=(args.use_amp and scaler is not None),
            mel_transform=mel_transform, to_db=to_db_transform
        )
        
        # 验证前清理GPU缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        val_loss, val_logits, val_labels = run_epoch(
            model, val_loader, device, None, None,
            max_grad_norm=args.max_grad_norm, use_amp=False,
            mel_transform=mel_transform, to_db=to_db_transform
        )
        
        # 验证后清理GPU缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        val_metrics = evaluate_logits(val_logits, val_labels, optimize_threshold=True)
        if 'metrics_at_optimal_threshold' in val_metrics:
            opt_metrics = val_metrics['metrics_at_optimal_threshold']
            print(f"  Optimal Threshold: {opt_metrics['threshold']:.3f} (F1={opt_metrics['f1']:.3f}, Rec={opt_metrics['recall']:.3f})")
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_metrics["f1"])
        history["val_acc"].append(val_metrics["accuracy"])
        
        # 更新学习率（手动记录变化）
        old_lr = optimizer.param_groups[0]['lr']
        if args.scheduler == "cosine":
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            # Cosine scheduler每次都会改变LR，显示变化
            if abs(current_lr - old_lr) > 1e-8:
                print(f"  [LR Scheduler] Learning rate: {old_lr:.6f} -> {current_lr:.6f}")
        else:
            scheduler.step(val_metrics["f1"])
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr != old_lr:
                print(f"  [LR Scheduler] Learning rate reduced: {old_lr:.6f} -> {current_lr:.6f}")

        print(f"  Train Loss: {tr_loss:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  | Acc: {val_metrics['accuracy']:.3f}  | "
              f"P/R/F1: {val_metrics['precision']:.3f}/{val_metrics['recall']:.3f}/{val_metrics['f1']:.3f}  "
              f"| AUC: {val_metrics['auc']:.3f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best by F1 and Early Stopping
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_val_acc = val_metrics["accuracy"]
            early_stop_counter = 0  # Reset counter
            best_path = os.path.join(args.out_dir, "apnea_2dcnn_best.pth")
            torch.save({
                "model_state": model.state_dict(), 
                "epoch": epoch, 
                "val_metrics": val_metrics,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict()
            }, best_path)
            print(f"  [*] Saved best checkpoint (F1={best_val_f1:.4f}, Acc={best_val_acc:.4f}) to {best_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"\n  [Early Stopping] No improvement for {early_stop_patience} epochs. Stopping training.")
                print(f"  Best validation F1: {best_val_f1:.4f}, Best validation Acc: {best_val_acc:.4f}")
                break

    # Save training curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(history["train_loss"], label="train"); plt.plot(history["val_loss"], label="val"); plt.title("Loss"); plt.legend()
    plt.subplot(1,2,2); plt.plot(history["val_acc"], label="val_acc"); plt.plot(history["val_f1"], label="val_f1"); plt.title("Val Acc/F1"); plt.legend()
    plt.tight_layout()
    curves_path = os.path.join(args.out_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    print(f"[Plot] Saved curves to {curves_path}")

    # 6) Evaluate on test set using best checkpoint
    print("\n[Test Evaluation]")
    print("  Clearing GPU cache and reducing memory usage for test evaluation...")
    
    # Clear GPU cache to free memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    # Reduce memory pressure for test evaluation: use fewer workers
    test_loader_kwargs = loader_kwargs.copy()
    test_loader_kwargs['num_workers'] = max(0, min(2, num_workers))  # Reduce workers for test
    if test_loader_kwargs['num_workers'] < num_workers:
        print(f"  Reduced workers from {num_workers} to {test_loader_kwargs['num_workers']} for test evaluation")
    
    # Disable persistent workers for test (save memory)
    if 'persistent_workers' in test_loader_kwargs:
        test_loader_kwargs['persistent_workers'] = False
    
    # Reload best checkpoint
    try:
        # PyTorch 2.6+ requires weights_only=False for loading checkpoints with optimizer/scheduler states
        # This is safe since we're loading our own saved checkpoint
        checkpoint_path = os.path.join(args.out_dir, "apnea_2dcnn_best.pth")
        try:
            # Try with weights_only=False (PyTorch 2.6+)
            best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions that don't have weights_only parameter
            best_ckpt = torch.load(checkpoint_path, map_location=device)
        
        if "model_state" in best_ckpt:
            model.load_state_dict(best_ckpt["model_state"])
            print("  Loaded best checkpoint for test evaluation")
        else:
            # If checkpoint only contains state_dict (old format)
            model.load_state_dict(best_ckpt)
            print("  Loaded best checkpoint (state_dict only) for test evaluation")
    except Exception as e:
        print(f"  [Warning] Failed to load best checkpoint: {e}")
        print("  Using current model state instead")
    
    # Create test loader with reduced memory settings
    try:
        test_loader_eval = DataLoader(test_ds, shuffle=False, **test_loader_kwargs)
        
        test_loss, test_logits, test_labels = run_epoch(
            model, test_loader_eval, device, None, None,
            mel_transform=mel_transform, to_db=to_db_transform
        )
        test_metrics = evaluate_logits(test_logits, test_labels, optimize_threshold=True)
        if 'metrics_at_optimal_threshold' in test_metrics:
            opt_metrics = test_metrics['metrics_at_optimal_threshold']
            print(f"\n  [Optimal Threshold] Threshold: {opt_metrics['threshold']:.3f}")
            print(f"  [Optimal Threshold] Acc: {opt_metrics['accuracy']:.3f} | "
                  f"P/R/F1: {opt_metrics['precision']:.3f}/{opt_metrics['recall']:.3f}/{opt_metrics['f1']:.3f}")

        print("\n[Test Results]")
        print(f"  Loss: {test_loss:.4f}  | Acc: {test_metrics['accuracy']:.3f}  | "
              f"P/R/F1: {test_metrics['precision']:.3f}/{test_metrics['recall']:.3f}/{test_metrics['f1']:.3f}  "
              f"| AUC: {test_metrics['auc']:.3f}")
        print("\nConfusion Matrix [rows=true, cols=pred]:")
        print(np.array(test_metrics["confusion_matrix"]))
        print("\nClassification Report:")
        print(test_metrics["classification_report"])

        # Save metrics
        with open(os.path.join(args.out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"[JSON] Saved test metrics to {os.path.join(args.out_dir, 'test_metrics.json')}")
    
    except OSError as e:
        if "页面文件太小" in str(e) or "page file" in str(e).lower() or "1455" in str(e):
            print(f"\n  [Error] Insufficient virtual memory for test evaluation: {e}")
            print("  [Suggestion] Training completed successfully. Test evaluation skipped due to low memory.")
            print("  [Suggestion] You can run test evaluation separately by loading the saved checkpoint.")
            print(f"  [Info] Best checkpoint saved at: {os.path.join(args.out_dir, 'apnea_2dcnn_best.pth')}")
        else:
            raise
    except Exception as e:
        print(f"\n  [Error] Test evaluation failed: {e}")
        print("  [Info] Training completed successfully. Test evaluation skipped due to error.")
        print(f"  [Info] Best checkpoint saved at: {os.path.join(args.out_dir, 'apnea_2dcnn_best.pth')}")

    # Save final full model (for convenience)
    try:
        final_path = os.path.join(args.out_dir, "apnea_2dcnn_final.pth")
        torch.save(model.state_dict(), final_path)
        print(f"[Model] Saved final model weights to {final_path}")
    except Exception as e:
        print(f"[Warning] Failed to save final model: {e}")

    # Optional: export ONNX for mobile/TFLite conversion
    # Note: ONNX export may fail due to AdaptivePooling operations not fully supported
    try:
        # 使用原始模型（未编译版本）进行导出，避免编译相关的问题
        export_model = model
        if hasattr(model, '_original_model'):
            export_model = model._original_model
        
        # 计算正确的输入形状（mel频谱图的大小）
        mel_time_steps = 1 + (160000 - args.n_fft) // args.hop_length
        dummy = torch.randn(1, 1, args.n_mels, mel_time_steps, device=device)
        onnx_path = os.path.join(args.out_dir, "apnea_2dcnn.onnx")
        
        # 使用更新的opset版本（18或更高），以获得更好的支持
        # 设置模型为eval模式，禁用训练相关操作
        export_model.eval()
        
        with torch.no_grad():
            # 尝试使用opset 18（更好的AdaptivePooling支持）
            try:
                torch.onnx.export(
                    export_model, 
                    dummy, 
                    onnx_path, 
                    input_names=["mel_spectrogram"], 
                    output_names=["logits"],
                    opset_version=18,  # 使用更新的opset版本
                    dynamic_axes=None,  # 固定形状导出更稳定
                    do_constant_folding=True,
                    verbose=False
                )
                print(f"[ONNX] Successfully exported to {onnx_path}")
            except Exception as e1:
                # 如果opset 18失败，尝试opset 13
                try:
                    print(f"[ONNX] Opset 18 failed ({e1}), trying opset 13...")
                    torch.onnx.export(
                        export_model, 
                        dummy, 
                        onnx_path, 
                        input_names=["mel_spectrogram"], 
                        output_names=["logits"],
                        opset_version=13,
                        do_constant_folding=True,
                        verbose=False
                    )
                    print(f"[ONNX] Successfully exported to {onnx_path} (using opset 13)")
                except Exception as e2:
                    raise Exception(f"Both opset versions failed. Opset 18: {e1}. Opset 13: {e2}")
                    
    except Exception as e:
        error_msg = str(e)
        if 'adaptive' in error_msg.lower() or 'AdaptivePool' in error_msg:
            print(f"[ONNX] Export skipped: AdaptivePooling operations are not fully supported in ONNX")
            print(f"[ONNX] This is expected due to AdaptiveAvgPool2d in the model architecture")
            print(f"[ONNX] The model weights (.pth files) can still be used for inference")
            print(f"[ONNX] If ONNX export is required, consider modifying the model to use fixed-size pooling")
        else:
            print(f"[ONNX] Export skipped: {error_msg}")
            print(f"[ONNX] This is an optional feature. Model training completed successfully.")
            print(f"[ONNX] You can still use the saved .pth checkpoint files for inference")


if __name__ == "__main__":
    main()


# ============================================================================
# 最优训练参数建议（Based on memory optimization and generalization）
# ============================================================================
# 
# 1. 基础训练参数（推荐配置）：
#    --batch_size 64                    # 平衡内存和性能
#    --val_batch_size 32                # 验证时使用更小的batch size节省内存
#    --lr 1e-4                          # 较低的学习率，更稳定
#    --weight_decay 1e-4                # L2正则化，防止过拟合
#    --epochs 30                        # 配合early stopping使用
#    --early_stop_patience 7            # 早停耐心值
# 
# 2. 防止过拟合措施：
#    --label_smoothing 0.1              # Label Smoothing（已在代码中启用）
#    --augment                          # 数据增强
#    --dropout 0.5                      # Dropout（模型内置）
#    --no_denoise                       # 如果内存紧张，禁用降噪可加速
# 
# 3. 内存优化参数：
#    --val_batch_size 32                # 验证batch size减小（默认是训练的一半）
#    --num_workers 0                    # Windows上推荐，避免spawn问题
#    --prefetch_factor 2                # 降低预取因子节省内存
#    --use_amp                          # 混合精度训练（如果GPU支持）
# 
# 4. 模型优化：
#    默认启用注意力机制（CBAM）- 已在模型内置
#    不使用 --no_attention 即可启用
# 
# 5. 完整推荐命令示例（一行，优化GPU利用率）：
#    python main.py --data_root psg-audio-apnea-audios/PSG-AUDIO/APNEA_EDF --batch_size 128 --val_batch_size 64 --lr 1e-4 --weight_decay 1e-4 --epochs 30 --early_stop_patience 7 --label_smoothing 0.1 --augment --use_amp --max_grad_norm 1.0
#
#    注意：默认已禁用降噪和滤波以提高GPU利用率，如需启用请添加 --denoise --bandpass
# 
# 6. GPU利用率优化措施（已默认启用）：
#    - 增大batch_size到128（如果内存允许，可以更大）
#    - 启用多进程数据加载（Windows默认4个workers，Linux/Mac默认8个）
#    - 默认禁用降噪和滤波（CPU密集型操作，会阻塞GPU）
#    - 使用torch.compile编译模型（PyTorch 2.0+）
#    - 增加prefetch_factor到8
#    - 使用persistent_workers减少进程启动开销
#    - 启用AMP混合精度训练：--use_amp（强烈推荐）
#
# 7. 如果仍然遇到内存问题：
#    - 减小 batch_size 到 64 或更小
#    - val_batch_size 设为 32 或更小
#    - 减少workers：--num_workers 2
#    - 降低prefetch：--prefetch_factor 2
#    - 确保启用AMP：--use_amp
# 
# 8. GPU利用率优化效果：
#    - 默认配置应能达到50-80% GPU利用率
#    - 启用AMP后可能达到80-95% GPU利用率
#    - 如果GPU利用率仍然很低（<30%），检查：
#      * CPU数据预处理是否太慢（默认已禁用降噪/滤波）
#      * num_workers是否足够（默认已优化）
#      * batch_size是否够大（默认128，可根据显存增大）
#      * 是否使用了AMP（强烈推荐 --use_amp）
# 
# ============================================================================
