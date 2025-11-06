# -*- coding: utf-8 -*-
"""
predict_microphone.py
---------------------
实时麦克风睡眠呼吸暂停识别程序

功能：
- 从麦克风实时录音（10秒片段，16kHz采样率）
- 使用训练好的模型进行实时推理
- 显示检测结果（正常/呼吸暂停）和置信度

用法:
    python predict_microphone.py --model_path outputs/apnea_2dcnn_best.pth
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Tuple, Optional

# 音频捕获库
HAS_SOUNDDEVICE = False
HAS_PYAUDIO = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    try:
        import pyaudio
        HAS_PYAUDIO = True
    except ImportError:
        pass

if not HAS_SOUNDDEVICE and not HAS_PYAUDIO:
    print("[警告] 未检测到音频捕获库，请在运行前安装:")
    print("  pip install sounddevice  (推荐)")
    print("  或 pip install pyaudio")

# 信号处理
from scipy.signal import butter, filtfilt

# 尝试导入降噪库（可选）
try:
    import noisereduce as nr
    HAS_NR = True
except ImportError:
    HAS_NR = False
    print("[信息] 'noisereduce' 未安装，将跳过降噪步骤")


# ==================== 从 main.py 复制的模型和预处理函数 ====================

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


class CNN2D(nn.Module):
    """2D CNN模型用于睡眠呼吸暂停分类（与训练时一致）"""
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


def bandpass_filter(sig: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 4) -> np.ndarray:
    """Butterworth带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, sig, method="pad")


def preprocess_waveform(
    wav: np.ndarray,
    fs: int = 16000,
    do_denoise: bool = False,  # 默认关闭，与训练时一致
    do_bandpass: bool = False,  # 默认关闭，与训练时一致
    lowcut: float = 100.0,
    highcut: float = 2000.0
) -> np.ndarray:
    """预处理音频波形：降噪（可选）+ 带通滤波 + 归一化
    
    注意：默认不启用降噪和滤波，与训练时保持一致。
    如果训练时启用了这些选项，预测时也应该启用。
    """
    # 降噪
    if do_denoise and HAS_NR:
        wav = nr.reduce_noise(y=wav, sr=fs, stationary=False, prop_decrease=0.9)
    
    # 带通滤波
    if do_bandpass:
        wav = bandpass_filter(wav, lowcut=lowcut, highcut=highcut, fs=fs, order=4)
    
    # Z-score归一化（与训练时一致）
    wav = wav - np.mean(wav)
    std = np.std(wav) + 1e-8
    wav = wav / std
    
    return wav.astype(np.float32)


def audio_to_melspectrogram(
    wav: np.ndarray,
    sr: int = 16000,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 512,
    fmin: int = 50,
    fmax: int = 4000
) -> torch.Tensor:
    """将音频转换为Mel频谱图（与训练时完全一致）"""
    # 转换为torch tensor
    wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)  # (1, L)
    
    # Mel频谱图转换（确保参数与训练时完全一致）
    # 注意：torchaudio的MelSpectrogram默认center=True，这会在两端填充n_fft//2
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        center=True,  # 显式指定，与训练时一致（默认值）
        pad_mode='reflect'  # 显式指定填充模式
    )
    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    
    mel = mel_transform(wav_t)  # (1, n_mels, T)
    mel_db = to_db(mel)
    
    # 调试信息：打印dB转换后的原始值（归一化前）
    print(f"[调试] Mel频谱图dB值（归一化前）: min={mel_db.min().item():.4f}, max={mel_db.max().item():.4f}, mean={mel_db.mean().item():.4f}")
    
    # 归一化：使用min-max归一化到[-1, 1]（与训练时完全一致）
    # 训练时的归一化方式：对每个样本计算min和max，然后归一化
    # 训练时mel_db形状是(B, n_mels, T)，然后unsqueeze成(B, 1, n_mels, T)
    # 归一化时：min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    # 这里mel_db是(1, n_mels, T)，需要先unsqueeze成(1, 1, n_mels, T)再归一化
    mel_db = mel_db.unsqueeze(0)  # (1, 1, n_mels, T)
    
    # 按照训练时的方式归一化
    mel_min = mel_db.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]  # (1, 1, 1, 1)
    mel_max = mel_db.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]  # (1, 1, 1, 1)
    mel_range = (mel_max - mel_min) + 1e-8
    mel_db = 2.0 * (mel_db - mel_min) / mel_range - 1.0
    
    # 调试信息：打印Mel频谱图形状
    print(f"[调试] Mel频谱图形状: {mel_db.shape}, 时间步数: {mel_db.shape[-1]}")
    print(f"[调试] Mel频谱图范围: min={mel_db.min().item():.4f}, max={mel_db.max().item():.4f}")
    print(f"[调试] Mel频谱图统计: mean={mel_db.mean().item():.4f}, std={mel_db.std().item():.4f}")
    
    return mel_db  # (1, 1, n_mels, T)


# ==================== 音频捕获和推理 ====================

def record_audio_sounddevice(duration: float = 10.0, sr: int = 16000, device: int = None) -> np.ndarray:
    """使用sounddevice录制音频"""
    print(f"\n[录音中] 正在录制 {duration} 秒...")
    audio_data = sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=1,
        dtype='float32',
        device=device
    )
    sd.wait()  # 等待录音完成
    return audio_data.flatten()


def record_audio_pyaudio(duration: float = 10.0, sr: int = 16000, chunk: int = 1024) -> np.ndarray:
    """使用pyaudio录制音频"""
    p = pyaudio.PyAudio()
    
    print(f"\n[录音中] 正在录制 {duration} 秒...")
    
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sr,
        input=True,
        frames_per_buffer=chunk
    )
    
    frames = []
    num_frames = int(sr * duration / chunk)
    
    for _ in range(num_frames):
        data = stream.read(chunk)
        frames.append(np.frombuffer(data, dtype=np.float32))
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    audio = np.concatenate(frames)
    return audio


def load_optimal_threshold(model_path: str, default_threshold: float = 0.34) -> float:
    """
    从test_metrics.json加载最佳阈值，如果没有则使用默认值
    
    参数:
        model_path: 模型文件路径
        default_threshold: 默认阈值（从训练结果看，最佳阈值约为0.34）
    
    返回:
        optimal_threshold: 最佳阈值
    """
    # 尝试从同目录下的test_metrics.json读取
    model_dir = os.path.dirname(model_path)
    metrics_path = os.path.join(model_dir, "test_metrics.json")
    
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                if 'optimal_threshold' in metrics:
                    threshold = metrics['optimal_threshold']
                    print(f"[信息] 从 {metrics_path} 加载最佳阈值: {threshold:.3f}")
                    return float(threshold)
        except Exception as e:
            print(f"[警告] 读取阈值文件失败: {e}，使用默认阈值 {default_threshold:.3f}")
    
    print(f"[信息] 使用默认阈值: {default_threshold:.3f} (训练时最佳阈值约为0.34)")
    return default_threshold


def predict_apnea(
    model: nn.Module,
    audio: np.ndarray,
    device: torch.device,
    sr: int = 16000,
    do_denoise: bool = False,  # 默认关闭，与训练时一致
    do_bandpass: bool = False,  # 默认关闭，与训练时一致
    threshold: float = 0.34  # 分类阈值（训练时最佳阈值约为0.34）
) -> Tuple[int, float, float]:
    """
    对音频进行呼吸暂停预测
    
    参数:
        threshold: 分类阈值，呼吸暂停概率 >= threshold 时预测为呼吸暂停
                  训练时最佳阈值约为0.34（而非默认0.5）
    
    返回:
        pred: 预测类别 (0=正常, 1=呼吸暂停)
        prob_apnea: 呼吸暂停概率
        prob_normal: 正常概率
    """
    model.eval()
    
    # 确保音频长度正确（160000 samples = 10秒 @ 16kHz）
    target_length = 160000
    if len(audio) != target_length:
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            # 如果不够长，用零填充
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    
    # 预处理
    wav_processed = preprocess_waveform(
        audio,
        fs=sr,
        do_denoise=do_denoise,
        do_bandpass=do_bandpass
    )
    
    # 转换为Mel频谱图
    mel_spec = audio_to_melspectrogram(wav_processed, sr=sr)
    mel_spec = mel_spec.to(device)
    
    # 推理
    with torch.no_grad():
        # 检查输入形状
        print(f"[调试] 模型输入形状: {mel_spec.shape}")
        expected_shape = (1, 1, 64, 313)  # 基于center=True的计算
        if mel_spec.shape != expected_shape:
            print(f"[警告] 输入形状不匹配！期望: {expected_shape}, 实际: {mel_spec.shape}")
            # 尝试调整形状
            if mel_spec.shape[-1] < expected_shape[-1]:
                # 时间步数不足，用零填充
                pad_size = expected_shape[-1] - mel_spec.shape[-1]
                mel_spec = F.pad(mel_spec, (0, pad_size), mode='constant', value=0)
                print(f"[调整] 已填充到期望形状: {mel_spec.shape}")
            elif mel_spec.shape[-1] > expected_shape[-1]:
                # 时间步数过多，截断
                mel_spec = mel_spec[..., :expected_shape[-1]]
                print(f"[调整] 已截断到期望形状: {mel_spec.shape}")
        
        logits = model(mel_spec)
        
        # 调试信息：打印logits
        print(f"[调试] Logits: 正常={logits[0, 0].item():.4f}, 呼吸暂停={logits[0, 1].item():.4f}")
        
        probs = F.softmax(logits, dim=1)
        prob_normal = probs[0, 0].item()
        prob_apnea = probs[0, 1].item()
        
        # 使用阈值进行预测（与训练时一致）
        # 训练时最佳阈值约为0.34，而不是默认的0.5
        pred = 1 if prob_apnea >= threshold else 0
        
        # 调试信息：打印概率和阈值
        print(f"[调试] 概率: 正常={prob_normal*100:.2f}%, 呼吸暂停={prob_apnea*100:.2f}%")
        print(f"[调试] 使用阈值: {threshold:.3f} (训练时最佳阈值)")
        print(f"[调试] 预测类别: {'呼吸暂停' if pred == 1 else '正常'}")
    
    return pred, prob_apnea, prob_normal


def print_result(pred: int, prob_apnea: float, prob_normal: float, timestamp: str = None):
    """打印检测结果"""
    if timestamp:
        print(f"\n[{timestamp}]", end=" ")
    
    status = "⚠️  呼吸暂停" if pred == 1 else "✅ 正常呼吸"
    color_code = "\033[91m" if pred == 1 else "\033[92m"  # 红色=呼吸暂停，绿色=正常
    reset_code = "\033[0m"
    
    print(f"{color_code}{status}{reset_code}")
    print(f"   正常概率: {prob_normal*100:.1f}%  |  呼吸暂停概率: {prob_apnea*100:.1f}%")
    print("-" * 60)


def continuous_monitoring(
    model: nn.Module,
    device: torch.device,
    duration: float = 10.0,
    sr: int = 16000,
    do_denoise: bool = False,  # 默认关闭，与训练时一致
    do_bandpass: bool = False,  # 默认关闭，与训练时一致
    audio_device: int = None,
    continuous: bool = True,
    threshold: float = 0.34  # 分类阈值（训练时最佳阈值）
):
    """持续监控模式"""
    # 检查音频库是否可用
    if not HAS_SOUNDDEVICE and not HAS_PYAUDIO:
        print("[错误] 需要安装 sounddevice 或 pyaudio 库用于音频捕获")
        print("安装命令: pip install sounddevice 或 pip install pyaudio")
        sys.exit(1)
    
    print("=" * 60)
    print("睡眠呼吸暂停实时检测系统")
    print("=" * 60)
    print(f"录音时长: {duration} 秒")
    print(f"采样率: {sr} Hz")
    print(f"设备: {'GPU (CUDA)' if device.type == 'cuda' else 'CPU'}")
    print(f"降噪: {'启用' if do_denoise else '禁用 (默认，与训练时一致)'}")
    print(f"带通滤波: {'启用' if do_bandpass else '禁用 (默认，与训练时一致)'}")
    print(f"分类阈值: {threshold:.3f} (训练时最佳阈值，而非默认0.5)")
    print("=" * 60)
    print("\n提示: 按 Ctrl+C 停止监控")
    
    if continuous:
        print("\n[连续监控模式] 将每10秒自动录音并分析...")
    else:
        print("\n[单次检测模式] 按 Enter 键开始录音...")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            if not continuous:
                input("\n按 Enter 键开始录音（或 Ctrl+C 退出）...")
            
            # 录音
            if HAS_SOUNDDEVICE:
                audio = record_audio_sounddevice(duration, sr, audio_device)
            else:
                audio = record_audio_pyaudio(duration, sr)
            
            # 检查音频是否有效（不是全零或过小）
            if np.max(np.abs(audio)) < 0.001:
                print("\n⚠️  警告: 检测到的音频信号非常弱，请检查麦克风连接")
                continue
            
            # 预测
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"\n[分析中] 第 {iteration} 次检测...")
            
            pred, prob_apnea, prob_normal = predict_apnea(
                model, audio, device, sr, do_denoise, do_bandpass, threshold=threshold
            )
            
            # 显示结果
            print_result(pred, prob_apnea, prob_normal, timestamp)
            
            if not continuous:
                # 单次模式，询问是否继续
                choice = input("\n是否继续检测？(y/n): ").strip().lower()
                if choice != 'y':
                    break
            
    except KeyboardInterrupt:
        print("\n\n[退出] 监控已停止")
    except Exception as e:
        print(f"\n[错误] 发生异常: {e}")
        import traceback
        traceback.print_exc()


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description="实时麦克风睡眠呼吸暂停识别")
    parser.add_argument("--model_path", type=str, default="outputs/apnea_2dcnn_best.pth",
                        help="模型权重文件路径")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="每次录音时长（秒），建议10秒")
    parser.add_argument("--sr", type=int, default=16000,
                        help="采样率（Hz）")
    parser.add_argument("--denoise", action="store_true",
                        help="启用降噪（默认关闭，与训练时一致）")
    parser.add_argument("--bandpass", action="store_true",
                        help="启用带通滤波（默认关闭，与训练时一致）")
    parser.add_argument("--single", action="store_true",
                        help="单次检测模式（需要手动触发）")
    parser.add_argument("--audio_device", type=int, default=None,
                        help="音频设备ID（用于sounddevice）")
    parser.add_argument("--list_devices", action="store_true",
                        help="列出可用的音频设备")
    parser.add_argument("--threshold", type=float, default=None,
                        help="分类阈值（默认从test_metrics.json读取，或使用0.34）")
    
    args = parser.parse_args()
    
    # 列出音频设备
    if args.list_devices:
        if HAS_SOUNDDEVICE:
            print("可用的音频输入设备:")
            print(sd.query_devices(kind='input'))
        else:
            p = pyaudio.PyAudio()
            print("可用的音频输入设备:")
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    print(f"  设备 {i}: {info['name']} (输入通道数: {info['maxInputChannels']})")
            p.terminate()
        return
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"[错误] 模型文件不存在: {args.model_path}")
        print("请先训练模型或指定正确的模型路径")
        sys.exit(1)
    
    # 加载模型
    print(f"[加载] 正在加载模型: {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CNN2D(num_classes=2, dropout=0.5, use_attention=True).to(device)
    
    # 尝试加载检查点格式
    # PyTorch 2.6+ requires weights_only=False for loading checkpoints with optimizer/scheduler states
    try:
        # Try with weights_only=False (PyTorch 2.6+)
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't have weights_only parameter
        checkpoint = torch.load(args.model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print(f"[信息] 加载检查点，epoch: {checkpoint.get('epoch', 'unknown')}")
        if 'val_f1' in checkpoint:
            print(f"[信息] 验证集F1分数: {checkpoint['val_f1']:.4f}")
        if 'val_acc' in checkpoint:
            print(f"[信息] 验证集准确率: {checkpoint['val_acc']:.4f}")
    else:
        # 直接加载state_dict
        model.load_state_dict(checkpoint)
        print("[信息] 直接加载state_dict")
    
    model.eval()
    
    # 测试模型输出（使用随机输入）
    print("[测试] 测试模型前向传播...")
    with torch.no_grad():
        test_input = torch.randn(1, 1, 64, 313, device=device)
        test_output = model(test_input)
        print(f"[测试] 测试输入形状: {test_input.shape}")
        print(f"[测试] 测试输出形状: {test_output.shape}")
        print(f"[测试] 测试输出logits: {test_output[0].cpu().numpy()}")
    
    print("[完成] 模型加载成功")
    
    # 加载最佳阈值
    if args.threshold is not None:
        threshold = args.threshold
        print(f"[信息] 使用命令行指定的阈值: {threshold:.3f}")
    else:
        threshold = load_optimal_threshold(args.model_path)
    
    # 运行监控
    continuous_monitoring(
        model,
        device,
        duration=args.duration,
        sr=args.sr,
        do_denoise=args.denoise,
        do_bandpass=args.bandpass,
        audio_device=args.audio_device,
        continuous=(not args.single),
        threshold=threshold
    )


if __name__ == "__main__":
    main()

