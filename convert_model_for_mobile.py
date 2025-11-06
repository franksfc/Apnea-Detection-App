# -*- coding: utf-8 -*-
"""
convert_model_for_mobile.py
----------------------------
将训练好的PyTorch模型转换为移动端可用的格式（TorchScript）

使用方法:
    python convert_model_for_mobile.py --model_path outputs/apnea_2dcnn_best.pth --output_path mobile_app/assets/apnea_model.pt
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio

# 从main.py导入模型定义和注意力机制
import sys
import os

# 添加当前目录到路径，以便导入main模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import CNN2D, CBAM, ChannelAttention, SpatialAttention


def convert_model_to_torchscript(model_path: str, output_path: str, device: str = 'cpu'):
    """
    将PyTorch模型转换为TorchScript格式，用于移动端部署
    
    参数:
        model_path: 训练好的模型权重路径
        output_path: 输出TorchScript模型路径
        device: 转换设备（'cpu'或'cuda'）
    """
    print(f"[转换] 加载模型: {model_path}")
    
    # 检查模型文件
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 创建模型实例（需要与训练时完全一致）
    model = CNN2D(num_classes=2, dropout=0.5, use_attention=True)
    
    # 加载权重
    device_obj = torch.device(device)
    checkpoint = torch.load(model_path, map_location=device_obj, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print(f"[信息] 加载检查点，epoch: {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device_obj)
    
    print(f"[信息] 模型加载成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例输入（Mel频谱图的形状）
    # 输入: (batch_size, channels, n_mels, time_steps)
    # 10秒音频 @ 16kHz = 160000 samples
    # Mel频谱图: (1, 64, T) where T = (160000 - 1024) // 512 + 1 ≈ 312
    n_mels = 64
    n_fft = 1024
    hop_length = 512
    target_length = 160000
    mel_time_steps = 1 + (target_length - n_fft) // hop_length
    
    # 创建示例输入
    example_input = torch.randn(1, 1, n_mels, mel_time_steps, device=device_obj)
    
    print(f"[信息] 示例输入形状: {example_input.shape}")
    
    # 测试模型前向传播
    print("[测试] 测试模型前向传播...")
    with torch.no_grad():
        output = model(example_input)
        print(f"[信息] 输出形状: {output.shape}")
        print(f"[信息] 输出示例: {output}")
    
    # 转换为TorchScript
    print("[转换] 转换为TorchScript...")
    try:
        # 方法1: 使用trace（推荐，更快）
        traced_model = torch.jit.trace(model, example_input)
        
        # 验证转换后的模型
        print("[验证] 验证转换后的模型...")
        with torch.no_grad():
            traced_output = traced_model(example_input)
            
        # 检查输出是否一致
        if torch.allclose(output, traced_output, atol=1e-5):
            print("[成功] TorchScript模型输出与原始模型一致")
        else:
            print("[警告] TorchScript模型输出与原始模型存在差异")
            print(f"原始输出: {output}")
            print(f"转换后输出: {traced_output}")
        
        # 保存模型
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        traced_model.save(output_path)
        print(f"[成功] 模型已保存到: {output_path}")
        
        # 获取文件大小
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"[信息] 模型文件大小: {file_size:.2f} MB")
        
        return traced_model
        
    except Exception as e:
        print(f"[错误] TorchScript转换失败: {e}")
        print("[尝试] 尝试使用script模式...")
        
        # 方法2: 使用script（如果trace失败）
        try:
            scripted_model = torch.jit.script(model)
            scripted_model.save(output_path)
            print(f"[成功] 使用script模式保存模型到: {output_path}")
            return scripted_model
        except Exception as e2:
            print(f"[错误] Script模式也失败: {e2}")
            raise


def create_audio_preprocessing_module():
    """
    创建音频预处理模块（Mel频谱图转换）
    这个模块也需要转换为TorchScript以便在移动端使用
    """
    class AudioPreprocessor(nn.Module):
        """音频预处理模块：将音频波形转换为Mel频谱图"""
        def __init__(self, sr=16000, n_mels=64, n_fft=1024, hop_length=512, 
                     fmin=50, fmax=4000):
            super().__init__()
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=fmin,
                f_max=fmax
            )
            self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        
        def forward(self, audio_waveform):
            """
            参数:
                audio_waveform: (batch_size, samples) 或 (samples,)
            返回:
                mel_spectrogram: (batch_size, 1, n_mels, time_steps)
            """
            # 确保输入是2D
            if audio_waveform.dim() == 1:
                audio_waveform = audio_waveform.unsqueeze(0)
            
            # 转换为Mel频谱图
            mel = self.mel_transform(audio_waveform)  # (B, n_mels, T)
            mel_db = self.to_db(mel)
            
            # 归一化到[-1, 1]
            mel_min = mel_db.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
            mel_max = mel_db.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            mel_range = (mel_max - mel_min) + 1e-8
            mel_db = 2.0 * (mel_db - mel_min) / mel_range - 1.0
            
            # 添加channel维度: (B, n_mels, T) -> (B, 1, n_mels, T)
            mel_db = mel_db.unsqueeze(1)
            
            return mel_db
    
    return AudioPreprocessor


def convert_preprocessor_to_torchscript(output_path: str):
    """
    转换音频预处理模块为TorchScript
    """
    print("[转换] 创建音频预处理模块...")
    preprocessor = create_audio_preprocessing_module()()
    preprocessor.eval()
    
    # 创建示例输入（10秒音频 @ 16kHz）
    example_audio = torch.randn(160000)
    
    print("[转换] 转换预处理模块为TorchScript...")
    traced_preprocessor = torch.jit.trace(preprocessor, example_audio)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced_preprocessor.save(output_path)
    print(f"[成功] 预处理模块已保存到: {output_path}")
    
    return traced_preprocessor


def main():
    parser = argparse.ArgumentParser(description="将PyTorch模型转换为移动端格式")
    parser.add_argument("--model_path", type=str, 
                        default="outputs/apnea_2dcnn_best.pth",
                        help="训练好的模型权重路径")
    parser.add_argument("--output_dir", type=str, 
                        default="mobile_app/assets",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="转换设备")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 转换模型
    model_output_path = os.path.join(args.output_dir, "apnea_model.pt")
    convert_model_to_torchscript(
        args.model_path, 
        model_output_path, 
        device=args.device
    )
    
    # 转换预处理模块
    preprocessor_output_path = os.path.join(args.output_dir, "audio_preprocessor.pt")
    convert_preprocessor_to_torchscript(preprocessor_output_path)
    
    print("\n" + "="*60)
    print("转换完成！")
    print("="*60)
    print(f"模型文件: {model_output_path}")
    print(f"预处理模块: {preprocessor_output_path}")
    print("\n这些文件可以用于移动端部署。")


if __name__ == "__main__":
    main()

