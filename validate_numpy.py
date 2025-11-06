# -*- coding: utf-8 -*-
"""
validate_numpy.py
-----------------
直接使用numpy文件验证模型性能

功能：
- 从numpy文件直接加载音频数据
- 使用训练好的模型进行预测
- 显示预测结果和统计信息
- 支持批量测试

用法:
    python validate_numpy.py --model_path outputs2/apnea_2dcnn_best.pth --data_root psg-audio-apnea-audios/PSG-AUDIO/APNEA_EDF
"""

import os
import sys
import argparse
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Tuple, List
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

# 从 predict_microphone.py 导入模型定义和预处理函数
from predict_microphone import (
    CNN2D, CBAM, ChannelAttention, SpatialAttention,
    preprocess_waveform, audio_to_melspectrogram,
    load_optimal_threshold
)


def load_numpy_samples(file_path: str, sample_idx: int = None) -> Tuple[np.ndarray, int]:
    """
    从numpy文件加载音频样本
    
    参数:
        file_path: numpy文件路径
        sample_idx: 样本索引（如果为None，返回所有样本）
    
    返回:
        audio_data: 音频数据 (160000,) 或 (N, 160000)
        label: 标签 (0=正常, 1=呼吸暂停)
    """
    data = np.load(file_path, mmap_mode='r')
    
    # 确定标签
    if '_ap.npy' in file_path:
        label = 1  # 呼吸暂停
    elif '_nap.npy' in file_path:
        label = 0  # 正常
    else:
        label = -1  # 未知
    
    if sample_idx is not None:
        if sample_idx >= len(data):
            raise IndexError(f"样本索引 {sample_idx} 超出范围 (总共 {len(data)} 个样本)")
        return data[sample_idx], label
    else:
        return data, label


def predict_single_sample(
    model: nn.Module,
    audio: np.ndarray,
    device: torch.device,
    sr: int = 16000,
    do_denoise: bool = False,
    do_bandpass: bool = False,
    threshold: float = 0.34,  # 分类阈值（训练时最佳阈值约为0.34）
    verbose: bool = True
) -> Tuple[int, float, float, float]:
    """
    对单个音频样本进行预测（与predict_microphone.py和移动端一致）
    
    参数:
        threshold: 分类阈值，呼吸暂停概率 >= threshold 时预测为呼吸暂停
                  训练时最佳阈值约为0.34（而非默认0.5）
    
    返回:
        pred: 预测类别 (0=正常, 1=呼吸暂停)
        prob_apnea: 呼吸暂停概率
        prob_normal: 正常概率
        confidence: 置信度 (最大概率)
    """
    model.eval()
    
    # 确保音频长度正确
    target_length = 160000
    if len(audio) != target_length:
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
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
    
    # 检查输入形状
    expected_shape = (1, 1, 64, 313)
    if mel_spec.shape != expected_shape:
        if verbose:
            print(f"[警告] 输入形状不匹配！期望: {expected_shape}, 实际: {mel_spec.shape}")
        if mel_spec.shape[-1] < expected_shape[-1]:
            pad_size = expected_shape[-1] - mel_spec.shape[-1]
            mel_spec = F.pad(mel_spec, (0, pad_size), mode='constant', value=0)
        elif mel_spec.shape[-1] > expected_shape[-1]:
            mel_spec = mel_spec[..., :expected_shape[-1]]
    
    # 推理
    with torch.no_grad():
        logits = model(mel_spec)
        probs = F.softmax(logits, dim=1)
        prob_normal = probs[0, 0].item()
        prob_apnea = probs[0, 1].item()
        
        # 使用阈值进行预测（与训练时一致）
        # 训练时最佳阈值约为0.34，而不是默认的0.5
        pred = 1 if prob_apnea >= threshold else 0
        confidence = max(prob_normal, prob_apnea)
        
        if verbose:
            print(f"  Logits: 正常={logits[0, 0].item():.4f}, 呼吸暂停={logits[0, 1].item():.4f}")
            print(f"  概率: 正常={prob_normal*100:.2f}%, 呼吸暂停={prob_apnea*100:.2f}%")
            print(f"  使用阈值: {threshold:.3f} (训练时最佳阈值)")
            print(f"  预测: {'呼吸暂停' if pred == 1 else '正常'} (置信度: {confidence*100:.2f}%)")
    
    return pred, prob_apnea, prob_normal, confidence


def validate_single_file(
    model: nn.Module,
    file_path: str,
    device: torch.device,
    sample_idx: int = None,
    do_denoise: bool = False,
    do_bandpass: bool = False,
    threshold: float = 0.34  # 分类阈值（训练时最佳阈值）
):
    """验证单个numpy文件"""
    print(f"\n{'='*60}")
    print(f"文件: {os.path.basename(file_path)}")
    
    try:
        audio_data, true_label = load_numpy_samples(file_path, sample_idx)
        
        if sample_idx is not None:
            # 单个样本
            audio = audio_data
            label_name = "呼吸暂停" if true_label == 1 else "正常"
            print(f"样本索引: {sample_idx}")
            print(f"真实标签: {label_name}")
            
            pred, prob_apnea, prob_normal, confidence = predict_single_sample(
                model, audio, device, do_denoise=do_denoise, do_bandpass=do_bandpass, threshold=threshold
            )
            
            is_correct = (pred == true_label)
            status = "✅ 正确" if is_correct else "❌ 错误"
            print(f"结果: {status}")
            
            return {
                'file': file_path,
                'sample_idx': sample_idx,
                'true_label': true_label,
                'pred': pred,
                'prob_apnea': prob_apnea,
                'prob_normal': prob_normal,
                'confidence': confidence,
                'correct': is_correct
            }
        else:
            # 批量测试
            print(f"总样本数: {len(audio_data)}")
            results = []
            correct_count = 0
            
            for i in range(len(audio_data)):
                audio = audio_data[i]
                pred, prob_apnea, prob_normal, confidence = predict_single_sample(
                    model, audio, device, do_denoise=do_denoise, 
                    do_bandpass=do_bandpass, threshold=threshold, verbose=False
                )
                
                is_correct = (pred == true_label)
                if is_correct:
                    correct_count += 1
                
                results.append({
                    'sample_idx': i,
                    'true_label': true_label,
                    'pred': pred,
                    'prob_apnea': prob_apnea,
                    'prob_normal': prob_normal,
                    'confidence': confidence,
                    'correct': is_correct
                })
                
                if (i + 1) % 10 == 0:
                    print(f"  已处理: {i+1}/{len(audio_data)}")
            
            accuracy = correct_count / len(audio_data) * 100
            label_name = "呼吸暂停" if true_label == 1 else "正常"
            print(f"\n真实标签: {label_name}")
            print(f"准确率: {accuracy:.2f}% ({correct_count}/{len(audio_data)})")
            
            return {
                'file': file_path,
                'true_label': true_label,
                'total_samples': len(audio_data),
                'correct_count': correct_count,
                'accuracy': accuracy,
                'results': results
            }
            
    except Exception as e:
        print(f"[错误] 处理文件失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_optimal_threshold(probs_apnea, labels, metric='f1'):
    """寻找最佳阈值以最大化指定指标"""
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (probs_apnea >= threshold).astype(int)
        if metric == 'f1':
            _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
            score = f1
        elif metric == 'recall':
            _, recall, _, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
            score = recall
        else:
            acc = accuracy_score(labels, preds)
            score = acc
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def validate_dataset(
    model: nn.Module,
    data_root: str,
    device: torch.device,
    num_samples: int = None,
    do_denoise: bool = False,
    do_bandpass: bool = False,
    test_apnea: bool = True,
    test_normal: bool = True,
    optimize_threshold: bool = True,
    threshold: float = 0.34  # 分类阈值（训练时最佳阈值）
):
    """验证整个数据集"""
    print("="*60)
    print("数据集验证")
    print("="*60)
    
    # 查找所有numpy文件
    ap_files = []
    nap_files = []
    
    if test_apnea:
        ap_files = glob.glob(os.path.join(data_root, "**", "*_ap.npy"), recursive=True)
    if test_normal:
        nap_files = glob.glob(os.path.join(data_root, "**", "*_nap.npy"), recursive=True)
    
    print(f"找到 {len(ap_files)} 个呼吸暂停文件")
    print(f"找到 {len(nap_files)} 个正常文件")
    
    all_results = []
    
    # 测试呼吸暂停样本
    if ap_files and num_samples:
        print(f"\n测试前 {num_samples} 个呼吸暂停样本...")
        for i, file_path in enumerate(ap_files[:num_samples]):
            result = validate_single_file(
                model, file_path, device, sample_idx=0,
                do_denoise=do_denoise, do_bandpass=do_bandpass, threshold=threshold
            )
            if result:
                all_results.append(result)
    
    # 测试正常样本
    if nap_files and num_samples:
        print(f"\n测试前 {num_samples} 个正常样本...")
        for i, file_path in enumerate(nap_files[:num_samples]):
            result = validate_single_file(
                model, file_path, device, sample_idx=0,
                do_denoise=do_denoise, do_bandpass=do_bandpass, threshold=threshold
            )
            if result:
                all_results.append(result)
    
    # 统计结果
    if all_results:
        print("\n" + "="*60)
        print(f"总体统计（使用阈值 {threshold:.3f}，训练时最佳阈值）")
        print("="*60)
        
        correct_count = sum(1 for r in all_results if r.get('correct', False))
        total_count = len(all_results)
        overall_accuracy = correct_count / total_count * 100 if total_count > 0 else 0
        
        # 按类别统计
        apnea_results = [r for r in all_results if r.get('true_label') == 1]
        normal_results = [r for r in all_results if r.get('true_label') == 0]
        
        print(f"总样本数: {total_count}")
        print(f"正确预测: {correct_count}")
        print(f"总体准确率: {overall_accuracy:.2f}%")
        
        if apnea_results:
            apnea_correct = sum(1 for r in apnea_results if r.get('correct', False))
            apnea_acc = apnea_correct / len(apnea_results) * 100
            print(f"\n呼吸暂停样本:")
            print(f"  总数: {len(apnea_results)}")
            print(f"  正确: {apnea_correct}")
            print(f"  准确率: {apnea_acc:.2f}%")
        
        if normal_results:
            normal_correct = sum(1 for r in normal_results if r.get('correct', False))
            normal_acc = normal_correct / len(normal_results) * 100
            print(f"\n正常样本:")
            print(f"  总数: {len(normal_results)}")
            print(f"  正确: {normal_correct}")
            print(f"  准确率: {normal_acc:.2f}%")
        
        # 如果启用阈值优化，计算优化后的指标（与训练时一致）
        if optimize_threshold and len(all_results) > 0:
            print("\n" + "="*60)
            print("使用优化阈值（与训练时一致）")
            print("="*60)
            
            # 收集所有概率和标签
            probs_apnea = np.array([r.get('prob_apnea', 0) for r in all_results])
            labels = np.array([r.get('true_label', 0) for r in all_results])
            
            # 寻找最佳阈值
            optimal_threshold, best_f1 = find_optimal_threshold(probs_apnea, labels, metric='f1')
            preds_optimal = (probs_apnea >= optimal_threshold).astype(int)
            
            # 计算优化后的指标
            acc_optimal = accuracy_score(labels, preds_optimal)
            precision_optimal, recall_optimal, f1_optimal, _ = precision_recall_fscore_support(
                labels, preds_optimal, average='binary', zero_division=0
            )
            
            try:
                auc = roc_auc_score(labels, probs_apnea)
            except:
                auc = float('nan')
            
            cm = confusion_matrix(labels, preds_optimal)
            
            print(f"最佳阈值: {optimal_threshold:.3f}")
            print(f"准确率: {acc_optimal*100:.2f}%")
            print(f"精确率: {precision_optimal*100:.2f}%")
            print(f"召回率: {recall_optimal*100:.2f}%")
            print(f"F1分数: {f1_optimal*100:.2f}%")
            print(f"AUC: {auc:.4f}")
            print(f"\n混淆矩阵:")
            print(f"              预测")
            print(f"           正常  呼吸暂停")
            print(f"真实 正常   {cm[0,0]:4d}   {cm[0,1]:4d}")
            print(f"    呼吸暂停 {cm[1,0]:4d}   {cm[1,1]:4d}")
            
            # 按类别统计优化后的结果
            apnea_preds_optimal = preds_optimal[labels == 1]
            normal_preds_optimal = preds_optimal[labels == 0]
            
            apnea_correct_optimal = np.sum(apnea_preds_optimal == 1)
            normal_correct_optimal = np.sum(normal_preds_optimal == 0)
            
            print(f"\n优化后的分类准确率:")
            print(f"  呼吸暂停: {apnea_correct_optimal}/{len(apnea_preds_optimal)} = {apnea_correct_optimal/len(apnea_preds_optimal)*100:.2f}%")
            print(f"  正常: {normal_correct_optimal}/{len(normal_preds_optimal)} = {normal_correct_optimal/len(normal_preds_optimal)*100:.2f}%")
        
        # 显示错误预测
        wrong_results = [r for r in all_results if not r.get('correct', True)]
        if wrong_results:
            print(f"\n错误预测 ({len(wrong_results)} 个，使用阈值 {threshold:.3f}):")
            for r in wrong_results[:10]:  # 只显示前10个
                true_label = "呼吸暂停" if r.get('true_label') == 1 else "正常"
                pred_label = "呼吸暂停" if r.get('pred') == 1 else "正常"
                print(f"  {os.path.basename(r.get('file', 'unknown'))}: "
                      f"真实={true_label}, 预测={pred_label}, "
                      f"置信度={r.get('confidence', 0)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="使用numpy文件验证模型")
    parser.add_argument("--model_path", type=str, default="outputs/apnea_2dcnn_best.pth",
                        help="模型权重文件路径")
    parser.add_argument("--data_root", type=str,
                        default="psg-audio-apnea-audios/PSG-AUDIO/APNEA_EDF",
                        help="数据集根目录")
    parser.add_argument("--file_path", type=str, default=None,
                        help="单个numpy文件路径（如果指定，只测试该文件）")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="样本索引（如果指定，只测试该索引的样本）")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="批量测试时的样本数量")
    parser.add_argument("--denoise", action="store_true",
                        help="启用降噪（默认关闭，与训练时一致）")
    parser.add_argument("--bandpass", action="store_true",
                        help="启用带通滤波（默认关闭，与训练时一致）")
    parser.add_argument("--no_apnea", action="store_true",
                        help="不测试呼吸暂停样本")
    parser.add_argument("--no_normal", action="store_true",
                        help="不测试正常样本")
    parser.add_argument("--threshold", type=float, default=None,
                        help="分类阈值（默认从test_metrics.json读取，或使用0.34）")
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"[错误] 模型文件不存在: {args.model_path}")
        sys.exit(1)
    
    # 加载模型
    print(f"[加载] 正在加载模型: {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CNN2D(num_classes=2, dropout=0.5, use_attention=True).to(device)
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print(f"[信息] 加载检查点，epoch: {checkpoint.get('epoch', 'unknown')}")
        if 'val_f1' in checkpoint:
            print(f"[信息] 验证集F1分数: {checkpoint['val_f1']:.4f}")
        if 'val_acc' in checkpoint:
            print(f"[信息] 验证集准确率: {checkpoint['val_acc']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("[信息] 直接加载state_dict")
    
    model.eval()
    print("[完成] 模型加载成功")
    
    # 测试模型
    print("[测试] 测试模型前向传播...")
    with torch.no_grad():
        test_input = torch.randn(1, 1, 64, 313, device=device)
        test_output = model(test_input)
        print(f"[测试] 测试输出logits: {test_output[0].cpu().numpy()}")
    
    # 加载最佳阈值（与predict_microphone.py一致）
    if args.threshold is not None:
        threshold = args.threshold
        print(f"[信息] 使用命令行指定的阈值: {threshold:.3f}")
    else:
        threshold = load_optimal_threshold(args.model_path)
    
    # 运行验证
    if args.file_path:
        # 单个文件测试
        validate_single_file(
            model, args.file_path, device,
            sample_idx=args.sample_idx,
            do_denoise=args.denoise,
            do_bandpass=args.bandpass,
            threshold=threshold
        )
    else:
        # 数据集验证
        if not os.path.exists(args.data_root):
            print(f"[错误] 数据集目录不存在: {args.data_root}")
            sys.exit(1)
        
        validate_dataset(
            model, args.data_root, device,
            num_samples=args.num_samples,
            do_denoise=args.denoise,
            do_bandpass=args.bandpass,
            test_apnea=not args.no_apnea,
            test_normal=not args.no_normal,
            optimize_threshold=True,  # 与训练时一致，使用阈值优化
            threshold=threshold  # 使用加载的阈值
        )


if __name__ == "__main__":
    main()

