/**
 * ApneaDetector - 睡眠呼吸暂停检测移动应用
 *
 * 功能：
 * - 实时录音
 * - 滑动窗口算法检测
 * - 显示检测结果和置信度
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  Alert,
  Platform,
  Switch,
} from 'react-native';
import { check, request, PERMISSIONS, RESULTS } from 'react-native-permissions';
import RNFS from 'react-native-fs';
import KeepAwake from 'react-native-keep-awake';
import AudioProcessor from './src/AudioProcessor';
import ModelInference from './src/ModelInference';
import AudioRecorder from './src/AudioRecorder';
import AudioCaptureNative, { audioCaptureEventEmitter } from './src/AudioCaptureNative';

// 注意：PyTorch Mobile需要原生模块，这里使用伪代码说明
// 实际部署时需要安装react-native-pytorch-core或使用ONNX Runtime
// import { torch, torchvision } from 'react-native-pytorch-core';

interface DetectionResult {
  timestamp: string;
  prediction: 0 | 1;
  probNormal: string;
  probApnea: string;
}

const App = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectionResults, setDetectionResults] = useState<DetectionResult[]>([]);
  const [currentStatus, setCurrentStatus] = useState('空闲');
  const [currentConfidence, setCurrentConfidence] = useState(0);
  const [detectionCount, setDetectionCount] = useState(0);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [apneaCount, setApneaCount] = useState(0); // apnea检测次数
  const [enableDenoise, setEnableDenoise] = useState(false); // 降噪开关
  const [enableBandpass, setEnableBandpass] = useState(false); // 滤波开关
  const [keepScreenAwake, setKeepScreenAwake] = useState(false); // 防止息屏开关
  
  const audioRecorderPlayer = useRef(new AudioRecorder()).current;
  const audioProcessor = useRef(new AudioProcessor()).current;
  // 使用训练时最佳阈值0.34（而非默认0.5）
  const modelInference = useRef(new ModelInference(0.34)).current;
  
  const windowSize = 160000; // 10秒 @ 16kHz = 160000 samples
  const hopSize = 80000; // 5秒步长，用于滑动窗口
  const sampleRate = 16000;
  const detectionInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const recordingStartTime = useRef<number>(0);
  const durationInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioDataInterval = useRef<ReturnType<typeof setInterval> | null>(null); // 音频数据添加间隔
  const waitCountRef = useRef<number>(0);
  const hasSimulatedDataRef = useRef<boolean>(false);
  const isProcessingRef = useRef<boolean>(false); // 用于在闭包中检查处理状态
  const detectionCountRef = useRef<number>(0); // 使用ref存储检测计数，避免状态更新问题
  const isRecordingRef = useRef<boolean>(false); // 用于在闭包中检查录音状态
  const isReadyForAudioDataRef = useRef<boolean>(false); // 标记是否准备好接收音频数据（用于丢弃启动时的缓冲数据）

  useEffect(() => {
    // 检查 Hermes 引擎
    try {
      // @ts-ignore - global 在 React Native 环境中存在
      if (typeof global !== 'undefined' && (global as any).HermesInternal) {
        console.log('✅ Hermes 引擎已启用');
      } else {
        console.warn('⚠️ Hermes 引擎未启用，DevTools 可能无法正常工作');
      }
    } catch (e) {
      console.warn('⚠️ 无法检查 Hermes 引擎状态');
    }
    
    // 请求麦克风权限
    requestMicrophonePermission();
    
    // 加载模型
    loadModel();
    
    // 设置音频捕获事件监听
    let audioDataListener: any = null;
    if (audioCaptureEventEmitter) {
      audioDataListener = audioCaptureEventEmitter.addListener(
        'onAudioData',
        (event: { audioData: number[]; sampleCount: number }) => {
          // 只有在准备好接收数据时才添加（用于丢弃启动时的缓冲数据）
          if (isReadyForAudioDataRef.current && event.audioData && event.audioData.length > 0) {
            audioProcessor.addAudioData(event.audioData);
          }
        }
      );
    }
    
    return () => {
      // 清理资源
      if (detectionInterval.current) {
        clearInterval(detectionInterval.current);
      }
      if (audioDataListener) {
        audioDataListener.remove();
      }
      // 确保在组件卸载时禁用防止息屏
      KeepAwake.deactivate();
      stopRecording();
    };
  }, []);

  const requestMicrophonePermission = async () => {
    try {
      const permission = Platform.OS === 'android' 
        ? PERMISSIONS.ANDROID.RECORD_AUDIO 
        : PERMISSIONS.IOS.MICROPHONE;

      const result = await check(permission);
      
      if (result === RESULTS.GRANTED) {
        console.log('麦克风权限已授予');
      } else {
        const requestResult = await request(permission);
        if (requestResult !== RESULTS.GRANTED) {
          Alert.alert('权限被拒绝', '需要麦克风权限才能使用此应用');
        }
      }
    } catch (err) {
      console.warn('权限请求错误:', err);
    }
  };

  const loadModel = async () => {
    try {
      console.log('正在加载模型...');
      setStatusMessage('正在加载模型文件...');
      
      // 尝试加载真实模型文件
      let modelPath: string | undefined;
      let preprocessorPath: string | undefined;
      
      if (Platform.OS === 'android') {
        // Android: assets文件需要使用 asset:// 协议
        // 原生模块会自动从assets目录加载文件
        try {
          // 尝试多个可能的路径
          const possiblePaths = [
            // Android assets 目录（原生模块会自动处理）
            'asset://apnea_model.pt',
            // 从assets复制到可访问目录后的路径
            `${RNFS.DocumentDirectoryPath}/apnea_model.pt`,
            // Bundle路径
            `${RNFS.MainBundlePath}/apnea_model.pt`,
          ];
          
          // 首先尝试asset路径（推荐，原生模块会自动处理）
          modelPath = 'asset://apnea_model.pt';
          preprocessorPath = 'asset://audio_preprocessor.pt';
          console.log('使用asset路径加载模型（原生模块会自动处理）:', modelPath);
          
          // 可选：检查文件是否在其他位置
          // for (const path of possiblePaths.slice(1)) {
          //   try {
          //     const exists = await RNFS.exists(path);
          //     if (exists) {
          //       modelPath = path;
          //       const ppPath = path.replace('apnea_model.pt', 'audio_preprocessor.pt');
          //       if (await RNFS.exists(ppPath)) {
          //         preprocessorPath = ppPath;
          //       }
          //       console.log('找到模型文件:', modelPath);
          //       break;
          //     }
          //   } catch (e) {
          //     continue;
          //   }
          // }
        } catch (e) {
          console.log('检查模型文件时出错:', e);
        }
      } else {
        // iOS: 使用Bundle路径
        try {
          const possiblePaths = [
            `${RNFS.MainBundlePath}/apnea_model.pt`,
            `${RNFS.DocumentDirectoryPath}/apnea_model.pt`,
          ];
          
          for (const path of possiblePaths) {
            const exists = await RNFS.exists(path);
            if (exists) {
              modelPath = path;
              const ppPath = path.replace('apnea_model.pt', 'audio_preprocessor.pt');
              if (await RNFS.exists(ppPath)) {
                preprocessorPath = ppPath;
              }
              console.log('找到模型文件:', modelPath);
              break;
            }
          }
        } catch (e) {
          console.log('检查模型文件时出错:', e);
        }
      }
      
          // 加载模型（如果找到了文件）
          if (modelPath) {
            console.log('尝试加载模型文件:', modelPath);
            await modelInference.loadModel(modelPath, preprocessorPath || null);
            
            // 设置Temperature Scaling（提高置信度）
            try {
              await modelInference.setTemperature(0.7); // 推荐值：0.7-0.9，0.7更自信
              console.log('Temperature Scaling已设置: 0.7');
            } catch (e) {
              console.warn('设置Temperature失败，使用默认值:', e);
            }
            
            // 确保使用最佳阈值（训练时最佳阈值约为0.34）
            modelInference.setThreshold(0.34);
            console.log(`分类阈值已设置为: ${modelInference.getThreshold().toFixed(3)} (训练时最佳阈值)`);
            
            setStatusMessage('模型加载成功');
            console.log('真实模型加载成功');
          } else {
            // 使用模拟模式
            await modelInference.loadModel(); // 不传路径，使用模拟模式
            setStatusMessage('使用模拟模式（未找到模型文件）');
            console.log('使用模拟模式（模型文件未找到）');
          }
      
      setCurrentStatus('就绪');
    } catch (error) {
      console.error('模型加载失败:', error);
      // 如果加载失败，尝试使用模拟模式
      try {
        await modelInference.loadModel();
        setStatusMessage('模型加载失败，已切换到模拟模式');
        setCurrentStatus('就绪（模拟模式）');
      } catch (e) {
        Alert.alert('错误', '模型加载失败，请检查模型文件');
        setCurrentStatus('错误');
      }
    }
  };

  const startRecording = async () => {
    try {
      // 同步音频处理设置
      audioProcessor.setDenoiseEnabled(enableDenoise);
      audioProcessor.setBandpassEnabled(enableBandpass);
      
      // 清空缓冲区
      audioProcessor.clear();
      detectionCountRef.current = 0;
      setDetectionCount(0);
      setApneaCount(0); // 重置apnea计数
      setRecordingDuration(0);
      hasSimulatedDataRef.current = false; // 重置模拟数据标记
      isReadyForAudioDataRef.current = false; // 标记为未准备好，用于丢弃启动时的缓冲数据
      setCurrentStatus('正在启动录音...');
      setStatusMessage('正在初始化录音设备...');
      
      // 开始录音
      const audioSet = {
        AudioEncoderAndroid: 'AAC',
        AudioSourceAndroid: 'MIC',
        AVModeIOSOption: 'measurement',
        AVEncoderAudioQualityKeyIOS: 'high',
        AVNumberOfChannelsKeyIOS: 1,
        AVFormatIDKeyIOS: 'aac',
      };
      
      const uri = await audioRecorderPlayer.startRecorder(
        `${RNFS.CachesDirectoryPath}/apnea_recording.aac`,
        audioSet
      );
      
      audioRecorderPlayer.addRecordBackListener((e) => {
        // 实时获取音频数据（目前原生模块不提供实时数据流）
        // 这里主要用于显示录音进度
        console.log('录音中...', e.currentPosition);
      });
      
      setIsRecording(true);
      isRecordingRef.current = true; // 更新ref
      recordingStartTime.current = Date.now();
      setCurrentStatus('🔴 录音中');
      setStatusMessage('正在收集音频数据，等待首次检测...');
      
      // 启动实时音频捕获（使用AudioRecord）
      // 先确保之前的资源已完全释放
      if (AudioCaptureNative) {
        try {
          // 先尝试停止，确保资源已释放（即使之前可能已经停止）
          try {
            await AudioCaptureNative.stopCapture();
            // 等待一小段时间确保资源完全释放
            await new Promise<void>(resolve => setTimeout(() => resolve(), 200));
          } catch (e) {
            // 如果停止失败（可能是未在捕获），忽略错误
            console.log('清理之前的音频捕获资源（可能已停止）');
          }
          
          // 现在启动新的捕获
          await AudioCaptureNative.startCapture();
          console.log('实时音频捕获已启动');
          
          // 重要：等待系统稳定，并丢弃启动时的缓冲数据
          // AudioRecord 在启动时可能会读取到一些系统缓冲的旧数据
          // 我们需要等待一小段时间，让系统稳定，然后清空缓冲区
          console.log('等待音频系统稳定，丢弃可能的缓冲数据...');
          await new Promise<void>(resolve => setTimeout(() => resolve(), 500));
          
          // 再次清空缓冲区，确保不使用启动时的缓冲数据
          audioProcessor.clear();
          
          // 现在标记为准备好接收音频数据（之前的缓冲数据已被丢弃）
          isReadyForAudioDataRef.current = true;
          console.log('音频缓冲区已清空，开始使用真实录音数据');
        } catch (error: any) {
          console.error('启动实时音频捕获失败:', error);
          // 如果实时捕获失败，降级到模拟模式
          console.warn('降级到模拟音频数据模式');
        }
      } else {
        console.warn('AudioCaptureNative不可用，使用模拟音频数据');
        // 模拟模式下立即标记为准备好接收数据
        isReadyForAudioDataRef.current = true;
      }
      
      // 启动录音时长计时器
      durationInterval.current = setInterval(() => {
        const duration = Math.floor((Date.now() - recordingStartTime.current) / 1000);
        setRecordingDuration(duration);
      }, 1000);
      
      // 启动滑动窗口检测循环
      startSlidingWindowDetection();
      
    } catch (error) {
      console.error('开始录音失败:', error);
      Alert.alert('错误', '无法开始录音');
      setIsRecording(false);
      setCurrentStatus('录音失败');
      setStatusMessage('');
    }
  };

  const stopRecording = async () => {
    try {
      // 先停止录音状态
      isRecordingRef.current = false;
      setIsRecording(false);
      
      // 清理所有间隔
      if (detectionInterval.current) {
        clearInterval(detectionInterval.current);
        detectionInterval.current = null;
      }
      
      if (audioDataInterval.current) {
        clearInterval(audioDataInterval.current);
        audioDataInterval.current = null;
      }
      
      if (durationInterval.current) {
        clearInterval(durationInterval.current);
        durationInterval.current = null;
      }
      
      // 停止实时音频捕获
      if (AudioCaptureNative) {
        try {
          await AudioCaptureNative.stopCapture();
          console.log('实时音频捕获已停止');
          // 等待一小段时间确保资源完全释放
          await new Promise<void>(resolve => setTimeout(() => resolve(), 100));
        } catch (error: any) {
          console.error('停止实时音频捕获失败:', error);
          // 即使失败也继续，因为可能已经停止了
        }
      }
      
      // 重置音频数据接收标志
      isReadyForAudioDataRef.current = false;
      
      // 停止录音
      const result = await audioRecorderPlayer.stopRecorder();
      audioRecorderPlayer.removeRecordBackListener();
      
      setCurrentStatus('已停止');
      setStatusMessage(`录音已停止，共进行了 ${detectionCountRef.current} 次检测`);
      setRecordingDuration(0);
      audioProcessor.clear();
    } catch (error) {
      console.error('停止录音失败:', error);
      setStatusMessage('停止录音时出错');
    }
  };

  const startSlidingWindowDetection = () => {
    // 重置等待计数和模拟数据标记
    waitCountRef.current = 0;
    hasSimulatedDataRef.current = false;
    
    // 注意：如果AudioCaptureNative可用，音频数据会通过事件监听器自动添加
    // 这里不再需要模拟数据
    // 如果AudioCaptureNative不可用，保留模拟数据作为后备方案
    if (!AudioCaptureNative) {
      console.warn('AudioCaptureNative不可用，使用模拟音频数据');
      audioDataInterval.current = setInterval(() => {
        if (!isRecordingRef.current) {
          if (audioDataInterval.current) {
            clearInterval(audioDataInterval.current);
            audioDataInterval.current = null;
          }
          return;
        }
        
        // 模拟每0.5秒的音频数据（0.5秒 @ 16kHz = 8000 samples）
        const chunkSize = sampleRate * 0.5; // 0.5秒的数据
        const simulatedChunk = new Array(chunkSize).fill(0).map(() => Math.random() * 0.1 - 0.05);
        audioProcessor.addAudioData(simulatedChunk);
      }, 500); // 每0.5秒添加一次数据
    }
    
    // 滑动窗口检测循环（每5秒检测一次）
    detectionInterval.current = setInterval(async () => {
      // 检查是否还在录音（使用ref）
      if (!isRecordingRef.current) {
        if (detectionInterval.current) {
          clearInterval(detectionInterval.current);
          detectionInterval.current = null;
        }
        if (audioDataInterval.current) {
          clearInterval(audioDataInterval.current);
          audioDataInterval.current = null;
        }
        return;
      }
      
      // 如果正在处理中，跳过本次检测
      if (isProcessingRef.current) {
        return;
      }
      
      try {
        // 检查是否有足够的数据（10秒）
        if (audioProcessor.hasEnoughData()) {
          // 获取滑动窗口（最近10秒的数据，应用预处理）
          const audioWindow = audioProcessor.getNextWindow(true);
          
          if (audioWindow && audioWindow.length === windowSize) {
            // 进行检测
            await processAudioWindow(audioWindow);
          } else {
            console.warn('音频窗口数据长度不正确:', audioWindow?.length);
          }
        } else {
          // 数据不足，显示等待信息
          waitCountRef.current++;
          const bufferedDuration = audioProcessor.getBufferedDuration();
          
          if (bufferedDuration < 10) {
            setStatusMessage(`正在收集音频数据... (需要10秒，已收集 ${bufferedDuration.toFixed(1)}秒)`);
          } else {
            setStatusMessage('等待音频数据达到10秒...');
          }
        }
      } catch (error) {
        console.error('检测循环错误:', error);
        setStatusMessage('检测过程中出现错误');
        // 确保错误后重置处理状态
        isProcessingRef.current = false;
        setIsProcessing(false);
      }
    }, 5000); // 每5秒检测一次
  };

  const processAudioWindow = async (audioData: number[]) => {
    // 双重检查，避免重复处理
    if (isProcessingRef.current) {
      console.log('已在处理中，跳过本次检测');
      return;
    }
    
    if (!modelInference.isModelLoaded()) {
      console.warn('模型未加载');
      setStatusMessage('模型未加载，无法进行检测');
      return;
    }

    // 验证音频数据
    if (!audioData || audioData.length !== windowSize) {
      console.warn('音频数据长度不正确:', audioData?.length, '期望:', windowSize);
      return;
    }

    // 设置处理状态（必须在所有检查之后）
    isProcessingRef.current = true;
    setIsProcessing(true);
    
    // 增加检测计数
    detectionCountRef.current += 1;
    const currentCount = detectionCountRef.current;
    setDetectionCount(currentCount);
    
    setStatusMessage(`正在分析音频数据... (第 ${currentCount} 次检测)`);
    
    try {
      console.log(`开始执行模型推理 #${currentCount}，音频数据大小:`, audioData.length);
      
      // 执行模型推理（异步操作）
      const result = await modelInference.predict(audioData);
      
      if (!result) {
        throw new Error('模型推理返回空结果');
      }
      
      if (typeof result.prediction !== 'number' || 
          typeof result.probNormal !== 'number' || 
          typeof result.probApnea !== 'number') {
        throw new Error('模型推理结果格式不正确');
      }
      
      console.log(`模型推理成功 #${currentCount}:`, result);
      
      // 更新状态
      const timestamp = new Date().toLocaleTimeString();
      const detectionResult: DetectionResult = {
        timestamp,
        prediction: result.prediction,
        probNormal: (result.probNormal * 100).toFixed(1),
        probApnea: (result.probApnea * 100).toFixed(1),
      };
      
      // 根据检测结果更新状态
      if (result.prediction === 1) {
        setCurrentStatus('⚠️ 检测到呼吸暂停');
        setStatusMessage(`⚠️ 警告：第 ${currentCount} 次检测发现呼吸暂停！`);
        // 更新apnea计数
        setApneaCount(prev => prev + 1);
      } else {
        setCurrentStatus('✅ 正常呼吸');
        setStatusMessage(`✅ 第 ${currentCount} 次检测：未检测到呼吸暂停，呼吸正常`);
      }
      
      setCurrentConfidence(Math.max(result.probNormal, result.probApnea) * 100);
      
      // 无论结果如何，都添加到结果列表（包括正常结果）
      setDetectionResults(prev => {
        const newResults = [detectionResult, ...prev].slice(0, 20);
        return newResults;
      });
      
      console.log(`检测完成 #${currentCount}:`, detectionResult);
      
    } catch (error) {
      console.error('处理音频失败:', error);
      console.error('错误详情:', error instanceof Error ? error.message : String(error));
      console.error('错误堆栈:', error instanceof Error ? error.stack : '无堆栈信息');
      
      setCurrentStatus('处理错误');
      const errorMsg = error instanceof Error ? error.message : String(error);
      setStatusMessage(`检测失败: ${errorMsg}`);
      
      // 显示错误提示（但不阻塞后续检测）
      Alert.alert(
        '检测错误',
        `第 ${currentCount} 次检测失败: ${errorMsg}\n\n请检查:\n1. 模型是否正确加载\n2. 音频数据是否有效\n3. 查看控制台日志获取详细信息`,
        [{ text: '确定' }]
      );
    } finally {
      // 确保在处理完成后重置状态
      isProcessingRef.current = false;
      setIsProcessing(false);
      
      // 1秒后恢复状态提示
      setTimeout(() => {
        if (isRecordingRef.current && !isProcessingRef.current) {
          setStatusMessage('等待下一次检测...');
        }
      }, 1000);
    }
  };

  const clearResults = () => {
    setDetectionResults([]);
    setCurrentStatus('就绪');
    setCurrentConfidence(0);
    detectionCountRef.current = 0;
    setDetectionCount(0);
    setApneaCount(0);
    setStatusMessage('');
  };

  // 处理降噪开关变化
  const handleDenoiseToggle = (value: boolean) => {
    setEnableDenoise(value);
    audioProcessor.setDenoiseEnabled(value);
    console.log(`降噪已${value ? '启用' : '禁用'}`);
  };

  // 处理滤波开关变化
  const handleBandpassToggle = (value: boolean) => {
    setEnableBandpass(value);
    audioProcessor.setBandpassEnabled(value);
    console.log(`带通滤波已${value ? '启用' : '禁用'}`);
  };

  // 处理防止息屏开关变化
  const handleKeepAwakeToggle = (value: boolean) => {
    setKeepScreenAwake(value);
    if (value) {
      KeepAwake.activate();
      console.log('已启用防止息屏');
    } else {
      KeepAwake.deactivate();
      console.log('已禁用防止息屏');
    }
  };

  // 计算apnea比例和状态
  const getApneaRatio = (): { ratio: number; status: string; color: string; bgColor: string } => {
    if (detectionCount === 0) {
      return { ratio: 0, status: '暂无数据', color: '#666', bgColor: '#f5f5f5' };
    }
    
    const ratio = (apneaCount / detectionCount) * 100;
    
    if (ratio < 33) {
      return { ratio, status: '不怀疑', color: '#4CAF50', bgColor: '#e8f5e9' };
    } else if (ratio <= 66) {
      return { ratio, status: '怀疑', color: '#FF9800', bgColor: '#fff3e0' };
    } else {
      return { ratio, status: '高度怀疑', color: '#F44336', bgColor: '#ffebee' };
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* 防止息屏组件 */}
      {keepScreenAwake && <KeepAwake />}
      <ScrollView style={styles.scrollView}>
        <View style={styles.header}>
          <Text style={styles.title}>睡眠呼吸暂停检测</Text>
          <Text style={styles.subtitle}>实时监测您的呼吸状态</Text>
        </View>

        {/* 状态显示 */}
        <View style={styles.statusContainer}>
          <Text style={styles.statusLabel}>当前状态</Text>
          <Text style={[styles.statusText, currentStatus.includes('⚠️') && styles.warningText]}>
            {currentStatus}
          </Text>
          {isRecording && (
            <Text style={styles.recordingDurationText}>
              录音时长: {Math.floor(recordingDuration / 60)}:{(recordingDuration % 60).toString().padStart(2, '0')}
            </Text>
          )}
          {detectionCount > 0 && (
            <Text style={styles.detectionCountText}>
              检测次数: {detectionCount}
            </Text>
          )}
          {currentConfidence > 0 && (
            <Text style={styles.confidenceText}>
              置信度: {currentConfidence.toFixed(1)}%
            </Text>
          )}
          {statusMessage ? (
            <View style={styles.statusMessageContainer}>
              <Text style={styles.statusMessageText}>{statusMessage}</Text>
            </View>
          ) : null}
          {isProcessing && (
            <View style={styles.processingIndicator}>
              <Text style={styles.processingText}>⏳ 正在处理中...</Text>
            </View>
          )}
        </View>

        {/* Apnea比例显示 */}
        {detectionCount > 0 && (
          <View style={[
            styles.apneaRatioContainer, 
            { 
              backgroundColor: getApneaRatio().bgColor,
              borderColor: getApneaRatio().color,
            }
          ]}>
            <Text style={styles.apneaRatioLabel}>本次录音 Apnea 比例</Text>
            <View style={styles.apneaRatioContent}>
              <Text style={[styles.apneaRatioValue, { color: getApneaRatio().color }]}>
                {getApneaRatio().ratio.toFixed(1)}%
              </Text>
              <Text style={[styles.apneaRatioStatus, { color: getApneaRatio().color }]}>
                {getApneaRatio().status}
              </Text>
            </View>
            <Text style={styles.apneaRatioDetail}>
              ({apneaCount} / {detectionCount} 次检测为 Apnea)
            </Text>
          </View>
        )}

        {/* 音频处理设置 */}
        <View style={styles.settingsContainer}>
          <Text style={styles.settingsTitle}>音频处理设置</Text>
          
          <View style={styles.settingItem}>
            <View style={styles.settingLabelContainer}>
              <Text style={styles.settingLabel}>降噪处理</Text>
              <Text style={styles.settingDescription}>
                减少背景噪声，提高检测准确率
              </Text>
            </View>
            <Switch
              value={enableDenoise}
              onValueChange={handleDenoiseToggle}
              disabled={isRecording}
              trackColor={{ false: '#767577', true: '#4A90E2' }}
              thumbColor={enableDenoise ? '#fff' : '#f4f3f4'}
            />
          </View>

          <View style={styles.settingItem}>
            <View style={styles.settingLabelContainer}>
              <Text style={styles.settingLabel}>带通滤波</Text>
              <Text style={styles.settingDescription}>
                保留100-2000Hz频段，过滤无关频率
              </Text>
            </View>
            <Switch
              value={enableBandpass}
              onValueChange={handleBandpassToggle}
              disabled={isRecording}
              trackColor={{ false: '#767577', true: '#4A90E2' }}
              thumbColor={enableBandpass ? '#fff' : '#f4f3f4'}
            />
          </View>

          <View style={styles.settingItem}>
            <View style={styles.settingLabelContainer}>
              <Text style={styles.settingLabel}>防止息屏</Text>
              <Text style={styles.settingDescription}>
                应用在前台时保持屏幕常亮
              </Text>
            </View>
            <Switch
              value={keepScreenAwake}
              onValueChange={handleKeepAwakeToggle}
              trackColor={{ false: '#767577', true: '#4A90E2' }}
              thumbColor={keepScreenAwake ? '#fff' : '#f4f3f4'}
            />
          </View>
        </View>

        {/* 控制按钮 */}
        <View style={styles.controlContainer}>
          <TouchableOpacity
            style={[styles.button, isRecording && styles.buttonStop]}
            onPress={isRecording ? stopRecording : startRecording}
            disabled={isProcessing}
          >
            <Text style={styles.buttonText}>
              {isRecording ? '停止检测' : '开始检测'}
            </Text>
          </TouchableOpacity>

          {detectionResults.length > 0 && (
            <TouchableOpacity
              style={[styles.button, styles.buttonClear]}
              onPress={clearResults}
            >
              <Text style={styles.buttonText}>清空记录</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* 检测结果列表 */}
        {detectionResults.length > 0 && (
          <View style={styles.resultsContainer}>
            <Text style={styles.resultsTitle}>检测记录</Text>
            {detectionResults.map((result, index) => (
              <View
                key={index}
                style={[
                  styles.resultItem,
                  result.prediction === 1 && styles.resultItemWarning,
                ]}
              >
                <View style={styles.resultHeader}>
                  <Text style={styles.resultTime}>{result.timestamp}</Text>
                  <Text
                    style={[
                      styles.resultStatus,
                      result.prediction === 1 && styles.resultStatusWarning,
                    ]}
                  >
                    {result.prediction === 1 ? '⚠️ 呼吸暂停' : '✅ 正常'}
                  </Text>
                </View>
                <View style={styles.resultDetails}>
                  <Text style={styles.resultDetail}>
                    正常: {result.probNormal}%
                  </Text>
                  <Text style={styles.resultDetail}>
                    呼吸暂停: {result.probApnea}%
                  </Text>
                </View>
              </View>
            ))}
          </View>
        )}

        {/* 使用说明 */}
        <View style={styles.infoContainer}>
          <Text style={styles.infoTitle}>使用说明</Text>
          <Text style={styles.infoText}>
            1. 点击"开始检测"按钮开始录音{'\n'}
            2. 应用将使用滑动窗口算法实时分析音频{'\n'}
            3. 每5秒进行一次检测，显示最近的结果{'\n'}
            4. 检测到呼吸暂停时会显示警告{'\n'}
            5. 点击"停止检测"结束监控
          </Text>
    </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollView: {
    flex: 1,
  },
  header: {
    backgroundColor: '#4A90E2',
    padding: 20,
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 14,
    color: '#fff',
    opacity: 0.9,
  },
  statusContainer: {
    backgroundColor: '#fff',
    margin: 15,
    padding: 20,
    borderRadius: 10,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statusLabel: {
    fontSize: 14,
    color: '#666',
    marginBottom: 10,
  },
  statusText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#4CAF50',
    marginBottom: 5,
  },
  warningText: {
    color: '#F44336',
  },
  recordingDurationText: {
    fontSize: 16,
    color: '#4A90E2',
    marginTop: 8,
    fontWeight: '600',
  },
  detectionCountText: {
    fontSize: 14,
    color: '#666',
    marginTop: 5,
  },
  confidenceText: {
    fontSize: 16,
    color: '#666',
    marginTop: 5,
  },
  statusMessageContainer: {
    marginTop: 15,
    padding: 12,
    backgroundColor: '#f0f7ff',
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#4A90E2',
    width: '100%',
  },
  statusMessageText: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
  processingIndicator: {
    marginTop: 10,
    padding: 8,
    backgroundColor: '#fff3cd',
    borderRadius: 6,
  },
  processingText: {
    fontSize: 14,
    color: '#856404',
    fontWeight: '500',
  },
  controlContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginHorizontal: 15,
    marginBottom: 15,
    gap: 10,
  },
  button: {
    flex: 1,
    backgroundColor: '#4A90E2',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonStop: {
    backgroundColor: '#F44336',
  },
  buttonClear: {
    backgroundColor: '#9E9E9E',
    flex: 0.5,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  resultsContainer: {
    margin: 15,
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  resultsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
  },
  resultItem: {
    padding: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#4CAF50',
    backgroundColor: '#f9f9f9',
    borderRadius: 5,
    marginBottom: 10,
  },
  resultItemWarning: {
    borderLeftColor: '#F44336',
    backgroundColor: '#fff5f5',
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 5,
  },
  resultTime: {
    fontSize: 12,
    color: '#666',
  },
  resultStatus: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#4CAF50',
  },
  resultStatusWarning: {
    color: '#F44336',
  },
  resultDetails: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 5,
  },
  resultDetail: {
    fontSize: 12,
    color: '#666',
  },
  infoContainer: {
    margin: 15,
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  infoText: {
    fontSize: 14,
    color: '#666',
    lineHeight: 22,
  },
  settingsContainer: {
    margin: 15,
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  settingsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
  },
  settingItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  settingLabelContainer: {
    flex: 1,
    marginRight: 15,
  },
  settingLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  settingDescription: {
    fontSize: 12,
    color: '#666',
    lineHeight: 16,
  },
  apneaRatioContainer: {
    margin: 15,
    padding: 20,
    borderRadius: 10,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  apneaRatioLabel: {
    fontSize: 14,
    color: '#666',
    marginBottom: 10,
    fontWeight: '500',
  },
  apneaRatioContent: {
    flexDirection: 'row',
    alignItems: 'baseline',
    justifyContent: 'center',
    marginBottom: 8,
  },
  apneaRatioValue: {
    fontSize: 32,
    fontWeight: 'bold',
    marginRight: 10,
  },
  apneaRatioStatus: {
    fontSize: 20,
    fontWeight: '600',
  },
  apneaRatioDetail: {
    fontSize: 12,
    color: '#666',
    marginTop: 5,
  },
});

export default App;
