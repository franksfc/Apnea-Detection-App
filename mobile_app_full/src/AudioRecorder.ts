/**
 * AudioRecorder.ts
 * 音频录制封装类 - 优先使用原生模块
 * 
 * 使用自定义原生模块实现音频录制，兼容 React Native 0.82+
 */

import { Platform } from 'react-native';
import RNFS from 'react-native-fs';
import AudioRecorderNative from './AudioRecorderNative';

class AudioRecorder {
  private isRecording: boolean = false;
  private recordPath: string = '';
  private listeners: Array<(data: any) => void> = [];
  private progressInterval: ReturnType<typeof setInterval> | null = null;

  constructor() {
    // 原生模块已通过 AudioRecorderNative 导入
    // 不需要额外初始化
  }

  /**
   * 开始录音
   */
  async startRecorder(
    path: string,
    audioSet?: any
  ): Promise<string> {
    // 如果已经在录音，先尝试停止（确保资源已释放）
    if (this.isRecording) {
      console.warn('检测到已在录音，先停止...');
      try {
        await this.stopRecorder();
        // 等待一小段时间确保资源完全释放
        await new Promise<void>(resolve => setTimeout(() => resolve(), 200));
      } catch (e) {
        // 如果停止失败，强制重置状态
        console.warn('停止之前的录音失败，强制重置状态');
        this.isRecording = false;
        if (this.progressInterval) {
          clearInterval(this.progressInterval);
          this.progressInterval = null;
        }
      }
    }

    this.recordPath = path;

    // 优先使用原生模块
    if (AudioRecorderNative) {
      try {
        const uri = await AudioRecorderNative.startRecording(path);
        this.isRecording = true;
        this.recordPath = uri;
        
        // 启动进度监听（模拟，原生模块不提供实时进度）
        this.startProgressSimulation();
        
        return uri;
      } catch (error: any) {
        console.error('原生录音启动失败:', error);
        // 如果是因为已经在录音，尝试先停止再重试
        if (error.code === 'ALREADY_RECORDING' || error.message?.includes('录音已在进行中')) {
          console.log('检测到遗留的录音状态，尝试清理后重试...');
          try {
            await AudioRecorderNative.stopRecording();
            await new Promise<void>(resolve => setTimeout(() => resolve(), 200));
            // 重试一次
            const uri = await AudioRecorderNative.startRecording(path);
            this.isRecording = true;
            this.recordPath = uri;
            this.startProgressSimulation();
            return uri;
          } catch (retryError: any) {
            console.error('重试启动录音失败:', retryError);
            throw retryError;
          }
        }
        throw error;
      }
    } else {
      // 原生模块不可用，使用模拟模式
      console.warn('原生录音模块不可用，使用模拟模式');
      return this.startSimulatedRecording(path);
    }
  }

  /**
   * 启动进度模拟（原生模块不提供实时进度，使用模拟）
   */
  private startProgressSimulation(): void {
    this.progressInterval = setInterval(() => {
      if (!this.isRecording) {
        if (this.progressInterval) {
          clearInterval(this.progressInterval);
          this.progressInterval = null;
        }
        return;
      }

      // 模拟进度更新
      this.listeners.forEach(listener => {
        listener({
          currentPosition: Date.now() % 10000,
          currentMetering: Math.random() * 100,
        });
      });
    }, 1000);
  }

  /**
   * 模拟录音（当原生模块不可用时）
   */
  private async startSimulatedRecording(path: string): Promise<string> {
    this.isRecording = true;
    this.recordPath = path;
    console.log('开始录音（模拟模式）:', path);
    console.warn('注意：这是模拟实现，原生模块未正确加载');

    // 启动进度模拟
    this.startProgressSimulation();

    return path;
  }

  /**
   * 停止录音
   */
  async stopRecorder(): Promise<string> {
    const wasRecording = this.isRecording;
    this.isRecording = false;

    // 清除进度监听
    if (this.progressInterval) {
      clearInterval(this.progressInterval);
      this.progressInterval = null;
    }

    // 如果使用原生模块，调用停止方法（使用之前的状态检查）
    if (AudioRecorderNative && wasRecording) {
      try {
        const result = await AudioRecorderNative.stopRecording();
        return result || this.recordPath;
      } catch (error: any) {
        console.error('停止录音失败:', error);
        // 即使失败也继续，因为可能已经停止了
      }
    }

    console.log('停止录音:', this.recordPath);
    return this.recordPath;
  }

  /**
   * 添加录音回调监听器
   */
  addRecordBackListener(callback: (data: any) => void): void {
    this.listeners.push(callback);
  }

  /**
   * 移除录音回调监听器
   */
  removeRecordBackListener(): void {
    this.listeners = [];
  }

  /**
   * 检查是否正在录音
   */
  isRecordingNow(): boolean {
    return this.isRecording;
  }

  /**
   * 获取当前录音路径
   */
  getRecordPath(): string {
    return this.recordPath;
  }
}

export default AudioRecorder;
