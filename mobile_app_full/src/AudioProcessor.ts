/**
 * AudioProcessor.ts
 * 音频处理和滑动窗口算法实现
 */

import AudioPreprocessor from './AudioPreprocessor';

class AudioProcessor {
  private windowSize: number;
  private hopSize: number;
  private sampleRate: number;
  private buffer: number[];
  private maxBufferSize: number;
  private preprocessor: AudioPreprocessor;
  private enableDenoise: boolean;
  private enableBandpass: boolean;

  constructor(
    windowSize = 160000, 
    hopSize = 80000, 
    sampleRate = 16000,
    enableDenoise = false,
    enableBandpass = false
  ) {
    this.windowSize = windowSize; // 10秒 @ 16kHz
    this.hopSize = hopSize; // 5秒步长
    this.sampleRate = sampleRate;
    this.buffer = [];
    this.maxBufferSize = windowSize * 2; // 保留足够大的缓冲区
    this.preprocessor = new AudioPreprocessor(sampleRate);
    this.enableDenoise = enableDenoise;
    this.enableBandpass = enableBandpass;
  }

  /**
   * 设置降噪开关
   */
  setDenoiseEnabled(enabled: boolean): void {
    this.enableDenoise = enabled;
  }

  /**
   * 设置滤波开关
   */
  setBandpassEnabled(enabled: boolean): void {
    this.enableBandpass = enabled;
  }

  /**
   * 获取降噪状态
   */
  isDenoiseEnabled(): boolean {
    return this.enableDenoise;
  }

  /**
   * 获取滤波状态
   */
  isBandpassEnabled(): boolean {
    return this.enableBandpass;
  }

  /**
   * 添加音频数据到缓冲区
   * @param audioData - 音频样本数组
   */
  addAudioData(audioData: number[]): void {
    this.buffer.push(...audioData);
    
    // 限制缓冲区大小，只保留最近的数据
    if (this.buffer.length > this.maxBufferSize) {
      const removeCount = this.buffer.length - this.maxBufferSize;
      this.buffer = this.buffer.slice(removeCount);
    }
  }

  /**
   * 获取下一个滑动窗口的音频数据
   * @param applyPreprocessing - 是否应用预处理（降噪和滤波）
   * @returns 音频窗口数据，如果数据不足则返回null
   */
  getNextWindow(applyPreprocessing: boolean = true): number[] | null {
    if (this.buffer.length < this.windowSize) {
      return null; // 数据不足
    }

    // 提取窗口数据
    let window = this.buffer.slice(0, this.windowSize);
    
    // 应用预处理（如果启用）
    // 注意：这里只做降噪和滤波，不做归一化（归一化在Mel频谱图转换时进行，与PC端一致）
    if (applyPreprocessing && (this.enableDenoise || this.enableBandpass)) {
      window = this.preprocessor.preprocess(window, {
        denoise: this.enableDenoise,
        bandpass: this.enableBandpass,
        normalize: false, // 不在预处理时归一化，与PC端保持一致
        lowcut: 100.0,
        highcut: 2000.0,
        denoiseFactor: 0.9
      });
    }
    
    // 滑动窗口：移除hopSize个样本
    this.buffer = this.buffer.slice(this.hopSize);
    
    return window;
  }

  /**
   * 检查是否有足够的数据进行检测
   */
  hasEnoughData(): boolean {
    return this.buffer.length >= this.windowSize;
  }

  /**
   * 清空缓冲区
   */
  clear(): void {
    this.buffer = [];
  }

  /**
   * 获取当前缓冲区中的数据时长（秒）
   */
  getBufferedDuration(): number {
    return this.buffer.length / this.sampleRate;
  }
}

export default AudioProcessor;

