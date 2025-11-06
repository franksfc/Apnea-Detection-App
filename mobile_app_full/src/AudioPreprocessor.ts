/**
 * AudioPreprocessor.ts
 * 音频预处理工具：降噪和滤波
 */

class AudioPreprocessor {
  private sampleRate: number;
  
  constructor(sampleRate: number = 16000) {
    this.sampleRate = sampleRate;
  }

  /**
   * 带通滤波器（Butterworth）
   * @param audioData - 输入音频数据
   * @param lowcut - 低截止频率 (Hz)
   * @param highcut - 高截止频率 (Hz)
   * @param order - 滤波器阶数
   * @returns 滤波后的音频数据
   */
  bandpassFilter(
    audioData: number[],
    lowcut: number = 100.0,
    highcut: number = 2000.0,
    order: number = 4
  ): number[] {
    // 简化的带通滤波实现（移动端优化版本）
    // 注意：完整的Butterworth滤波器需要复杂的IIR实现
    // 这里使用简化的频率域滤波
    
    const nyquist = this.sampleRate / 2;
    const lowNorm = lowcut / nyquist;
    const highNorm = highcut / nyquist;
    
    // 使用简单的移动平均和频率域滤波组合
    // 这是一个简化实现，实际应用中可以使用原生模块实现完整的Butterworth滤波器
    
    // 1. 简单的低通滤波（去除高频噪声）
    const filtered = this._simpleLowPass(audioData, highNorm);
    
    // 2. 简单的高通滤波（去除低频噪声）
    const result = this._simpleHighPass(filtered, lowNorm);
    
    return result;
  }

  /**
   * 简化的低通滤波
   * @private
   */
  private _simpleLowPass(audioData: number[], cutoff: number): number[] {
    // 使用移动平均作为简化的低通滤波
    const windowSize = Math.max(1, Math.floor(1 / cutoff));
    const result: number[] = [];
    
    for (let i = 0; i < audioData.length; i++) {
      const start = Math.max(0, i - Math.floor(windowSize / 2));
      const end = Math.min(audioData.length, i + Math.floor(windowSize / 2));
      let sum = 0;
      for (let j = start; j < end; j++) {
        sum += audioData[j];
      }
      result.push(sum / (end - start));
    }
    
    return result;
  }

  /**
   * 简化的高通滤波
   * @private
   */
  private _simpleHighPass(audioData: number[], cutoff: number): number[] {
    // 使用一阶高通滤波器（简化版）
    const alpha = 1 / (1 + 2 * Math.PI * cutoff);
    const result: number[] = [];
    let prevInput = 0;
    let prevOutput = 0;
    
    for (let i = 0; i < audioData.length; i++) {
      const input = audioData[i];
      const output = alpha * (prevOutput + input - prevInput);
      result.push(output);
      prevInput = input;
      prevOutput = output;
    }
    
    return result;
  }

  /**
   * 降噪处理（基于频谱门控的简化实现）
   * @param audioData - 输入音频数据
   * @param reductionFactor - 降噪强度 (0.0-1.0)
   * @returns 降噪后的音频数据
   */
  denoise(
    audioData: number[],
    reductionFactor: number = 0.9
  ): number[] {
    // 简化的降噪实现
    // 实际应用中，完整的降噪算法（如spectral gating）需要在原生模块中实现
    // 这里使用基于统计的简单降噪
    
    // 1. 计算信号统计量
    const mean = audioData.reduce((a, b) => a + b, 0) / audioData.length;
    const variance = audioData.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / audioData.length;
    const std = Math.sqrt(variance);
    
    // 2. 估计噪声水平（假设噪声是低幅度的）
    const noiseThreshold = std * 0.3; // 噪声阈值
    
    // 3. 应用软阈值降噪
    const result = audioData.map(sample => {
      const absValue = Math.abs(sample - mean);
      if (absValue < noiseThreshold) {
        // 低于阈值的部分被认为是噪声，进行衰减
        const attenuation = 1 - reductionFactor * (1 - absValue / noiseThreshold);
        return mean + (sample - mean) * attenuation;
      } else {
        // 高于阈值的部分保留
        return sample;
      }
    });
    
    return result;
  }

  /**
   * 归一化处理
   * @param audioData - 输入音频数据
   * @returns 归一化后的音频数据
   */
  normalize(audioData: number[]): number[] {
    // Z-score归一化
    const mean = audioData.reduce((a, b) => a + b, 0) / audioData.length;
    const centered = audioData.map(x => x - mean);
    const variance = centered.reduce((sum, x) => sum + x * x, 0) / centered.length;
    const std = Math.sqrt(variance) || 1;
    
    return centered.map(x => x / std);
  }

  /**
   * 完整的预处理流程
   * @param audioData - 原始音频数据
   * @param options - 预处理选项
   * @returns 预处理后的音频数据
   */
  preprocess(
    audioData: number[],
    options: {
      denoise?: boolean;
      bandpass?: boolean;
      normalize?: boolean;
      lowcut?: number;
      highcut?: number;
      denoiseFactor?: number;
    } = {}
  ): number[] {
    let processed = [...audioData];
    
    // 应用降噪
    if (options.denoise) {
      processed = this.denoise(processed, options.denoiseFactor || 0.9);
    }
    
    // 应用带通滤波
    if (options.bandpass) {
      processed = this.bandpassFilter(
        processed,
        options.lowcut || 100.0,
        options.highcut || 2000.0
      );
    }
    
    // 归一化（通常最后进行）
    if (options.normalize !== false) {
      processed = this.normalize(processed);
    }
    
    return processed;
  }
}

export default AudioPreprocessor;



