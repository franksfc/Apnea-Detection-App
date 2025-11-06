/**
 * MelSpectrogramGenerator.ts
 * 改进的Mel频谱图生成器，更接近PC端的实现
 * 
 * 基于FFT和Mel滤波器组的实现，与predict_microphone.py保持一致
 */

class MelSpectrogramGenerator {
  private sampleRate: number;
  private n_mels: number;
  private n_fft: number;
  private hop_length: number;
  private fmin: number;
  private fmax: number;
  private melFilters: number[][]; // Mel滤波器组

  constructor(
    sampleRate: number = 16000,
    n_mels: number = 64,
    n_fft: number = 1024,
    hop_length: number = 512,
    fmin: number = 50,
    fmax: number = 4000
  ) {
    this.sampleRate = sampleRate;
    this.n_mels = n_mels;
    this.n_fft = n_fft;
    this.hop_length = hop_length;
    this.fmin = fmin;
    this.fmax = fmax;
    
    // 预计算Mel滤波器组
    this.melFilters = this._createMelFilterbank();
  }

  /**
   * 创建Mel滤波器组
   * @private
   */
  private _createMelFilterbank(): number[][] {
    const nyquist = this.sampleRate / 2;
    const n_freqs = Math.floor(this.n_fft / 2) + 1;
    
    // 将频率转换为Mel刻度
    const melMin = this._hzToMel(this.fmin);
    const melMax = this._hzToMel(this.fmax);
    
    // 在Mel刻度上均匀分布
    const melPoints = [];
    for (let i = 0; i <= this.n_mels + 1; i++) {
      melPoints.push(melMin + (melMax - melMin) * (i / (this.n_mels + 1)));
    }
    
    // 转换回Hz
    const hzPoints = melPoints.map(mel => this._melToHz(mel));
    
    // 转换为FFT bin索引
    const fftBins = hzPoints.map(hz => Math.floor((hz / nyquist) * n_freqs));
    
    // 创建滤波器组
    const filters: number[][] = [];
    for (let i = 0; i < this.n_mels; i++) {
      const filter = new Array(n_freqs).fill(0);
      const start = fftBins[i];
      const center = fftBins[i + 1];
      const end = fftBins[i + 2];
      
      // 上升沿
      for (let j = start; j < center; j++) {
        if (j >= 0 && j < n_freqs) {
          filter[j] = (j - start) / (center - start);
        }
      }
      
      // 下降沿
      for (let j = center; j < end; j++) {
        if (j >= 0 && j < n_freqs) {
          filter[j] = (end - j) / (end - center);
        }
      }
      
      filters.push(filter);
    }
    
    return filters;
  }

  /**
   * Hz转Mel
   * @private
   */
  private _hzToMel(hz: number): number {
    return 2595 * Math.log10(1 + hz / 700);
  }

  /**
   * Mel转Hz
   * @private
   */
  private _melToHz(mel: number): number {
    return 700 * (Math.pow(10, mel / 2595) - 1);
  }

  /**
   * 计算FFT（使用迭代FFT算法，性能更好）
   * @private
   */
  private _fft(signal: number[]): { real: number[]; imag: number[] } {
    const N = signal.length;
    
    // 如果长度不是2的幂，需要零填充到1024（n_fft）
    let n = this.n_fft;
    const real = [...signal];
    const imag = new Array(n).fill(0);
    
    if (N < n) {
      real.push(...new Array(n - N).fill(0));
    } else if (N > n) {
      real.splice(n);
    }
    
    // 使用迭代FFT（比递归更快）
    return this._iterativeFFT(real, imag, n);
  }

  /**
   * 迭代FFT实现（性能优化版本）
   * @private
   */
  private _iterativeFFT(real: number[], imag: number[], n: number): { real: number[]; imag: number[] } {
    // 位反转
    let j = 0;
    for (let i = 1; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) {
        j ^= bit;
      }
      j ^= bit;
      
      if (i < j) {
        [real[i], real[j]] = [real[j], real[i]];
        [imag[i], imag[j]] = [imag[j], imag[i]];
      }
    }
    
    // 迭代计算FFT
    for (let len = 2; len <= n; len <<= 1) {
      const angle = -2 * Math.PI / len;
      const wlenReal = Math.cos(angle);
      const wlenImag = Math.sin(angle);
      
      for (let i = 0; i < n; i += len) {
        let wReal = 1;
        let wImag = 0;
        
        for (let j = 0; j < len / 2; j++) {
          const uReal = real[i + j];
          const uImag = imag[i + j];
          const vReal = real[i + j + len / 2] * wReal - imag[i + j + len / 2] * wImag;
          const vImag = real[i + j + len / 2] * wImag + imag[i + j + len / 2] * wReal;
          
          real[i + j] = uReal + vReal;
          imag[i + j] = uImag + vImag;
          real[i + j + len / 2] = uReal - vReal;
          imag[i + j + len / 2] = uImag - vImag;
          
          const nextWReal = wReal * wlenReal - wImag * wlenImag;
          const nextWImag = wReal * wlenImag + wImag * wlenReal;
          wReal = nextWReal;
          wImag = nextWImag;
        }
      }
    }
    
    return { real, imag };
  }

  /**
   * 计算幅度谱（与torchaudio MelSpectrogram一致）
   * torchaudio的MelSpectrogram默认输出magnitude spectrum，不是power spectrum
   * 
   * 注意：torchaudio的STFT默认normalized=False，所以不需要除以n_fft
   * 但是，窗函数的影响需要考虑，窗函数的能量归一化可能已经在MelSpectrogram内部处理
   * @private
   */
  private _magnitudeSpectrum(fft: { real: number[]; imag: number[] }): number[] {
    const n = fft.real.length;
    const magnitude = [];
    
    // 只取前n_fft/2+1个频率点
    const n_freqs = Math.floor(this.n_fft / 2) + 1;
    
    // torchaudio的STFT默认normalized=False，所以直接使用FFT输出
    // 不需要除以n_fft，因为torchaudio的MelSpectrogram内部会处理归一化
    for (let i = 0; i < n_freqs; i++) {
      // 计算幅度（magnitude），不进行归一化
      const mag = Math.sqrt(fft.real[i] ** 2 + fft.imag[i] ** 2);
      magnitude.push(mag);
    }
    
    return magnitude;
  }

  /**
   * 应用Mel滤波器组（对magnitude spectrum应用）
   * @private
   */
  private _applyMelFilters(magnitudeSpectrum: number[]): number[] {
    const melSpectrum = [];
    
    for (const filter of this.melFilters) {
      let energy = 0;
      for (let i = 0; i < Math.min(filter.length, magnitudeSpectrum.length); i++) {
        energy += filter[i] * magnitudeSpectrum[i];
      }
      melSpectrum.push(energy);
    }
    
    return melSpectrum;
  }

  /**
   * 转换为dB刻度（与torchaudio AmplitudeToDB一致）
   * @private
   * 
   * torchaudio的AmplitudeToDB实现：
   * - 对于amplitude (magnitude spectrum): db = 20 * log10(max(amplitude, ref) / ref)
   * - 对于power spectrum: db = 10 * log10(max(power, ref) / ref)
   * - 然后clip到 [max_db - top_db, max_db]，其中max_db是每个样本的最大值
   * 
   * 注意：
   * 1. torchaudio的MelSpectrogram默认输出magnitude spectrum，所以AmplitudeToDB使用20*log10
   * 2. ref的默认值是1.0，但实际计算时需要考虑信号的幅度范围
   * 3. 如果amplitude < ref，log10会得到负数，这是正常的（dB值通常是负数）
   */
  private _toDB(melSpectrum: number[]): number[] {
    const ref = 1.0;
    const topDB = 80.0;
    
    // 先转换为dB
    // torchaudio的MelSpectrogram输出magnitude，AmplitudeToDB使用20*log10（对于amplitude）
    // 公式：db = 20 * log10(max(amplitude, ref) / ref)
    // 如果amplitude < ref，db会是负数（这是正常的）
    const dbArray = melSpectrum.map(amplitude => {
      // 使用20*log10因为这是magnitude spectrum（幅度），不是power spectrum
      // 注意：如果amplitude很小（<1.0），log10会得到负数，这是正常的dB值
      const normalizedAmplitude = Math.max(amplitude, 1e-10) / ref;
      return 20 * Math.log10(normalizedAmplitude);
    });
    
    // 找到最大值
    const maxDB = Math.max(...dbArray);
    
    // Clip到 [max_db - top_db, max_db]
    const minDB = maxDB - topDB;
    
    return dbArray.map(db => {
      // Clip到范围
      return Math.max(minDB, Math.min(db, maxDB));
    });
  }

  /**
   * 生成Mel频谱图
   * @param audioData - 音频数据（已预处理）
   * @returns Mel频谱图（一维数组，大小为 n_mels * time_steps）
   */
  generate(audioData: number[]): number[] {
    // torchaudio的MelSpectrogram默认center=True，会在两端添加n_fft//2的padding
    // 这会导致时间步数增加
    // 计算方式：time_steps = (audio_length + n_fft - n_fft) // hop_length + 1
    // 对于160000个样本：time_steps = 160000 // 512 + 1 = 312 + 1 = 313
    const paddedLength = audioData.length + this.n_fft;
    const timeSteps = Math.floor((paddedLength - this.n_fft) / this.hop_length) + 1;
    const melSpectrogram: number[] = [];
    
    // 计算padding大小（center=True时，两端各添加n_fft//2）
    const padLeft = Math.floor(this.n_fft / 2);
    
    // 对每个时间窗口计算Mel频谱
    for (let t = 0; t < timeSteps; t++) {
      // 计算窗口中心位置（在padding后的信号中）
      const windowCenter = t * this.hop_length;
      // 计算窗口在原始音频中的起始位置（考虑左padding）
      const windowStart = windowCenter - padLeft;
      
      // 提取窗口（处理边界情况，使用reflect填充模式，与PC端torchaudio一致）
      const window: number[] = [];
      for (let i = 0; i < this.n_fft; i++) {
        const sampleIndex = windowStart + i;
        if (sampleIndex < 0) {
          // 左边界：reflect填充（镜像）
          // 对于负索引，reflect模式会镜像：-1 -> 0, -2 -> 1, -3 -> 2, ...
          const reflectIndex = -sampleIndex - 1;
          if (reflectIndex < audioData.length) {
            window.push(audioData[reflectIndex]);
          } else {
            // 如果镜像索引也超出范围，继续镜像
            window.push(audioData[2 * audioData.length - reflectIndex - 1]);
          }
        } else if (sampleIndex >= audioData.length) {
          // 右边界：reflect填充（镜像）
          // 对于超出范围的索引，reflect模式会镜像：len -> len-1, len+1 -> len-2, ...
          const reflectIndex = 2 * audioData.length - sampleIndex - 1;
          if (reflectIndex >= 0) {
            window.push(audioData[reflectIndex]);
          } else {
            // 如果镜像索引也超出范围，继续镜像
            window.push(audioData[-reflectIndex - 1]);
          }
        } else {
          // 正常范围：使用实际数据
          window.push(audioData[sampleIndex]);
        }
      }
      
      // 应用窗函数（Hanning窗）
      const windowed = this._applyWindow(window);
      
      // FFT
      const fft = this._fft(windowed);
      
      // 幅度谱（与torchaudio MelSpectrogram一致，输出magnitude不是power）
      const magnitude = this._magnitudeSpectrum(fft);
      
      // 调试信息：打印幅度谱的范围（仅第一个时间步）
      if (t === 0) {
        const magMin = Math.min(...magnitude);
        const magMax = Math.max(...magnitude);
        const magMean = magnitude.reduce((a, b) => a + b, 0) / magnitude.length;
        console.log(`[Mel生成] 时间步0的幅度谱: min=${magMin.toFixed(6)}, max=${magMax.toFixed(6)}, mean=${magMean.toFixed(6)}`);
      }
      
      // 应用Mel滤波器组（对magnitude spectrum应用）
      const melSpectrum = this._applyMelFilters(magnitude);
      
      // 调试信息：打印Mel频谱的范围（仅第一个时间步）
      if (t === 0) {
        const melMin = Math.min(...melSpectrum);
        const melMax = Math.max(...melSpectrum);
        const melMean = melSpectrum.reduce((a, b) => a + b, 0) / melSpectrum.length;
        console.log(`[Mel生成] 时间步0的Mel频谱（应用滤波器后）: min=${melMin.toFixed(6)}, max=${melMax.toFixed(6)}, mean=${melMean.toFixed(6)}`);
      }
      
      // 转换为dB
      const melDB = this._toDB(melSpectrum);
      
      // 调试信息：打印每个时间步的dB值范围（仅第一个时间步）
      if (t === 0) {
        const dbMin = Math.min(...melDB);
        const dbMax = Math.max(...melDB);
        const dbMean = melDB.reduce((a, b) => a + b, 0) / melDB.length;
        console.log(`[Mel生成] 时间步0的dB值: min=${dbMin.toFixed(4)}, max=${dbMax.toFixed(4)}, mean=${dbMean.toFixed(4)}`);
      }
      
      // 添加到频谱图
      melSpectrogram.push(...melDB);
    }
    
    // 如果长度不够，填充0
    const expectedLength = this.n_mels * timeSteps;
    while (melSpectrogram.length < expectedLength) {
      melSpectrogram.push(0);
    }
    
    // 如果太长，截断
    return melSpectrogram.slice(0, expectedLength);
  }

  /**
   * 应用窗函数（Hanning窗）
   * @private
   */
  private _applyWindow(signal: number[]): number[] {
    const windowed = [];
    const n = signal.length;
    
    for (let i = 0; i < n; i++) {
      const windowValue = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));
      windowed.push(signal[i] * windowValue);
    }
    
    return windowed;
  }

  /**
   * 归一化Mel频谱图到[-1, 1]（与PC端一致）
   * @param melSpectrogram - 原始Mel频谱图
   * @returns 归一化后的Mel频谱图
   */
  normalize(melSpectrogram: number[]): number[] {
    // Min-Max归一化到[-1, 1]（与PC端一致）
    let min = Infinity;
    let max = -Infinity;
    
    for (const value of melSpectrogram) {
      if (value < min) min = value;
      if (value > max) max = value;
    }
    
    const range = max - min || 1e-8;
    
    return melSpectrogram.map(value => {
      // 归一化到[0, 1]，然后映射到[-1, 1]
      return 2.0 * ((value - min) / range) - 1.0;
    });
  }
}

export default MelSpectrogramGenerator;

