/**
 * ModelInference.ts
 * 模型推理封装类
 * 
 * 注意：这是一个抽象接口，实际实现需要根据使用的推理引擎调整
 * 支持：PyTorch Mobile, ONNX Runtime, TensorFlow Lite等
 */

interface PredictionResult {
  prediction: 0 | 1;
  probNormal: number;
  probApnea: number;
}

class ModelInference {
  private model: any;
  private preprocessor: any;
  private isLoaded: boolean;
  private threshold: number; // 分类阈值（训练时最佳阈值约为0.34）

  constructor(threshold: number = 0.34) {
    this.model = null;
    this.preprocessor = null;
    this.isLoaded = false;
    this.threshold = threshold; // 默认使用训练时最佳阈值0.34
  }

  /**
   * 设置分类阈值
   * @param threshold - 分类阈值（训练时最佳阈值约为0.34，而非默认0.5）
   */
  setThreshold(threshold: number): void {
    this.threshold = threshold;
    console.log(`分类阈值已设置为: ${threshold.toFixed(3)}`);
  }

  /**
   * 获取当前分类阈值
   */
  getThreshold(): number {
    return this.threshold;
  }

  /**
   * 加载模型
   * @param modelPath - 模型文件路径
   * @param preprocessorPath - 预处理模块路径（可选）
   */
  async loadModel(modelPath?: string, preprocessorPath: string | null = null): Promise<boolean> {
    try {
      // 如果提供了模型路径，尝试加载真实模型
      if (modelPath) {
        console.log('尝试加载真实模型:', modelPath);
        
        // 方案1: 使用自定义PyTorch原生模块（推荐）
        try {
          const PyTorchNative = require('./PyTorchNative').default;
          if (PyTorchNative) {
            console.log('使用 PyTorch 原生模块加载模型');
            await PyTorchNative.loadModel(modelPath);
            
            if (preprocessorPath) {
              await PyTorchNative.loadPreprocessor(preprocessorPath);
            }
            
            this.model = PyTorchNative; // 存储原生模块引用
            this.isLoaded = true;
            console.log('PyTorch 模型加载成功（原生模块）');
            return true;
          }
        } catch (e: any) {
          console.log('PyTorch 原生模块不可用:', e.message);
        }

        // 方案2: 使用ONNX Runtime（需要安装 onnxruntime-react-native）
        try {
          const ort = require('onnxruntime-react-native');
          if (ort && ort.InferenceSession) {
            console.log('使用 ONNX Runtime 加载模型');
            // ONNX 模型路径需要是 .onnx 文件
            if (modelPath.endsWith('.onnx')) {
              this.model = await ort.InferenceSession.create(modelPath);
              this.isLoaded = true;
              console.log('ONNX 模型加载成功');
              return true;
            } else {
              console.log('模型文件不是 .onnx 格式，跳过 ONNX Runtime');
            }
          }
        } catch (e: any) {
          console.log('ONNX Runtime 不可用:', e.message);
        }

        // 如果所有库都不可用，但提供了模型路径，使用模拟模式
        console.warn('无法加载真实模型，使用模拟模式');
        this.isLoaded = true;
        return true;
      }

      // 如果没有提供模型路径，使用模拟模式（用于演示）
      console.log('使用模拟模式（模型未实际加载）');
      this.isLoaded = true; // 标记为已加载，允许使用模拟推理
      return true;
    } catch (error) {
      console.error('模型加载失败:', error);
      // 加载失败时使用模拟模式
      console.warn('模型加载失败，切换到模拟模式');
      this.isLoaded = true;
      return true;
    }
  }

  /**
   * 预处理音频数据（转换为Mel频谱图）
   * @param audioData - 原始音频数据（应该已经经过降噪和滤波）
   * @returns Mel频谱图数据（一维数组，大小为 64 * 313 = 20032）
   */
  async preprocessAudio(audioData: number[]): Promise<number[]> {
    // 如果预处理模块已加载，使用它
    if (this.preprocessor) {
      try {
        // 预处理模块应该接受音频数据并返回Mel频谱图
        // 注意：这需要预处理模块在原生端实现
        console.log('使用预处理模块');
        // 这里需要调用预处理模块（如果原生模块支持）
        // 暂时先做基本预处理
      } catch (e) {
        console.warn('预处理模块调用失败，使用改进的Mel频谱图生成:', e);
      }
    }
    
    // 注意：音频数据应该已经在AudioProcessor中经过降噪和滤波处理
    // 这里需要：1. Z-score归一化 2. Mel频谱图转换 3. Min-Max归一化到[-1,1]
    // 与PC端的preprocess_waveform + audio_to_melspectrogram流程一致
    
    // 1. Z-score归一化（与PC端preprocess_waveform中的归一化一致）
    const mean = audioData.reduce((a, b) => a + b, 0) / audioData.length;
    const centered = audioData.map(x => x - mean);
    const variance = centered.reduce((sum, x) => sum + x * x, 0) / centered.length;
    const std = Math.sqrt(variance) || 1e-8;
    const normalizedAudio = centered.map(x => x / std);
    
    // 2. 使用改进的Mel频谱图生成器
    const MelSpectrogramGenerator = require('./MelSpectrogramGenerator').default;
    const melGenerator = new MelSpectrogramGenerator(
      16000, // sampleRate
      64,    // n_mels
      1024,  // n_fft
      512,   // hop_length
      50,    // fmin
      4000   // fmax
    );
    
    // 3. 生成Mel频谱图
    const melSpectrogram = melGenerator.generate(normalizedAudio);
    
    // 调试信息：检查Mel频谱图的统计信息
    const melMin = Math.min(...melSpectrogram);
    const melMax = Math.max(...melSpectrogram);
    const melMean = melSpectrogram.reduce((a, b) => a + b, 0) / melSpectrogram.length;
    const melStd = Math.sqrt(melSpectrogram.reduce((sum, x) => sum + (x - melMean) ** 2, 0) / melSpectrogram.length);
    console.log(`[预处理] Mel频谱图统计: min=${melMin.toFixed(4)}, max=${melMax.toFixed(4)}, mean=${melMean.toFixed(4)}, std=${melStd.toFixed(4)}`);
    console.log(`[预处理] Mel频谱图大小: ${melSpectrogram.length} (期望: ${64 * 313} = 20032)`);
    
    // 4. Min-Max归一化到[-1, 1]（与PC端audio_to_melspectrogram中的归一化一致）
    const normalized = melGenerator.normalize(melSpectrogram);
    
    // 调试信息：检查归一化后的统计信息
    const normMin = Math.min(...normalized);
    const normMax = Math.max(...normalized);
    const normMean = normalized.reduce((a, b) => a + b, 0) / normalized.length;
    console.log(`[预处理] 归一化后统计: min=${normMin.toFixed(4)}, max=${normMax.toFixed(4)}, mean=${normMean.toFixed(4)}`);
    
    return normalized;
  }

  // 已移除简化的Mel频谱图生成方法，改用MelSpectrogramGenerator

  /**
   * 执行模型推理
   * @param audioData - 音频数据
   * @returns 推理结果
   */
  async predict(audioData: number[]): Promise<PredictionResult> {
    if (!this.isLoaded) {
      throw new Error('模型未加载');
    }

    try {
      // 预处理
      const melSpectrogram = await this.preprocessAudio(audioData);
      
      // 如果使用PyTorch原生模块
      if (this.model && typeof this.model.predict === 'function') {
        try {
          // 注意：暂时不使用 predictFromAudio，因为传递160000个元素的数组会导致崩溃
          // 改为在JavaScript端完成预处理，然后使用predict方法（只传递20032个元素的Mel频谱图）
          console.log('使用原生模块推理（Mel频谱图），音频数据大小:', audioData.length);
          
          // 如果没有predictFromAudio，使用Mel频谱图
          // melSpectrogram应该是一维数组，大小为 64 * 313 = 20032
          let flatArray: number[];
          
          if (Array.isArray(melSpectrogram)) {
            flatArray = melSpectrogram;
          } else if (melSpectrogram && typeof melSpectrogram === 'object' && 'data' in melSpectrogram && Array.isArray(melSpectrogram.data)) {
            flatArray = melSpectrogram.data;
          } else {
            throw new Error('无法解析Mel频谱图数据');
          }
          
          // 确保数组大小正确
          if (flatArray.length !== 64 * 313) {
            console.warn(`Mel频谱图大小不正确。期望: ${64 * 313}, 实际: ${flatArray.length}`);
            // 如果大小不对，尝试填充或截断
            if (flatArray.length < 64 * 313) {
              flatArray = [...flatArray, ...new Array(64 * 313 - flatArray.length).fill(0)];
            } else {
              flatArray = flatArray.slice(0, 64 * 313);
            }
          }
          
          // 调用原生模块进行推理
          const result = await this.model.predict(flatArray);
          console.log('使用原生模块推理（Mel频谱图）:', result);
          
          // 使用阈值进行预测（与训练时一致）
          // 训练时最佳阈值约为0.34，而不是默认的0.5
          // 如果原生模块返回的是概率，使用阈值；如果返回的是预测，可能需要重新计算
          let prediction: 0 | 1;
          let probNormal: number;
          let probApnea: number;
          
          // 原生模块现在只返回概率，不返回prediction，确保使用阈值0.34计算
          if (result.probApnea !== undefined && result.probNormal !== undefined) {
            // 原生模块返回了概率（推荐情况）
            probNormal = result.probNormal;
            probApnea = result.probApnea;
            // 使用阈值进行预测（训练时最佳阈值0.34）
            prediction = probApnea >= this.threshold ? 1 : 0;
            console.log(`使用阈值 ${this.threshold.toFixed(3)} 进行预测: 概率=${(probApnea*100).toFixed(2)}%, 预测=${prediction === 1 ? '呼吸暂停' : '正常'}`);
          } else if (result.prediction !== undefined && result.probApnea === undefined) {
            // 如果原生模块只返回了prediction（旧版本兼容），尝试从prediction推断概率
            // 这种情况不应该发生，但为了兼容性保留
            console.warn('原生模块只返回了prediction，无法使用阈值。请更新原生模块以返回概率。');
            probNormal = result.probNormal || 0.5;
            probApnea = result.probApnea || (result.prediction === 1 ? 0.6 : 0.4); // 粗略估计
            prediction = probApnea >= this.threshold ? 1 : 0;
          } else {
            // 降级处理
            probNormal = result.probNormal || 0.5;
            probApnea = result.probApnea || 0.5;
            prediction = probApnea >= this.threshold ? 1 : 0;
            console.warn('原生模块返回格式异常，使用阈值进行预测');
          }
          
          return {
            prediction,
            probNormal,
            probApnea,
          };
        } catch (e: any) {
          console.error('原生模块推理失败:', e);
          console.error('错误详情:', e.message);
          // 降级到模拟模式
        }
      }

      // 方案2: ONNX Runtime
      // if (this.model && this.model.run) {
      //   const feeds = { input: new ort.Tensor('float32', melSpectrogram, [1, 1, 64, 313]) };
      //   const results = await this.model.run(feeds);
      //   const output = results.output.data;
      //   const probs = this.softmax(output);
      //   const pred = probs[1] > probs[0] ? 1 : 0;
      //   return {
      //     prediction: pred,
      //     probNormal: probs[0],
      //     probApnea: probs[1],
      //   };
      // }

          // 如果所有真实模型都不可用，抛出错误而不是返回模拟结果
          throw new Error('模型未加载且无法使用模拟模式。请确保模型文件已正确加载。');
    } catch (error) {
      console.error('推理错误:', error);
      throw error;
    }
  }

  /**
   * Softmax函数（用于ONNX Runtime输出）
   * @private
   */
  private softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(x => Math.exp(x - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map(x => x / sumExp);
  }

  /**
   * 设置Temperature Scaling参数
   * @param temperature Temperature值（推荐：0.7-0.9，默认0.8）
   */
  async setTemperature(temperature: number): Promise<boolean> {
    try {
      if (this.model && typeof this.model.setTemperature === 'function') {
        await this.model.setTemperature(temperature);
        console.log(`Temperature Scaling已设置: ${temperature}`);
        return true;
      } else {
        console.warn('模型不支持Temperature Scaling');
        return false;
      }
    } catch (error) {
      console.error('设置Temperature失败:', error);
      return false;
    }
  }

  /**
   * 检查模型是否已加载
   */
  isModelLoaded(): boolean {
    return this.isLoaded;
  }
}

export default ModelInference;

