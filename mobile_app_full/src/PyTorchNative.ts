/**
 * PyTorchNative.ts
 * PyTorch原生模块的TypeScript接口
 */

import { NativeModules, Platform } from 'react-native';

interface PyTorchNativeModule {
  loadModel(modelPath: string): Promise<boolean>;
  loadPreprocessor(preprocessorPath: string): Promise<boolean>;
  setTemperature(temperature: number): Promise<boolean>;
  predict(melSpectrogram: number[]): Promise<{
    prediction: 0 | 1;
    probNormal: number;
    probApnea: number;
  }>;
  predictFromAudio(audioData: number[]): Promise<{
    prediction: 0 | 1;
    probNormal: number;
    probApnea: number;
  }>;
  isModelLoaded(): Promise<boolean>;
}

const { PyTorchModule } = NativeModules;

// 检查原生模块是否可用
const isNativeModuleAvailable = PyTorchModule != null;

if (!isNativeModuleAvailable) {
  console.warn('PyTorchModule 原生模块不可用');
}

export const PyTorchNative: PyTorchNativeModule | null = 
  isNativeModuleAvailable ? PyTorchModule : null;

export default PyTorchNative;

