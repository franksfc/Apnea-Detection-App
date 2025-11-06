/**
 * AudioCaptureNative.ts
 * 实时音频捕获原生模块的TypeScript接口
 */

import { NativeModules, NativeEventEmitter } from 'react-native';

interface AudioCaptureNativeModule {
  startCapture(): Promise<boolean>;
  stopCapture(): Promise<boolean>;
}

const { AudioCaptureModule } = NativeModules;

const isNativeModuleAvailable = AudioCaptureModule != null;

if (!isNativeModuleAvailable) {
  console.warn('AudioCaptureModule 原生模块不可用');
}

export const AudioCaptureNative: AudioCaptureNativeModule | null = 
  isNativeModuleAvailable ? AudioCaptureModule : null;

// 创建事件发射器
export const audioCaptureEventEmitter = isNativeModuleAvailable
  ? new NativeEventEmitter(AudioCaptureModule)
  : null;

export default AudioCaptureNative;






