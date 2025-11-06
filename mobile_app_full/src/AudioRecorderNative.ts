/**
 * AudioRecorderNative.ts
 * 原生音频录制模块的TypeScript接口
 */

import { NativeModules, Platform } from 'react-native';

interface AudioRecorderNativeModule {
  startRecording(path: string): Promise<string>;
  stopRecording(): Promise<string>;
  isRecording(): Promise<boolean>;
}

const { AudioRecorderModule } = NativeModules;

// 检查原生模块是否可用
const isNativeModuleAvailable = AudioRecorderModule != null;

if (!isNativeModuleAvailable) {
  console.warn('AudioRecorderModule 原生模块不可用');
}

export const AudioRecorderNative: AudioRecorderNativeModule | null = 
  isNativeModuleAvailable ? AudioRecorderModule : null;

export default AudioRecorderNative;


