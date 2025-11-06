package com.mobile_app_full

import kotlin.math.*

/**
 * MelSpectrogramConverter - 完整的Mel频谱图转换器
 * 实现与训练时相同的Mel频谱图转换（使用FFT和Mel滤波器组）
 * 
 * 参数（与训练时一致）：
 * - sample_rate: 16000
 * - n_mels: 64
 * - n_fft: 1024
 * - hop_length: 512
 * - fmin: 50
 * - fmax: 4000
 */
class MelSpectrogramConverter(
    private val sampleRate: Int = 16000,
    private val nMels: Int = 64,
    private val nFft: Int = 1024,
    private val hopLength: Int = 512,
    private val fMin: Float = 50f,
    private val fMax: Float = 4000f
) {
    
    // Mel滤波器组（预计算）
    private val melFilterBank: Array<FloatArray> by lazy {
        createMelFilterBank()
    }
    
    /**
     * 创建Mel滤波器组
     */
    private fun createMelFilterBank(): Array<FloatArray> {
        val filterBank = Array(nMels) { FloatArray(nFft / 2 + 1) }
        
        // 计算Mel频率范围
        val melMin = hzToMel(fMin)
        val melMax = hzToMel(fMax)
        val melPoints = FloatArray(nMels + 2)
        for (i in 0 until nMels + 2) {
            melPoints[i] = melMin + (melMax - melMin) * i / (nMels + 1)
        }
        
        // 转换为Hz
        val hzPoints = FloatArray(nMels + 2)
        for (i in 0 until nMels + 2) {
            hzPoints[i] = melToHz(melPoints[i])
        }
        
        // 转换为FFT bin索引
        val fftBins = FloatArray(nMels + 2)
        for (i in 0 until nMels + 2) {
            fftBins[i] = (hzPoints[i] * nFft / sampleRate)
        }
        
        // 创建三角形滤波器
        for (m in 0 until nMels) {
            val left = fftBins[m].toInt()
            val center = fftBins[m + 1].toInt()
            val right = fftBins[m + 2].toInt()
            
            // 上升部分
            for (k in left until center) {
                if (k < filterBank[m].size && center > left) {
                    filterBank[m][k] = (k - left).toFloat() / (center - left)
                }
            }
            
            // 下降部分
            for (k in center until right) {
                if (k < filterBank[m].size && right > center) {
                    filterBank[m][k] = (right - k).toFloat() / (right - center)
                }
            }
        }
        
        return filterBank
    }
    
    /**
     * Hz转Mel
     */
    private fun hzToMel(hz: Float): Float {
        return 2595f * log10(1f + hz / 700f)
    }
    
    /**
     * Mel转Hz
     */
    private fun melToHz(mel: Float): Float {
        return 700f * (exp(mel / 2595f) - 1f)
    }
    
    /**
     * 计算FFT（使用优化的DFT实现）
     * 注意：这是O(n²)实现，对于实时应用可能较慢
     * 建议：如果性能不够，可以集成KissFFT或其他FFT库
     */
    private fun computeFFT(audio: FloatArray, window: FloatArray): FloatArray {
        val n = nFft
        val fftResult = FloatArray(n / 2 + 1)
        
        // 应用窗函数
        val windowed = FloatArray(n)
        for (i in audio.indices) {
            if (i < n) {
                windowed[i] = audio[i] * window[i]
            }
        }
        
        // 优化的DFT实现（预计算角度）
        val nDiv2 = n / 2 + 1
        for (k in 0 until nDiv2) {
            var real = 0f
            var imag = 0f
            
            // 使用查表法优化sin/cos计算
            val kDivN = k.toFloat() / n
            for (n_idx in 0 until n) {
                val angle = -2f * PI.toFloat() * kDivN * n_idx
                val cosVal = cos(angle)
                val sinVal = sin(angle)
                real += windowed[n_idx] * cosVal
                imag += windowed[n_idx] * sinVal
            }
            
            val magnitude = sqrt(real * real + imag * imag)
            fftResult[k] = magnitude * magnitude / n  // 功率谱
        }
        
        return fftResult
    }
    
    /**
     * 创建Hanning窗
     */
    private fun createHanningWindow(size: Int): FloatArray {
        val window = FloatArray(size)
        for (i in 0 until size) {
            window[i] = 0.5f * (1f - cos(2f * PI.toFloat() * i / (size - 1)))
        }
        return window
    }
    
    /**
     * 将音频转换为Mel频谱图
     * @param audioData 音频数据（160000个样本，10秒@16kHz）
     * @return Mel频谱图，形状为 (n_mels, time_steps)，其中 time_steps = (160000 - n_fft) / hop_length + 1 ≈ 313
     */
    fun convert(audioData: FloatArray): FloatArray {
        val audioLength = audioData.size
        val window = createHanningWindow(nFft)
        
        // 计算时间步数
        val nTimeSteps = (audioLength - nFft) / hopLength + 1
        
        // 初始化Mel频谱图
        val melSpectrogram = FloatArray(nMels * nTimeSteps)
        
        // 对每个时间窗口计算Mel频谱
        for (t in 0 until nTimeSteps) {
            val start = t * hopLength
            val end = minOf(start + nFft, audioLength)
            
            // 提取窗口数据
            val windowData = FloatArray(nFft)
            if (end <= start) break
            
            var paddingStart = 0
            if (start + nFft > audioLength) {
                // 需要填充
                paddingStart = audioLength - start
                for (i in paddingStart until nFft) {
                    windowData[i] = 0f
                }
            }
            
            for (i in start until end) {
                windowData[i - start] = audioData[i]
            }
            
            // 计算FFT
            val powerSpectrum = computeFFT(windowData, window)
            
            // 应用Mel滤波器组
            for (m in 0 until nMels) {
                var melEnergy = 0f
                for (k in powerSpectrum.indices) {
                    melEnergy += powerSpectrum[k] * melFilterBank[m][k]
                }
                // 转换为dB（与训练时一致）
                melEnergy = maxOf(1e-10f, melEnergy)
                val db = 20f * log10(melEnergy)
                melSpectrogram[t * nMels + m] = db
            }
        }
        
        // 归一化到[-1, 1]（与训练时一致）
        val minVal = melSpectrogram.minOrNull() ?: 0f
        val maxVal = melSpectrogram.maxOrNull() ?: 1f
        val range = maxOf(1e-8f, maxVal - minVal)
        
        for (i in melSpectrogram.indices) {
            melSpectrogram[i] = 2f * (melSpectrogram[i] - minVal) / range - 1f
        }
        
        return melSpectrogram
    }
}

