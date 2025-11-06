@file:Suppress("DEPRECATION")

package com.mobile_app_full

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import com.facebook.react.bridge.*
import com.facebook.react.modules.core.DeviceEventManagerModule
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * AudioCaptureModule - 实时音频捕获模块
 * 使用AudioRecord实时捕获PCM音频数据并发送到JavaScript
 */
class AudioCaptureModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    @Volatile
    private var audioRecord: AudioRecord? = null
    @Volatile
    private var isCapturing = false
    private var captureThread: Thread? = null
    private val lock = Any() // 用于同步
    private val SAMPLE_RATE = 16000
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    private val BUFFER_SIZE_MULTIPLIER = 2

    override fun getName(): String {
        return "AudioCaptureModule"
    }

    /**
     * 强制清理所有资源（内部方法）
     */
    private fun forceCleanup() {
        synchronized(lock) {
            isCapturing = false

            // 等待线程结束
            captureThread?.let { thread ->
                if (thread.isAlive) {
                    try {
                        thread.join(2000) // 等待最多2秒
                    } catch (e: InterruptedException) {
                        println("AudioCapture: 等待线程结束时被中断")
                    }
                }
            }
            captureThread = null

            // 停止并释放AudioRecord
            audioRecord?.let { record ->
                try {
                    if (record.state == AudioRecord.STATE_INITIALIZED) {
                        record.stop()
                    }
                } catch (e: Exception) {
                    println("AudioCapture: 停止AudioRecord时出错: ${e.message}")
                }
                try {
                    record.release()
                } catch (e: Exception) {
                    println("AudioCapture: 释放AudioRecord时出错: ${e.message}")
                }
            }
            audioRecord = null
        }
    }

    /**
     * 开始实时音频捕获
     */
    @ReactMethod
    fun startCapture(promise: Promise) {
        synchronized(lock) {
            // 如果已经在捕获，先强制清理
            if (isCapturing || audioRecord != null) {
                println("AudioCapture: 检测到遗留资源，先清理...")
                forceCleanup()
                // 等待一小段时间确保资源完全释放
                try {
                    Thread.sleep(100)
                } catch (e: InterruptedException) {
                    // 忽略中断
                }
            }

            // 检查权限
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (reactApplicationContext.checkSelfPermission(Manifest.permission.RECORD_AUDIO) 
                    != PackageManager.PERMISSION_GRANTED) {
                    promise.reject("PERMISSION_DENIED", "RECORD_AUDIO权限未授予")
                    return
                }
            }

            // 计算缓冲区大小
            val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
            if (bufferSize == AudioRecord.ERROR_BAD_VALUE || bufferSize == AudioRecord.ERROR) {
                promise.reject("INIT_ERROR", "无法获取音频缓冲区大小")
                return
            }

            val actualBufferSize = bufferSize * BUFFER_SIZE_MULTIPLIER

            try {
                // 创建AudioRecord
                audioRecord = AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    SAMPLE_RATE,
                    CHANNEL_CONFIG,
                    AUDIO_FORMAT,
                    actualBufferSize
                )

                if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                    promise.reject("INIT_ERROR", "AudioRecord初始化失败")
                    forceCleanup()
                    return
                }

                // 开始录音
                try {
                    audioRecord?.startRecording()
                } catch (e: IllegalStateException) {
                    promise.reject("START_ERROR", "AudioRecord启动失败: ${e.message}", e)
                    forceCleanup()
                    return
                }

                isCapturing = true

                // 启动捕获线程
                captureThread = Thread {
                    captureAudioLoop()
                }
                captureThread?.start()

                promise.resolve(true)
                println("AudioCapture: 开始捕获音频")
            } catch (e: Exception) {
                promise.reject("START_ERROR", "启动音频捕获失败: ${e.message}", e)
                forceCleanup()
            }
        }
    }

    /**
     * 音频捕获循环
     */
    private fun captureAudioLoop() {
        val bufferSize = audioRecord?.bufferSizeInFrames ?: return
        val audioBuffer = ByteArray(bufferSize * 2) // 16-bit = 2 bytes per sample

        while (isCapturing && audioRecord != null) {
            try {
                val readSize = audioRecord?.read(audioBuffer, 0, audioBuffer.size) ?: 0

                if (readSize > 0) {
                    // 将PCM 16-bit整数转换为Float32数组
                    val floatArray = convertPCM16ToFloat32(audioBuffer, readSize)

                    // 发送到JavaScript
                    sendAudioData(floatArray)
                } else if (readSize == AudioRecord.ERROR_INVALID_OPERATION) {
                    println("AudioCapture: ERROR_INVALID_OPERATION")
                    break
                } else if (readSize == AudioRecord.ERROR_BAD_VALUE) {
                    println("AudioCapture: ERROR_BAD_VALUE")
                    break
                }
            } catch (e: Exception) {
                println("AudioCapture: 读取音频数据时出错: ${e.message}")
                break
            }
        }
    }

    /**
     * 将PCM 16-bit整数转换为Float32数组
     */
    private fun convertPCM16ToFloat32(buffer: ByteArray, size: Int): FloatArray {
        val sampleCount = size / 2
        val floatArray = FloatArray(sampleCount)
        val byteBuffer = ByteBuffer.wrap(buffer, 0, size)
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN)

        for (i in 0 until sampleCount) {
            val sample = byteBuffer.short.toInt()
            // 归一化到[-1.0, 1.0]
            floatArray[i] = sample / 32768.0f
        }

        return floatArray
    }

    /**
     * 发送音频数据到JavaScript
     */
    private fun sendAudioData(audioData: FloatArray) {
        val params = Arguments.createArray()
        for (sample in audioData) {
            params.pushDouble(sample.toDouble())
        }

        val eventParams = Arguments.createMap()
        eventParams.putArray("audioData", params)
        eventParams.putInt("sampleCount", audioData.size)

        sendEvent("onAudioData", eventParams)
    }

    /**
     * 发送事件到JavaScript
     */
    private fun sendEvent(eventName: String, params: WritableMap?) {
        reactApplicationContext
            .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit(eventName, params)
    }

    /**
     * 停止音频捕获
     */
    @ReactMethod
    fun stopCapture(promise: Promise) {
        synchronized(lock) {
            // 即使 isCapturing 是 false，也尝试清理资源（可能是之前的资源没有正确释放）
            if (!isCapturing && audioRecord == null) {
                promise.resolve(true)
                println("AudioCapture: 未在捕获音频，但资源已清理")
                return
            }

            forceCleanup()
            promise.resolve(true)
            println("AudioCapture: 停止捕获音频")
        }
    }

    @Suppress("DEPRECATION")
    override fun onCatalystInstanceDestroy() {
        super.onCatalystInstanceDestroy()
        forceCleanup()
    }
}

