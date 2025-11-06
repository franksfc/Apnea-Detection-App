@file:Suppress("DEPRECATION")

package com.mobile_app_full

import android.media.MediaRecorder
import android.os.Build
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.Promise
import com.facebook.react.modules.core.DeviceEventManagerModule
import java.io.IOException

class AudioRecorderModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    @Volatile
    private var mediaRecorder: MediaRecorder? = null
    private var outputPath: String? = null
    @Volatile
    private var isRecording: Boolean = false
    private val lock = Any() // 用于同步

    override fun getName(): String = "AudioRecorderModule"

    /**
     * 强制清理所有资源（内部方法）
     */
    private fun forceCleanup() {
        synchronized(lock) {
            mediaRecorder?.let { recorder ->
                try {
                    // 根据 isRecording 标志判断是否需要停止
                    // 注意：MediaRecorder 的状态检查在不同 Android 版本中 API 不同
                    // 直接使用我们自己的状态标志更可靠
                    if (this.isRecording) {
                        try {
                            recorder.stop()
                        } catch (e: IllegalStateException) {
                            // MediaRecorder 可能已经停止，忽略这个错误
                            println("AudioRecorder: MediaRecorder 可能已经停止: ${e.message}")
                        } catch (e: Exception) {
                            println("AudioRecorder: 停止MediaRecorder时出错: ${e.message}")
                        }
                    }
                } catch (e: Exception) {
                    println("AudioRecorder: 处理MediaRecorder时出错: ${e.message}")
                }
                try {
                    recorder.release()
                } catch (e: Exception) {
                    println("AudioRecorder: 释放MediaRecorder时出错: ${e.message}")
                }
            }
            mediaRecorder = null
            isRecording = false
            outputPath = null
        }
    }

    @ReactMethod
    fun startRecording(path: String, promise: Promise) {
        synchronized(lock) {
            // 如果已经在录音，先强制清理
            if (isRecording || mediaRecorder != null) {
                println("AudioRecorder: 检测到遗留资源，先清理...")
                forceCleanup()
                // 等待一小段时间确保资源完全释放
                try {
                    Thread.sleep(100)
                } catch (e: InterruptedException) {
                    // 忽略中断
                }
            }

            try {
                mediaRecorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    MediaRecorder(reactApplicationContext)
                } else {
                    @Suppress("DEPRECATION")
                    MediaRecorder()
                }

                outputPath = path
                mediaRecorder?.apply {
                    setAudioSource(MediaRecorder.AudioSource.MIC)
                    setOutputFormat(MediaRecorder.OutputFormat.AAC_ADTS)
                    setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                    setOutputFile(path)
                    setAudioSamplingRate(16000)
                    setAudioChannels(1)
                    prepare()
                    start()
                }
                
                isRecording = true
                promise.resolve(path)
                println("AudioRecorder: 开始录音")
            } catch (e: IOException) {
                promise.reject("RECORDING_ERROR", "录音启动失败: ${e.message}", e)
                forceCleanup()
            } catch (e: Exception) {
                promise.reject("RECORDING_ERROR", "录音启动失败: ${e.message}", e)
                forceCleanup()
            }
        }
    }

    @ReactMethod
    fun stopRecording(promise: Promise) {
        synchronized(lock) {
            // 即使 isRecording 是 false，也尝试清理资源（可能是之前的资源没有正确释放）
            if (!isRecording && mediaRecorder == null) {
                promise.resolve(null)
                println("AudioRecorder: 未在录音，但资源已清理")
                return
            }

            val path = outputPath
            forceCleanup()
            promise.resolve(path)
            println("AudioRecorder: 停止录音")
        }
    }

    @ReactMethod
    fun isRecording(promise: Promise) {
        promise.resolve(isRecording)
    }

    @Suppress("DEPRECATION")
    override fun onCatalystInstanceDestroy() {
        super.onCatalystInstanceDestroy()
        forceCleanup()
    }
}

