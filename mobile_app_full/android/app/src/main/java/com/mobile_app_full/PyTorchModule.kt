package com.mobile_app_full

import android.content.Context
import com.facebook.react.bridge.*
import com.facebook.react.modules.core.DeviceEventManagerModule
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.IValue
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.*

/**
 * PyTorchModule - React Native原生模块
 * 用于加载和运行PyTorch模型
 */
class PyTorchModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    private var model: Module? = null
    private var preprocessor: Module? = null
    private var temperature: Float = 0.7f  // Temperature scaling参数（默认0.7，推荐范围0.7-0.9，值越小越自信）

    override fun getName(): String {
        return "PyTorchModule"
    }
    
    /**
     * 设置Temperature Scaling参数
     * @param temp Temperature值（推荐：0.7-0.9，默认0.8）
     */
    @ReactMethod
    fun setTemperature(temp: Double, promise: Promise) {
        try {
            if (temp <= 0.0 || temp > 2.0) {
                promise.reject("INVALID_TEMPERATURE", "Temperature必须在(0, 2]范围内，推荐0.7-0.9")
                return
            }
            temperature = temp.toFloat()
            println("Temperature设置为: $temperature")
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("SET_TEMPERATURE_ERROR", "设置Temperature失败: ${e.message}", e)
        }
    }

    /**
     * 从assets复制模型文件到应用目录
     */
    private fun assetFilePath(context: Context, assetName: String): String? {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        try {
            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
            return file.absolutePath
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return null
    }

    /**
     * 加载模型
     */
    @ReactMethod
    fun loadModel(modelPath: String, promise: Promise) {
        try {
            val context = reactApplicationContext
            
            var actualPath = modelPath
            if (modelPath.startsWith("asset://")) {
                val assetName = modelPath.replace("asset://", "")
                actualPath = assetFilePath(context, assetName) ?: run {
                    promise.reject("LOAD_ERROR", "无法从assets加载模型文件: $assetName")
                    return
                }
            }
            
            // 检查文件是否存在
            val modelFile = File(actualPath)
            if (!modelFile.exists()) {
                promise.reject("LOAD_ERROR", "模型文件不存在: $actualPath")
                return
            }
            
            // 加载模型
            model = Module.load(actualPath)
            promise.resolve(true)
            println("PyTorch模型加载成功: $actualPath")
        } catch (e: Exception) {
            promise.reject("LOAD_ERROR", "模型加载失败: ${e.message}", e)
            e.printStackTrace()
        }
    }

    /**
     * 加载预处理模块
     */
    @ReactMethod
    fun loadPreprocessor(preprocessorPath: String, promise: Promise) {
        try {
            val context = reactApplicationContext
            
            var actualPath = preprocessorPath
            if (preprocessorPath.startsWith("asset://")) {
                val assetName = preprocessorPath.replace("asset://", "")
                actualPath = assetFilePath(context, assetName) ?: run {
                    promise.reject("LOAD_ERROR", "无法从assets加载预处理模块: $assetName")
                    return
                }
            }
            
            val preprocessorFile = File(actualPath)
            if (!preprocessorFile.exists()) {
                promise.reject("LOAD_ERROR", "预处理模块文件不存在: $actualPath")
                return
            }
            
            preprocessor = Module.load(actualPath)
            promise.resolve(true)
            println("PyTorch预处理模块加载成功: $actualPath")
        } catch (e: Exception) {
            promise.reject("LOAD_ERROR", "预处理模块加载失败: ${e.message}", e)
            e.printStackTrace()
        }
    }

    /**
     * 测试方法 - 验证方法调用是否正常
     */
    @ReactMethod
    fun testMethod(promise: Promise) {
        try {
            println("testMethod 被调用")
            promise.resolve("测试成功")
        } catch (e: Exception) {
            promise.reject("TEST_ERROR", "测试失败: ${e.message}", e)
        }
    }

    /**
     * 执行模型推理（接受原始音频数据）
     * @param audioData 原始音频数据（一维数组，160000个样本）
     * @param promise Promise
     */
    @ReactMethod
    fun predictFromAudio(audioData: ReadableArray, promise: Promise) {
        try {
            android.util.Log.e("PyTorchModule", "predictFromAudio 方法开始执行")
            println("predictFromAudio 方法被调用")
            
            // 检查模型
            if (model == null) {
                println("错误: 模型未加载")
                promise.reject("NOT_LOADED", "模型未加载")
                return
            }

            println("模型检查通过")

            // 检查输入数据
            if (audioData.size() == 0) {
                println("错误: 音频数据为空")
                promise.reject("INVALID_INPUT", "音频数据为空")
                return
            }

            println("音频数据非空检查通过")

            // 安全地获取数组大小
            val audioSize = try {
                audioData.size()
            } catch (e: Exception) {
                println("错误: 获取音频数据大小失败: ${e.message}")
                e.printStackTrace()
                promise.reject("INVALID_INPUT", "获取音频数据大小失败: ${e.message}", e)
                return
            }

            if (audioSize <= 0) {
                println("错误: 音频数据大小为0")
                promise.reject("INVALID_INPUT", "音频数据大小为0")
                return
            }

            println("开始处理音频数据，大小: $audioSize")

            // 检查音频数据大小是否合理（10秒@16kHz应该是160000）
            if (audioSize > 200000) {
                println("警告: 音频数据过大 ($audioSize)，可能截断到160000")
            }

            // 限制音频数据大小，避免内存问题
            val maxSize = 160000
            val actualSize = if (audioSize > maxSize) maxSize else audioSize
            
            println("准备创建FloatArray，大小: $actualSize")
            
            val audioFloatArray = try {
                FloatArray(actualSize)
            } catch (e: Exception) {
                println("错误: 创建FloatArray失败: ${e.message}")
                e.printStackTrace()
                promise.reject("MEMORY_ERROR", "创建FloatArray失败: ${e.message}", e)
                return
            }

            println("FloatArray创建成功，开始转换数据...")

            // 批量转换数据，添加更多错误处理
            var conversionErrors = 0
            for (i in 0 until actualSize) {
                try {
                    val value = audioData.getDouble(i)
                    audioFloatArray[i] = value.toFloat()
                } catch (e: Exception) {
                    conversionErrors++
                    if (conversionErrors <= 5) { // 只打印前5个错误
                        println("警告: 读取音频数据索引 $i 时出错: ${e.message}")
                    }
                    audioFloatArray[i] = 0f
                }
            }

            if (conversionErrors > 0) {
                println("警告: 共有 $conversionErrors 个数据转换错误")
            }

            println("音频数据转换完成，实际大小: $actualSize，转换错误: $conversionErrors")

            // 如果预处理模块已加载，使用它
            var melTensor: Tensor? = null
            if (preprocessor != null) {
                try {
                    println("尝试使用预处理模块...")
                    // 创建音频Tensor: (1, 160000)
                    val audioTensor = Tensor.fromBlob(audioFloatArray, longArrayOf(1, actualSize.toLong()))
                    
                    println("音频Tensor创建成功，形状: ${audioTensor.shape().contentToString()}")
                    
                    // 使用预处理模块转换为Mel频谱图
                    val preprocessedOutput = preprocessor!!.forward(IValue.from(audioTensor))
                    melTensor = preprocessedOutput.toTensor()
                    
                    println("使用预处理模块生成Mel频谱图成功，形状: ${melTensor.shape().contentToString()}")
                } catch (e: Exception) {
                    println("预处理模块失败: ${e.message}")
                    e.printStackTrace()
                    // 继续使用手动预处理
                    melTensor = null
                }
            } else {
                println("预处理模块未加载，使用手动预处理")
            }
            
            // 如果预处理模块失败或未加载，使用完整的Mel频谱图转换器
            if (melTensor == null) {
                try {
                    println("使用完整的Mel频谱图转换器生成Mel频谱图")
                    
                    // 先进行归一化（Z-score）
                    val mean = audioFloatArray.average().toFloat()
                    val normalized = audioFloatArray.map { it - mean }.toFloatArray()
                    val variance = normalized.map { it * it }.average().toFloat()
                    val std = sqrt(variance).coerceAtLeast(1e-8f)
                    val normalizedAudio = normalized.map { it / std }.toFloatArray()
                    
                    // 使用完整的Mel频谱图转换器
                    val converter = MelSpectrogramConverter(
                        sampleRate = 16000,
                        nMels = 64,
                        nFft = 1024,
                        hopLength = 512,
                        fMin = 50f,
                        fMax = 4000f
                    )
                    
                    val melSpectrogram = converter.convert(normalizedAudio)
                    
                    // Mel频谱图形状: (n_mels * time_steps) = 64 * 313 = 20032
                    // 需要重塑为 (1, 1, 64, 313)
                    val expectedSize = 64 * 313
                    val melFloatArray = if (melSpectrogram.size >= expectedSize) {
                        melSpectrogram.sliceArray(0 until expectedSize)
                    } else {
                        // 如果大小不够，填充0
                        val padded = FloatArray(expectedSize)
                        melSpectrogram.copyInto(padded, 0, 0, minOf(melSpectrogram.size, expectedSize))
                        padded
                    }
                    
                    melTensor = Tensor.fromBlob(melFloatArray, longArrayOf(1, 1, 64, 313))
                    println("完整的Mel频谱图转换完成，形状: ${melTensor.shape().contentToString()}")
                } catch (e: Exception) {
                    println("Mel频谱图转换失败: ${e.message}")
                    e.printStackTrace()
                    promise.reject("PREPROCESS_ERROR", "预处理失败: ${e.message}", e)
                    return
                }
            }

            // 确保Mel频谱图形状正确
            val melShape = try {
                melTensor?.shape()
            } catch (e: Exception) {
                println("获取Mel频谱图形状失败: ${e.message}")
                promise.reject("SHAPE_ERROR", "获取Mel频谱图形状失败: ${e.message}", e)
                return
            }
            
            if (melShape == null) {
                promise.reject("INVALID_INPUT", "Mel频谱图为null")
                return
            }
            
            println("Mel频谱图形状: ${melShape.contentToString()}")
            
            if (melShape.size != 4 || melShape[0] != 1L || melShape[1] != 1L || melShape[2] != 64L || melShape[3] != 313L) {
                println("警告: Mel频谱图形状不正确: ${melShape.contentToString()}, 期望: [1, 1, 64, 313]")
                // 尝试重塑
                try {
                    val melData = melTensor.getDataAsFloatArray()
                    if (melData.size >= 64 * 313) {
                        val reshapedData = melData.sliceArray(0 until (64 * 313))
                        melTensor = Tensor.fromBlob(reshapedData, longArrayOf(1, 1, 64, 313))
                        println("Mel频谱图重塑成功")
                    } else {
                        promise.reject("INVALID_INPUT", "Mel频谱图数据大小不足: ${melData.size}, 期望至少: ${64 * 313}")
                        return
                    }
                } catch (e: Exception) {
                    println("重塑Mel频谱图失败: ${e.message}")
                    e.printStackTrace()
                    promise.reject("RESHAPE_ERROR", "重塑Mel频谱图失败: ${e.message}", e)
                    return
                }
            }
            
            // 执行模型推理
            println("开始执行模型推理...")
            val output = try {
                model!!.forward(IValue.from(melTensor))
            } catch (e: Exception) {
                println("模型推理失败: ${e.message}")
                e.printStackTrace()
                promise.reject("INFERENCE_ERROR", "模型推理失败: ${e.message}", e)
                return
            }
            
            val outputTensor = try {
                output.toTensor()
            } catch (e: Exception) {
                println("转换输出Tensor失败: ${e.message}")
                e.printStackTrace()
                promise.reject("TENSOR_ERROR", "转换输出Tensor失败: ${e.message}", e)
                return
            }
            
            // 获取输出数据
            val outputData = try {
                outputTensor.getDataAsFloatArray()
            } catch (e: Exception) {
                println("获取输出数据失败: ${e.message}")
                e.printStackTrace()
                promise.reject("DATA_ERROR", "获取输出数据失败: ${e.message}", e)
                return
            }
            
            // Temperature Scaling + Softmax归一化
            if (outputData.size < 2) {
                promise.reject("INVALID_OUTPUT", "模型输出维度不正确: ${outputData.size}")
                return
            }

            // 应用Temperature Scaling（在softmax之前）
            val scaledLogit0 = outputData[0] / temperature
            val scaledLogit1 = outputData[1] / temperature
            
            val exp0 = try {
                Math.exp(scaledLogit0.toDouble())
            } catch (e: Exception) {
                println("计算exp0失败: ${e.message}")
                Math.exp(0.0)
            }
            val exp1 = try {
                Math.exp(scaledLogit1.toDouble())
            } catch (e: Exception) {
                println("计算exp1失败: ${e.message}")
                Math.exp(0.0)
            }
            val sum = exp0 + exp1
            
            if (sum == 0.0) {
                promise.reject("INVALID_OUTPUT", "Softmax求和为0")
                return
            }
            
            val probNormal = (exp0 / sum).toFloat()
            val probApnea = (exp1 / sum).toFloat()
            // 注意：不再在这里计算prediction，让JavaScript端使用阈值0.34计算
            // 这里只返回概率，JavaScript端会用阈值重新计算预测结果
            // val prediction = if (probApnea > probNormal) 1 else 0  // 移除，使用阈值0.34
            
            println("推理结果 (temperature=$temperature): probNormal=$probNormal, probApnea=$probApnea (JavaScript端将使用阈值0.34计算预测)")

            // 返回结果（只返回概率，不返回prediction，让JavaScript端用阈值计算）
            try {
                val result = Arguments.createMap()
                // result.putInt("prediction", prediction)  // 移除，让JavaScript端计算
                result.putDouble("probNormal", probNormal.toDouble())
                result.putDouble("probApnea", probApnea.toDouble())
                
                promise.resolve(result)
            } catch (e: Exception) {
                println("创建返回结果失败: ${e.message}")
                e.printStackTrace()
                promise.reject("RESULT_ERROR", "创建返回结果失败: ${e.message}", e)
            }
        } catch (e: Exception) {
            android.util.Log.e("PyTorchModule", "predictFromAudio 异常: ${e.message}", e)
            println("predictFromAudio 顶层异常: ${e.message}")
            e.printStackTrace()
            promise.reject("PREDICT_ERROR", "推理失败: ${e.message}", e)
        } catch (e: Throwable) {
            android.util.Log.e("PyTorchModule", "predictFromAudio Throwable: ${e.message}", e)
            println("predictFromAudio Throwable: ${e.message}")
            e.printStackTrace()
            promise.reject("PREDICT_ERROR", "推理失败(Throwable): ${e.message}", e)
        }
    }

    /**
     * 执行模型推理（接受Mel频谱图数据）
     * @param melSpectrogram Mel频谱图数据（一维数组，需要重新整形为[1, 1, 64, 313]）
     */
    @ReactMethod
    fun predict(melSpectrogram: ReadableArray, promise: Promise) {
        try {
            if (model == null) {
                promise.reject("NOT_LOADED", "模型未加载")
                return
            }

            // 将ReadableArray转换为FloatArray
            val dataSize = melSpectrogram.size()
            val floatArray = FloatArray(dataSize)
            for (i in 0 until dataSize) {
                floatArray[i] = melSpectrogram.getDouble(i).toFloat()
            }

            // 创建Tensor：输入形状为 [1, 1, 64, 313]
            // melSpectrogram应该是64 * 313 = 20032个元素
            if (dataSize != 64 * 313) {
                promise.reject("INVALID_INPUT", "输入数据大小不正确。期望: ${64 * 313}, 实际: $dataSize")
                return
            }

            val inputTensor = Tensor.fromBlob(floatArray, longArrayOf(1, 1, 64, 313))
            
            // 执行推理
            val output = model!!.forward(IValue.from(inputTensor))
            val outputTensor = output.toTensor()
            
            // 获取输出数据
            val outputData = outputTensor.getDataAsFloatArray()
            
            // Temperature Scaling + Softmax归一化
            if (outputData.size < 2) {
                promise.reject("INVALID_OUTPUT", "模型输出维度不正确")
                return
            }

            // 应用Temperature Scaling（在softmax之前）
            val scaledLogit0 = outputData[0] / temperature
            val scaledLogit1 = outputData[1] / temperature
            
            val exp0 = Math.exp(scaledLogit0.toDouble())
            val exp1 = Math.exp(scaledLogit1.toDouble())
            val sum = exp0 + exp1
            
            val probNormal = (exp0 / sum).toFloat()
            val probApnea = (exp1 / sum).toFloat()
            // 注意：不再在这里计算prediction，让JavaScript端使用阈值0.34计算
            // 这里只返回概率，JavaScript端会用阈值重新计算预测结果
            // val prediction = if (probApnea > probNormal) 1 else 0  // 移除，使用阈值0.34
            
            println("推理结果 (temperature=$temperature): probNormal=$probNormal, probApnea=$probApnea (JavaScript端将使用阈值0.34计算预测)")

            // 返回结果（只返回概率，不返回prediction，让JavaScript端用阈值计算）
            val result = Arguments.createMap()
            // result.putInt("prediction", prediction)  // 移除，让JavaScript端计算
            result.putDouble("probNormal", probNormal.toDouble())
            result.putDouble("probApnea", probApnea.toDouble())
            
            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("PREDICT_ERROR", "推理失败: ${e.message}", e)
            e.printStackTrace()
        }
    }

    /**
     * 检查模型是否已加载
     */
    @ReactMethod
    fun isModelLoaded(promise: Promise) {
        promise.resolve(model != null)
    }
}
