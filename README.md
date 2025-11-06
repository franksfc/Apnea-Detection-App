# 睡眠呼吸暂停检测系统

一个基于深度学习的睡眠呼吸暂停实时检测系统，支持PC端实时检测和移动端部署。

## 📋 目录

- [项目简介](#项目简介)
- [数据集](#数据集)
- [实现原理](#实现原理)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [训练模型](#训练模型)
- [模型转换](#模型转换)
- [PC端实时检测](#pc端实时检测)
- [移动端部署](#移动端部署)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

## 项目简介

本系统使用2D CNN模型对音频信号进行睡眠呼吸暂停检测。通过分析10秒音频片段的Mel频谱图，系统能够实时判断是否存在呼吸暂停事件。

### 主要特性

- ✅ **高精度检测**: 基于深度学习的2D CNN模型，准确识别呼吸暂停事件
- ✅ **实时检测**: 支持PC端麦克风实时录音和检测
- ✅ **移动端部署**: 支持Android/iOS移动应用部署
- ✅ **滑动窗口算法**: 移动端使用滑动窗口实现连续监测
- ✅ **注意力机制**: 模型集成CBAM注意力模块，提升特征提取能力

## 数据集

### 数据来源

本项目使用的数据集基于 **PSG-Audio** 数据集，具体信息如下：

#### 原始数据集

- **来源**: Science Data Bank (科学数据银行)
- **数据集ID**: 778740145531650048
- **链接**: [https://www.scidb.cn/en/detail?dataSetId=778740145531650048](https://www.scidb.cn/en/detail?dataSetId=778740145531650048)

#### 预处理版本（本项目实际使用）

- **来源**: Kaggle
- **数据集名称**: PSG-Audio Apnea Audios
- **链接**: [https://www.kaggle.com/datasets/bryandarquea/psg-audio-apnea-audios](https://www.kaggle.com/datasets/bryandarquea/psg-audio-apnea-audios)
- **说明**: 这是原始PSG-Audio数据集的预处理版本，已转换为适合机器学习训练的格式

### 数据集结构

数据集包含多个患者的睡眠音频数据，每个患者包含两类样本：

```
psg-audio-apnea-audios/
└── PSG-AUDIO/
    └── APNEA_EDF/
        ├── patient_001/
        │   ├── patient_001_ap.npy    # 呼吸暂停样本
        │   └── patient_001_nap.npy  # 正常样本
        ├── patient_002/
        │   ├── patient_002_ap.npy
        │   └── patient_002_nap.npy
        └── ...
```

### 数据格式

- **文件格式**: NumPy数组 (`.npy`)
- **音频片段长度**: 10秒
- **采样率**: 16 kHz
- **声道**: 单声道
- **数据形状**: `(N, 160000)` - N个样本，每个160,000个采样点
- **数据类型**: float32 或 int16

### 数据获取

1. **从Kaggle下载**（推荐）:
   ```bash
   # 使用Kaggle CLI
   kaggle datasets download -d bryandarquea/psg-audio-apnea-audios
   unzip psg-audio-apnea-audios.zip -d psg-audio-apnea-audios
   ```

2. **从Science Data Bank下载原始数据**:
   - 访问 [Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=778740145531650048)
   - 按照网站指引下载原始数据
   - 需要自行进行预处理

### 数据引用

如果使用本数据集进行研究，请引用原始数据集：

```
PSG-Audio Dataset
Science Data Bank
Dataset ID: 778740145531650048
URL: https://www.scidb.cn/en/detail?dataSetId=778740145531650048
```

**注意**: 本项目使用的是Kaggle上的预处理版本，该版本已对原始数据进行了格式转换和预处理，更适合直接用于机器学习训练。

## 实现原理

### 1. 系统架构概览

本系统采用端到端的深度学习方案，从原始音频到最终检测结果，包含以下核心组件：

```
┌─────────────────────────────────────────────────────────────┐
│                     训练阶段 (PC端)                          │
├─────────────────────────────────────────────────────────────┤
│  原始音频 → 预处理 → Mel频谱图 → 2D CNN → 分类结果            │
│  (10秒)    (降噪/滤波)  (64×313)    (模型)    (正常/呼吸暂停) │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    [模型转换: TorchScript]
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   推理阶段 (移动端)                           │
├─────────────────────────────────────────────────────────────┤
│  实时音频流 → 滑动窗口 → 预处理 → Mel频谱图 → 模型推理 → 结果  │
│  (16kHz)    (10秒窗口)  (可选)    (64×313)   (TorchScript)   │
└─────────────────────────────────────────────────────────────┘
```

### 2. 模型架构详解

系统采用2D CNN架构，专门设计用于处理Mel频谱图：

```
输入: Mel频谱图 (1, 64, 313) - 10秒音频 @ 16kHz
  ↓
卷积层1: Conv2d(1→48, 3×3) + ReLU + CBAM注意力 + MaxPool(2×2)
  ↓
卷积层2: Conv2d(48→96, 3×3) + ReLU + CBAM注意力 + MaxPool(2×2)
  ↓
卷积层3: Conv2d(96→192, 3×3) + ReLU + CBAM注意力 + MaxPool(2×2)
  ↓
卷积层4: Conv2d(192→256, 3×3) + ReLU + CBAM注意力 + MaxPool(2×2)
  ↓
全局平均池化 (4×4) → 256维特征向量
  ↓
全连接层1: Linear(256→512) + ReLU + Dropout(0.5)
  ↓
全连接层2: Linear(512→256) + ReLU + Dropout(0.5)
  ↓
全连接层3: Linear(256→128) + ReLU
  ↓
输出层: Linear(128→2) + Softmax
  ↓
输出: 二分类概率 [P(正常), P(呼吸暂停)]
```

**关键特性**:
- **CBAM注意力机制**: 结合通道注意力和空间注意力，自动聚焦重要特征
  - 通道注意力: 学习哪些频率通道更重要
  - 空间注意力: 学习时间序列中哪些时刻更重要
- **深度网络**: 4层卷积 + 3层全连接，充分提取音频特征
- **Dropout正则化**: 防止过拟合，提升泛化能力

### 3. 完整数据处理流程

#### 3.1 PC端训练流程

```
原始音频数据 (10秒, 16kHz, 单声道)
    ↓
[可选] 降噪处理 (Spectral Gating算法)
    ↓
[可选] 带通滤波 (Butterworth, 100-2000 Hz)
    ↓
Z-score归一化: (x - μ) / σ
    ↓
Mel频谱图转换:
  - FFT窗口: 1024 samples
  - Hop长度: 512 samples
  - Mel滤波器: 64个 (50-4000 Hz)
  - 输出: (64, 313) 矩阵
    ↓
[训练时] 数据增强:
  - 增益变化: ±30%
  - 时间偏移: ±1秒
  - 高斯噪声: SNR 5-20 dB
  - 时间拉伸: 0.9-1.1倍
    ↓
Min-Max归一化到 [-1, 1]
    ↓
输入模型: (1, 1, 64, 313)
```

#### 3.2 移动端推理流程

```
实时音频流 (16kHz, 单声道)
    ↓
AudioCaptureNative (原生模块)
  - Android: AudioRecord API
  - iOS: AVAudioEngine
    ↓
AudioProcessor (滑动窗口)
  - 窗口大小: 10秒 (160,000 samples)
  - 步长: 5秒 (80,000 samples)
  - 每5秒提取一个窗口
    ↓
[可选] AudioPreprocessor
  - 降噪: Spectral Gating
  - 带通滤波: Butterworth (100-2000 Hz)
    ↓
ModelInference.preprocessAudio()
  - Z-score归一化
  - Mel频谱图生成 (MelSpectrogramGenerator)
    * FFT: 1024
    * Hop: 512
    * Mel滤波器: 64个
    * 输出: 64×313 = 20,032 元素
  - Min-Max归一化到 [-1, 1]
    ↓
PyTorchNative.predict() (原生模块)
  - 加载TorchScript模型
  - 推理: (1, 1, 64, 313) → (2,)
  - Temperature Scaling (可选)
    ↓
结果后处理
  - Softmax归一化
  - 置信度计算
    ↓
UI显示
  - 实时结果
  - 历史记录
  - 统计信息
```

#### 3.3 Mel频谱图生成详解

Mel频谱图是音频信号在Mel频率尺度上的时频表示，生成过程如下：

1. **分帧**: 将10秒音频 (160,000 samples) 分成重叠的帧
   - 帧长: 1024 samples (64ms @ 16kHz)
   - 帧移: 512 samples (32ms)
   - 总帧数: (160000 + 1024) / 512 = 313 帧 (考虑center=True填充)

2. **FFT**: 对每帧进行快速傅里叶变换
   - 输入: 1024个时域样本
   - 输出: 513个频域系数 (复数)

3. **功率谱**: 计算功率谱密度
   - `P = |FFT|²`

4. **Mel滤波器组**: 应用64个Mel滤波器
   - Mel频率: 将线性频率转换为Mel频率 (更符合人耳感知)
   - 频率范围: 50-4000 Hz
   - 输出: 64维Mel频谱

5. **对数转换**: 转换为分贝 (dB) 尺度
   - `Mel_db = 10 * log10(Mel + ε)`

6. **归一化**: Min-Max归一化到 [-1, 1]

### 4. 训练策略

#### 4.1 损失函数

- **类别权重**: 增强呼吸暂停类权重，减少漏诊
  - 正常类权重: 1.0
  - 呼吸暂停类权重: 2.0-3.0 (根据数据集不平衡程度调整)
- **Label Smoothing**: 0.1平滑因子，防止过拟合
- **可选Focal Loss**: 关注困难样本，提升召回率
  - `FL = -α(1-p)^γ log(p)`

#### 4.2 优化器配置

- **优化器**: Adam
- **学习率**: 1e-4 到 2e-4
- **权重衰减**: 1e-4 (L2正则化)
- **学习率调度**: 
  - ReduceLROnPlateau: 验证集指标不提升时降低学习率
  - CosineAnnealingLR: 余弦退火调度

#### 4.3 训练技巧

- **混合精度训练**: 使用AMP加速训练，降低显存占用 (2倍速度提升)
- **梯度裁剪**: 最大梯度范数1.0，稳定训练
- **早停机制**: 验证集F1分数7个epoch无提升则停止
- **最佳模型保存**: 基于验证集F1分数保存最佳模型

### 5. 移动端检测算法

#### 5.1 滑动窗口算法

移动端使用滑动窗口实现连续监测，避免重复计算：

```
时间轴: 0s ────── 5s ────── 10s ────── 15s ────── 20s
窗口1:  [───────────]  (0-10s)
窗口2:          [───────────]  (5-15s)
窗口3:                   [───────────]  (10-20s)
```

- **窗口大小**: 10秒音频 (160,000 samples @ 16kHz)
- **步长**: 5秒 (80,000 samples) - 每5秒进行一次检测
- **重叠**: 50%重叠，确保不遗漏事件
- **缓冲区管理**: 维护一个滑动缓冲区，自动丢弃旧数据

#### 5.2 实时音频捕获

- **Android**: 使用 `AudioRecord` API
  - 采样率: 16kHz
  - 声道: 单声道 (MONO)
  - 编码: PCM_16BIT
  - 缓冲区大小: 动态调整
  
- **iOS**: 使用 `AVAudioEngine`
  - 采样率: 16kHz
  - 声道: 单声道
  - 格式: AVAudioFormat

#### 5.3 结果展示与统计

- **实时显示**: 当前检测结果和置信度
- **历史记录**: 保存最近20次检测结果
- **统计信息**: 
  - Apnea比例: 最近N次检测中呼吸暂停的比例
  - 风险评估: 基于比例的简单风险评估

## 环境要求

### PC端训练/检测

- **Python**: >= 3.7
- **PyTorch**: >= 1.9.0
- **CUDA**: >= 10.2 (可选，用于GPU加速)
- **操作系统**: Windows / Linux / macOS

### 移动端开发

- **Node.js**: >= 16
- **React Native**: 0.82.1
- **Android Studio**: 最新版本 (Android开发)
- **Xcode**: 最新版本 (iOS开发，仅macOS)

## 从零开始部署指南

本指南将帮助您从零开始完整部署整个项目，包括PC端训练和移动端应用。

### 第一部分: PC端环境搭建

#### 步骤1: 安装Python环境

**Windows**:
```bash
# 1. 下载并安装Python 3.8+ (推荐3.9或3.10)
# 从 https://www.python.org/downloads/ 下载

# 2. 验证安装
python --version
pip --version

# 3. (可选) 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
```

**Linux/macOS**:
```bash
# 使用包管理器安装Python
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv

# macOS (使用Homebrew):
brew install python3

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
```

#### 步骤2: 安装PyTorch

根据您的系统选择安装命令：

**CPU版本**:
```bash
pip install torch torchvision torchaudio
```

**GPU版本 (CUDA)**:
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

验证安装:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

#### 步骤3: 安装项目依赖

```bash
# 克隆项目
git clone <repository-url>
cd apnea

# 安装依赖
pip install -r requirements.txt
```

**主要依赖说明**:
- `torch>=1.9.0`: 深度学习框架
- `torchaudio>=0.9.0`: 音频处理 (Mel频谱图转换)
- `numpy>=1.19.0`: 数值计算
- `scipy>=1.5.0`: 信号处理 (Butterworth滤波器)
- `scikit-learn>=0.24.0`: 机器学习工具 (评估指标)
- `matplotlib>=3.3.0`: 可视化 (训练曲线)
- `sounddevice>=0.4.0`: 音频捕获 (PC端实时检测)
- `noisereduce>=2.0.0`: 音频降噪 (可选，Spectral Gating)

#### 步骤4: 准备数据集

**下载数据集**:

本项目使用Kaggle上的预处理版本（推荐）:

```bash
# 使用Kaggle CLI下载
kaggle datasets download -d bryandarquea/psg-audio-apnea-audios
unzip psg-audio-apnea-audios.zip -d psg-audio-apnea-audios
```

或者从 [Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=778740145531650048) 下载原始数据并自行预处理。

**数据集放置位置**:

将数据集放置在项目根目录下的 `psg-audio-apnea-audios/PSG-AUDIO/APNEA_EDF/` 目录。

详细的数据集信息、结构和格式要求，请参考 [数据集](#数据集) 部分。

### 第二部分: 移动端环境搭建

#### 步骤1: 安装Node.js和npm

**Windows**:
1. 从 [Node.js官网](https://nodejs.org/) 下载LTS版本 (推荐v18或v20)
2. 安装时选择"Add to PATH"
3. 验证安装:
```bash
node --version  # 应该 >= 16
npm --version
```

**Linux**:
```bash
# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# 验证
node --version
npm --version
```

**macOS**:
```bash
# 使用Homebrew
brew install node

# 验证
node --version
npm --version
```

#### 步骤2: 安装React Native CLI

```bash
npm install -g react-native-cli
```

#### 步骤3: Android开发环境 (仅Android开发需要)

**Windows**:
1. 下载并安装 [Android Studio](https://developer.android.com/studio)
2. 在Android Studio中:
   - 打开 "More Actions" → "SDK Manager"
   - 安装 Android SDK Platform 33 (或更高)
   - 安装 Android SDK Build-Tools
   - 安装 Android Emulator (可选，用于模拟器测试)
3. 配置环境变量:
   ```bash
   # 添加到系统环境变量
   ANDROID_HOME=C:\Users\YourUsername\AppData\Local\Android\Sdk
   PATH=%PATH%;%ANDROID_HOME%\platform-tools
   PATH=%PATH%;%ANDROID_HOME%\tools
   PATH=%PATH%;%ANDROID_HOME%\tools\bin
   ```
4. 验证安装:
   ```bash
   adb version
   ```

**Linux/macOS**:
```bash
# 1. 安装Android Studio (同上)

# 2. 配置环境变量 (添加到 ~/.bashrc 或 ~/.zshrc)
export ANDROID_HOME=$HOME/Library/Android/sdk  # macOS
# export ANDROID_HOME=$HOME/Android/Sdk  # Linux
export PATH=$PATH:$ANDROID_HOME/platform-tools
export PATH=$PATH:$ANDROID_HOME/tools
export PATH=$PATH:$ANDROID_HOME/tools/bin

# 3. 重新加载配置
source ~/.bashrc  # 或 source ~/.zshrc

# 4. 验证
adb version
```

#### 步骤4: iOS开发环境 (仅iOS开发需要，仅macOS)

1. 从App Store安装 [Xcode](https://developer.apple.com/xcode/)
2. 安装Xcode Command Line Tools:
   ```bash
   xcode-select --install
   ```
3. 安装CocoaPods:
   ```bash
   sudo gem install cocoapods
   ```

#### 步骤5: 安装移动应用依赖

```bash
cd mobile_app_full

# 安装npm依赖
npm install

# 注意: npm install 会自动运行 postinstall 脚本
# 该脚本会使用 patch-package 应用补丁文件
```

**重要**: 安装过程中会自动应用补丁，详见下面的"补丁说明"部分。

#### 步骤6: 配置Android原生模块

确保模型文件已转换并放置在正确位置:

```bash
# 模型文件应位于:
mobile_app_full/android/app/src/main/assets/
  ├── apnea_model.pt
  └── audio_preprocessor.pt
```

如果还没有模型文件，需要先完成PC端的模型训练和转换。

#### 步骤7: iOS配置 (仅iOS开发)

```bash
cd mobile_app_full/ios
pod install
cd ..
```

### 第三部分: 补丁说明 (重要!)

本项目使用了两个补丁来修复第三方库的兼容性问题。这些补丁会在 `npm install` 时自动应用。

#### 为什么需要补丁?

React Native生态系统中的一些第三方库可能使用了过时的Gradle配置或依赖语法，导致与较新版本的Android构建工具不兼容。补丁允许我们在不修改原始库代码的情况下修复这些问题。

#### 补丁1: react-native-keep-awake

**问题**: 
- 使用了已废弃的 `jcenter()` 仓库 (2021年已关闭)
- 使用了过时的 `compile` 依赖配置 (应使用 `implementation`)
- `compileSdkVersion` 过低 (23)，不支持Java 9+编译
- 缺少Java版本配置

**修复内容**:
```diff
# 1. 替换废弃的仓库
- jcenter()
+ google()
+ mavenCentral()

# 2. 更新SDK版本
- compileSdkVersion 23
+ compileSdkVersion 36

# 3. 更新依赖配置
- compile 'com.facebook.react:react-native:+'
+ implementation 'com.facebook.react:react-native:+'

# 4. 添加Java版本配置
+ compileOptions {
+     sourceCompatibility JavaVersion.VERSION_17
+     targetCompatibility JavaVersion.VERSION_17
+ }
```

**补丁文件**: `mobile_app_full/patches/react-native-keep-awake+2.0.6.patch`

#### 补丁2: react-native-pytorch-core

**问题**:
- 使用了已废弃的 `jcenter()` 仓库

**修复内容**:
```diff
- jcenter()
+ mavenCentral()
```

**补丁文件**: `mobile_app_full/patches/react-native-pytorch-core+0.2.4.patch`

#### 补丁如何工作?

1. **patch-package**: 使用 `patch-package` 工具管理补丁
2. **自动应用**: `package.json` 中的 `postinstall` 脚本会在 `npm install` 后自动运行
   ```json
   {
     "scripts": {
       "postinstall": "patch-package"
     }
   }
   ```
3. **补丁位置**: 补丁文件位于 `patches/` 目录，命名格式为 `<package-name>+<version>.patch`
4. **版本匹配**: 补丁文件名包含版本号，确保只应用于匹配的包版本

#### 如果补丁未自动应用?

如果遇到构建错误，可以手动应用补丁:

```bash
cd mobile_app_full
npx patch-package react-native-keep-awake
npx patch-package react-native-pytorch-core
```

#### 更新依赖后需要重新应用补丁

如果更新了 `react-native-keep-awake` 或 `react-native-pytorch-core` 的版本，可能需要更新补丁:

```bash
# 1. 修改 node_modules 中的文件
# 2. 生成新补丁
npx patch-package react-native-keep-awake
npx patch-package react-native-pytorch-core
```

## 训练模型

### 训练命令

```bash
python main.py \
  --data_root psg-audio-apnea-audios/PSG-AUDIO/APNEA_EDF \
  --epochs 50 \
  --batch_size 128 \
  --val_batch_size 64 \
  --lr 2e-4 \
  --weight_decay 1e-4 \
  --max_grad_norm 1.0 \
  --early_stop_patience 7 \
  --label_smoothing 0.1 \
  --scheduler cosine \
  --use_focal_loss \
  --focal_alpha 0.25 \
  --focal_gamma 3.0 \
  --augment \
  --use_amp \
  --seed 42
```

**参数说明**:
- `--epochs 50`: 训练50轮，配合早停机制使用
- `--use_focal_loss`: 启用Focal Loss，关注困难样本，适合处理类别不平衡数据
- `--focal_gamma 3.0`: Focal Loss的gamma参数，值越大越关注困难样本
- `--scheduler cosine`: 使用余弦退火学习率调度，平滑降低学习率
- `--lr 2e-4`: 使用Focal Loss时建议使用较高学习率（2e-4），因为Focal Loss产生的损失值通常较小
- `--use_amp`: 启用混合精度训练，可提升2倍速度并降低显存占用
- `--augment`: 启用数据增强，提升模型泛化能力

### 训练输出

训练完成后，模型和结果保存在 `outputs/` 目录:

- `apnea_2dcnn_best.pth`: 最佳模型检查点 (基于验证集F1)
- `apnea_2dcnn_final.pth`: 最终模型权重
- `test_metrics.json`: 测试集评估指标
- `training_curves.png`: 训练曲线图

### 性能优化建议

1. **GPU利用率优化**:
   - 使用 `--use_amp` 启用混合精度训练
   - 增大 `batch_size` (根据GPU显存调整)
   - 默认禁用降噪和滤波 (CPU密集型操作)

2. **内存优化**:
   - 使用 `--val_batch_size` 减小验证批次大小
   - 减少 `--num_workers` (Windows推荐0-4)
   - 降低 `--prefetch_factor` (内存紧张时)

3. **训练加速**:
   - 启用 `--use_amp` (2倍速度提升)
   - 使用 `torch.compile` (PyTorch 2.0+)
   - 禁用 `--denoise` 和 `--bandpass` (默认已禁用)

## 模型转换

训练完成后，需要将模型转换为移动端可用的TorchScript格式:

```bash
python convert_model_for_mobile.py \
  --model_path outputs/apnea_2dcnn_best.pth \
  --output_dir mobile_app_full/assets
```

转换后的文件:
- `apnea_model.pt`: 主模型 (TorchScript格式)
- `audio_preprocessor.pt`: 音频预处理模块 (Mel频谱图转换)

## PC端实时检测

使用训练好的模型进行实时麦克风检测:

```bash
python predict_microphone.py \
  --model_path outputs/apnea_2dcnn_best.pth \
  --duration 10.0
```

### 参数说明

- `--model_path`: 模型文件路径
- `--duration`: 每次录音时长 (秒，默认10秒)
- `--single`: 单次检测模式 (需要手动触发)
- `--no_denoise`: 禁用降噪
- `--no_bandpass`: 禁用带通滤波
- `--list_devices`: 列出可用音频设备

### 连续监控模式

默认情况下，程序会每10秒自动录音并分析。按 `Ctrl+C` 停止监控。

## 移动端部署详细步骤

### 前置条件

在开始移动端部署之前，请确保:
1. ✅ 已完成PC端模型训练 (见"训练模型"部分)
2. ✅ 已完成模型转换 (见"模型转换"部分)
3. ✅ 已完成移动端环境搭建 (见"从零开始部署指南 - 第二部分")

### 步骤1: 转换并复制模型文件

首先，将训练好的模型转换为移动端格式:

```bash
# 从项目根目录运行
python convert_model_for_mobile.py \
  --model_path outputs/apnea_2dcnn_best.pth \
  --output_dir mobile_app_full/assets
```

然后，将模型文件复制到Android assets目录:

```bash
# Windows
copy mobile_app_full\assets\*.pt mobile_app_full\android\app\src\main\assets\

# Linux/macOS
cp mobile_app_full/assets/*.pt mobile_app_full/android/app/src/main/assets/
```

**Android模型文件位置**:
```
mobile_app_full/android/app/src/main/assets/
  ├── apnea_model.pt              # 主模型 (TorchScript格式)
  └── audio_preprocessor.pt       # 预处理模块 (可选，移动端使用JS实现)
```

**iOS模型文件**: 在Xcode中手动添加模型文件到项目资源:
1. 打开 `mobile_app_full/ios/mobile_app_full.xcworkspace`
2. 右键点击项目 → "Add Files to..."
3. 选择 `mobile_app_full/assets/` 目录下的 `.pt` 文件

### 步骤2: 安装依赖并应用补丁

```bash
cd mobile_app_full

# 安装npm依赖 (会自动应用补丁)
npm install

# 如果补丁未自动应用，手动运行:
npx patch-package react-native-keep-awake
npx patch-package react-native-pytorch-core
```

### 步骤3: 配置权限

#### Android权限配置

确保 `mobile_app_full/android/app/src/main/AndroidManifest.xml` 包含以下权限:

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

#### iOS权限配置

在 `mobile_app_full/ios/mobile_app_full/Info.plist` 中添加:

```xml
<key>NSMicrophoneUsageDescription</key>
<string>需要访问麦克风以进行睡眠呼吸暂停检测</string>
```

### 步骤4: 运行开发版本

#### Android

**方法1: 使用React Native CLI**
```bash
cd mobile_app_full

# 启动Metro bundler (新终端窗口)
npm start

# 运行Android应用 (另一个终端窗口)
npm run android
```

**方法2: 使用Android Studio**
1. 打开Android Studio
2. 选择 "Open an Existing Project"
3. 导航到 `mobile_app_full/android/` 目录
4. 等待Gradle同步完成
5. 点击 "Run" 按钮

**首次运行注意事项**:
- 确保已连接Android设备或启动模拟器
- 首次构建可能需要10-20分钟 (下载依赖)
- 如果遇到构建错误，检查补丁是否已应用

#### iOS (仅macOS)

```bash
cd mobile_app_full

# 安装CocoaPods依赖
cd ios
pod install
cd ..

# 启动Metro bundler (新终端窗口)
npm start

# 运行iOS应用 (另一个终端窗口)
npm run ios
```

### 步骤5: 调试和测试

#### 查看日志

**Android**:
```bash
# 查看所有日志
adb logcat

# 只查看React Native日志
adb logcat *:S ReactNative:V ReactNativeJS:V

# 查看应用特定日志
adb logcat | grep "ApneaDetection"
```

**iOS**:
- 在Xcode中打开控制台查看日志
- 或使用 `react-native log-ios` 命令

#### 常见问题排查

1. **模型加载失败**:
   - 检查模型文件是否在正确位置
   - 检查文件路径是否正确 (Android使用 `asset://` 协议)
   - 查看原生模块日志获取详细错误

2. **音频捕获失败**:
   - 检查麦克风权限是否已授予
   - Android: 在设置中手动授予权限
   - iOS: 首次运行时会弹出权限请求

3. **构建失败**:
   - 检查补丁是否已应用: `ls mobile_app_full/patches/`
   - 清理构建缓存: `cd android && ./gradlew clean`
   - 重新安装依赖: `rm -rf node_modules && npm install`

### 步骤6: 构建发布版本

#### Android APK

**Debug版本** (用于测试):
```bash
cd mobile_app_full/android
./gradlew assembleDebug
```

APK文件位于: `android/app/build/outputs/apk/debug/app-debug.apk`

**Release版本** (用于分发):
```bash
cd mobile_app_full/android

# 1. 生成签名密钥 (首次需要)
keytool -genkeypair -v -storetype PKCS12 -keystore my-release-key.keystore \
  -alias my-key-alias -keyalg RSA -keysize 2048 -validity 10000

# 2. 配置签名 (编辑 android/gradle.properties)
# 添加:
# MYAPP_RELEASE_STORE_FILE=my-release-key.keystore
# MYAPP_RELEASE_KEY_ALIAS=my-key-alias
# MYAPP_RELEASE_STORE_PASSWORD=*****
# MYAPP_RELEASE_KEY_PASSWORD=*****

# 3. 构建Release APK
./gradlew assembleRelease
```

APK文件位于: `android/app/build/outputs/apk/release/app-release.apk`

**AAB格式** (用于Google Play):
```bash
./gradlew bundleRelease
```

AAB文件位于: `android/app/build/outputs/bundle/release/app-release.aab`

#### iOS IPA

1. 在Xcode中打开 `mobile_app_full/ios/mobile_app_full.xcworkspace`
2. 选择 "Product" → "Scheme" → "mobile_app_full"
3. 选择 "Any iOS Device" 作为目标
4. 选择 "Product" → "Archive"
5. 等待归档完成
6. 在Organizer窗口中选择 "Distribute App"
7. 选择分发方式:
   - **App Store Connect**: 上传到App Store
   - **Ad Hoc**: 分发给测试设备
   - **Enterprise**: 企业内部分发
   - **Development**: 开发版本

## 项目结构

```
apnea/
├── main.py                          # 主训练脚本
├── convert_model_for_mobile.py      # 模型转换脚本
├── predict_microphone.py            # PC端实时检测
├── requirements.txt                 # Python依赖
├── README.md                        # 本文档
│
├── outputs/                         # 训练输出目录
│   ├── apnea_2dcnn_best.pth        # 最佳模型
│   ├── apnea_2dcnn_final.pth       # 最终模型
│   ├── test_metrics.json           # 测试指标
│   └── training_curves.png         # 训练曲线
│
├── assets/                          # 模型资源
│   ├── apnea_model.pt              # 转换后的模型
│   └── audio_preprocessor.pt       # 预处理模块
│
├── mobile_app_full/                 # React Native移动应用
│   ├── App.tsx                     # 主应用组件
│   ├── src/
│   │   ├── AudioProcessor.ts       # 音频处理和滑动窗口
│   │   ├── ModelInference.ts      # 模型推理封装
│   │   ├── AudioRecorder.ts       # 音频录制
│   │   ├── AudioCaptureNative.ts  # 原生音频捕获
│   │   └── PyTorchNative.ts       # PyTorch原生模块
│   ├── android/                    # Android原生代码
│   │   └── app/src/main/
│   │       ├── java/.../           # Kotlin原生模块
│   │       └── assets/             # 模型文件
│   └── ios/                        # iOS原生代码
│       └── mobile_app_full/
│           └── *.swift             # Swift原生模块
│
└── psg-audio-apnea-audios/         # 数据集目录
    └── PSG-AUDIO/
        └── APNEA_EDF/
            └── patient_*/          # 患者数据
```

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题或建议，请通过GitHub Issues联系。

---

**注意**: 本系统仅用于研究和教育目的，不能替代专业医疗诊断。如有健康问题，请咨询专业医生。

