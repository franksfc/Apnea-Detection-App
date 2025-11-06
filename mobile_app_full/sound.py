import numpy as np
import soundfile as sf

sr = 16000  # 采样率
duration = 10  # 秒
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# 模拟打鼾：低频(70Hz~150Hz) + 噪声包络
def snore_segment(start_freq=80, end_freq=140, length=1.5):
    tt = np.linspace(0, length, int(sr * length))
    freq = np.linspace(start_freq, end_freq, tt.size)
    tone = np.sin(2 * np.pi * freq * tt)
    env = np.exp(-3 * tt)  # 打鼾衰减包络
    noise = 0.2 * np.random.randn(tt.size)
    return 0.3 * env * tone + noise * 0.05

# 拼接打鼾 + 停顿
audio = np.array([], dtype=np.float32)
for i in range(4):
    audio = np.concatenate((audio, snore_segment(length=1.2)))  # 呼噜
    silence = np.zeros(int(sr * 0.8))  # “断气”停顿
    audio = np.concatenate((audio, silence))

# 截取到10秒
audio = audio[: sr * duration]

# 归一化
audio /= np.max(np.abs(audio))

sf.write("synthetic_apnea_snore.wav", audio, sr)
print("✅ 已生成 synthetic_apnea_snore.wav (10s)")
