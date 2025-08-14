import numpy as np
from scipy.io import loadmat
from scipy.signal import resample
from app.config import (
    ORIGINAL_SAMPLING_RATE,
    TARGET_SAMPLING_RATE,
    PLAYBACK_DURATION_S
)

def _normalize_signal(signal):
    """一个内部辅助函数，用于信号归一化。"""
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

def process_ecg_signal_from_file(file_stream):
    """
    从文件流中读取、处理和准备ECG信号。
    
    Args:
        file_stream: 从Flask请求中获取的文件流对象。

    Returns:
        一个元组，包含:
        - resampled_signal (np.array): 重采样后的信号，用于API调用。
        - playback_waveform (np.array): 用于前端播放的波形。
    """
    # 1. 读取 .mat 文件
    mat_data = loadmat(file_stream)
    raw_signal = mat_data['val'].flatten()
    
    # 2. 重采样
    num_samples_resampled = int(len(raw_signal) * TARGET_SAMPLING_RATE / ORIGINAL_SAMPLING_RATE)
    resampled_signal = resample(raw_signal, num_samples_resampled)

    # 3. 归一化并生成播放波形
    normalized_signal = _normalize_signal(resampled_signal)
    target_length = TARGET_SAMPLING_RATE * PLAYBACK_DURATION_S
    
    # 平铺或截断信号以匹配播放时长
    if len(normalized_signal) < target_length:
        # 如果信号太短，就重复它直到达到目标长度
        num_tiles = target_length // len(normalized_signal)
        remainder = target_length % len(normalized_signal)
        playback_waveform = np.concatenate(
            (np.tile(normalized_signal, num_tiles), normalized_signal[:remainder])
        )
    else:
        # 如果信号太长，就截断它
        playback_waveform = normalized_signal[:target_length]

    return resampled_signal, playback_waveform