import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.io import loadmat
from scipy.signal import resample
from ecgdetectors import Detectors

# --- 初始化与常量定义 ---
app = Flask(__name__)
# 允许来自任何源的跨域请求，方便前端调试
CORS(app)

ORIGINAL_SAMPLING_RATE = 300  # 从您的Notebook确认
TARGET_SAMPLING_RATE = 100    # 您指定的前端播放频率
PLAYBACK_DURATION_S = 300     # 我们为前端生成一个5分钟(300秒)的循环播放数据

@app.route('/analyze', methods=['POST'])
def analyze_ecg():
    """
    接收 .mat 文件, 进行预处理(重采样、归一化)、初次分析, 并返回结果。
    """
    # 检查文件是否存在于请求中
    if 'file' not in request.files:
        return jsonify({"error": "未找到文件部分"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400

    try:
        # --- 1. 读取 .mat 文件 ---
        mat_data = loadmat(file.stream)
        raw_signal = mat_data['val'].flatten()

        # --- 2. 信号预处理：升降采样 ---
        num_samples_resampled = int(len(raw_signal) * TARGET_SAMPLING_RATE / ORIGINAL_SAMPLING_RATE)
        resampled_signal = resample(raw_signal, num_samples_resampled)
        
        # --- 2.1. 【新增】归一化处理 ---
        # 根据您提供的公式 (Z-score 标准化)
        def normalize_signal(signal):
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            # 添加 1e-8 防止标准差为0时除零错误
            return (signal - mean_val) / (std_val + 1e-8)

        normalized_signal = normalize_signal(resampled_signal)
        print("信号已完成归一化处理。")

        # --- 3. 初次全局分析 (在归一化后的信号上进行) ---
        detectors = Detectors(TARGET_SAMPLING_RATE)
        r_peaks = detectors.pan_tompkins_detector(normalized_signal)

        analysis_results = {}
        if len(r_peaks) > 1:
            rr_intervals = np.diff(r_peaks) / TARGET_SAMPLING_RATE
            mean_heart_rate = 60 / np.mean(rr_intervals)
            hrv_rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            analysis_results = {
                'Heart_Rate_Mean': mean_heart_rate,
                'HRV_RMSSD': hrv_rmssd,
                'Num_R_Peaks': len(r_peaks)
            }
        
        # --- 4. 为前端生成可循环播放的波形数据 (使用归一化后的信号) ---
        target_length = TARGET_SAMPLING_RATE * PLAYBACK_DURATION_S
        
        if len(normalized_signal) >= target_length:
            playback_waveform = normalized_signal[:target_length]
        else:
            repeat_times = target_length // len(normalized_signal)
            remainder = target_length % len(normalized_signal)
            playback_waveform = np.concatenate(
                (np.tile(normalized_signal, repeat_times), normalized_signal[:remainder])
            )
        
        print(f"全局分析完成: {analysis_results}")
        print(f"已生成 {PLAYBACK_DURATION_S} 秒，共 {len(playback_waveform)} 点的播放波形。")

        # --- 5. 构建并返回最终的JSON响应 ---
        response_data = {
            'waveform': playback_waveform.tolist(),
            'initialAnalysis': {k: (float(v) if v is not None and not np.isnan(v) else None) for k, v in analysis_results.items()}
        }
        
        return jsonify(response_data)

    except KeyError:
        return jsonify({"error": "处理失败：.mat文件中未找到名为 'val' 的变量。请检查文件内容。"}), 400
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return jsonify({"error": f"处理文件时出现未知错误: {str(e)}"}), 500

# 启动服务器
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)