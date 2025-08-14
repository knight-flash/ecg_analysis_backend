from flask import Blueprint, request, jsonify
from app.utils.data_processor import process_ecg_signal_from_file
from app.state import SESSIONS
from app.config import TARGET_SAMPLING_RATE
import requests # 假设HeartVoice调用仍在此处
import uuid
import numpy as np
from app.config import HEARTVOICE_API_URL, PLAYBACK_DURATION_S


analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/analyze', methods=['POST'])
def analyze_ecg():
    """
    【修改】此接口现在还负责创建会话并返回session_id。
    """
    if 'file' not in request.files: return jsonify({"error": "未找到文件部分"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "未选择文件"}), 400

    try:
        resampled_signal, playback_waveform = process_ecg_signal_from_file(file.stream)
        
        api_payload = {'ecgData': resampled_signal.tolist(), 'ecgSampleRate': TARGET_SAMPLING_RATE, 'method': 'FeatureDB'}
        response = requests.post(url=HEARTVOICE_API_URL, headers={'Content-Type': 'application/json'}, json=api_payload)
        response.raise_for_status()
        response_data_from_api = response.json()
        
        if response_data_from_api.get('code') != 200:
            raise Exception(f"HeartVoice API返回错误: {response_data_from_api.get('msg')}")
        
        full_api_data = response_data_from_api.get('data', {})
        
        # 【新增】创建会话并存储数据
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {'full_analysis': full_api_data}
        print(f"--- New session created: {session_id} ---")
        
        # 提取用于仪表盘的指标 (这部分逻辑保持不变)
        health_index = full_api_data.get('HealthIndex', {})
        dashboard_metrics = {
            'HR': full_api_data.get('Features', {}).get('HR'),
            'Pressure': health_index.get('Pressure'),
            'HRV': health_index.get('HRV'),
            'Emotion': health_index.get('Emotion'),
            'Fatigue': health_index.get('Fatigue'),
            'Vitality': health_index.get('Vitality')
        }
        
        def normalize_signal(signal): return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        normalized_signal = normalize_signal(resampled_signal)
        target_length = TARGET_SAMPLING_RATE * PLAYBACK_DURATION_S
        playback_waveform = np.concatenate((np.tile(normalized_signal, target_length // len(normalized_signal)), normalized_signal[:target_length % len(normalized_signal)])) if len(normalized_signal) < target_length else normalized_signal[:target_length]

        response_to_frontend = {
            'session_id': session_id,  # 【新增】返回session_id给前端
            'waveform': playback_waveform.tolist(),
            'initialAnalysis': {k: (float(v) if v is not None and not np.isnan(v) else None) for k, v in dashboard_metrics.items()},
        }
        
        return jsonify(response_to_frontend)

    except Exception as e:
        print(f"处理文件时出错: {e}")
        return jsonify({"error": f"处理文件时出现未知错误: {str(e)}"}), 500