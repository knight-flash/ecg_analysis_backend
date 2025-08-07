import os
import requests
import numpy as np
import pprint
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.io import loadmat
from scipy.signal import resample
from dotenv import load_dotenv

# --- 初始化与加载环境变量 ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- 从环境变量中读取所有外部配置，更安全、更灵活 ---
HEARTVOICE_API_URL = os.environ.get('HEARTVOICE_API_URL', "http://183.162.233.24:10081/HeartVoice")
DEEPSEEK_API_URL = os.environ.get('DEEPSEEK_API_URL', "https://api.deepseek.com/chat/completions")
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')

# --- 常量定义 ---
ORIGINAL_SAMPLING_RATE = 300
TARGET_SAMPLING_RATE = 100
PLAYBACK_DURATION_S = 300

# --- AI报告生成辅助函数 ---
def generate_report_from_data(api_data):
    """根据HeartVoice返回的详细数据，调用DeepSeek API生成一份总结报告。"""
    if not DEEPSEEK_API_KEY:
        return "报告生成失败：服务器未配置DeepSeek API Key。"

    prompt = "你是一名专业的心脏健康数据分析师HeartTalk。请根据以下提供的心电图（ECG）详细分析数据，为用户生成一份专业、简洁且通俗易懂的健康总结报告。报告应分点阐述，并给出一个总体的健康建议。请使用Markdown格式化你的回答。\n\n--- 分析数据摘要 ---\n"
    
    for key, value in api_data.items():
        if isinstance(value, dict):
            prompt += f"\n**{key}**:\n"
            for sub_key, sub_value in value.items():
                prompt += f"- {sub_key}: {sub_value}\n"
    prompt += "\n请基于以上完整数据开始生成报告："

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={'Authorization': f'Bearer {DEEPSEEK_API_KEY}', 'Content-Type': 'application/json'},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        response.raise_for_status()
        report_data = response.json()
        return report_data['choices'][0]['message']['content']
    except Exception as e:
        print(f"调用DeepSeek生成报告时出错: {e}")
        return f"调用AI生成报告时出错: {e}"

# --- 主分析端点 ---
@app.route('/analyze', methods=['POST'])
def analyze_ecg():
    if 'file' not in request.files:
        return jsonify({"error": "未找到文件部分"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400

    try:
        # 1. 读取和预处理信号
        mat_data = loadmat(file.stream)
        raw_signal = mat_data['val'].flatten()
        num_samples_resampled = int(len(raw_signal) * TARGET_SAMPLING_RATE / ORIGINAL_SAMPLING_RATE)
        resampled_signal = resample(raw_signal, num_samples_resampled)
        
        # 2. 调用 HeartVoice API 获取详细分析
        api_payload = {'ecgData': resampled_signal.tolist(), 'ecgSampleRate': TARGET_SAMPLING_RATE, 'method': 'FeatureDB'}
        response = requests.post(url=HEARTVOICE_API_URL, headers={'Content-Type': 'application/json'}, json=api_payload)
        response.raise_for_status()
        response_data_from_api = response.json()
        
        if response_data_from_api.get('code') != 200:
            raise Exception(f"HeartVoice API返回错误: {response_data_from_api.get('msg')}")
        
        full_api_data = response_data_from_api.get('data', {})
        
        # 3. 调用DeepSeek API生成文本报告
        text_report = generate_report_from_data(full_api_data)

        # 4. 提取用于仪表盘的6个核心指标
        health_index = full_api_data.get('HealthIndex', {})
        dashboard_metrics = {
            'HR': full_api_data.get('Features', {}).get('HR'),
            'Pressure': health_index.get('Pressure'),
            'HRV': health_index.get('HRV'),
            'Emotion': health_index.get('Emotion'),
            'Fatigue': health_index.get('Fatigue'),
            'Vitality': health_index.get('Vitality')
        }
        
        # 5. 归一化并生成播放波形
        def normalize_signal(signal):
            return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        normalized_signal = normalize_signal(resampled_signal)
        target_length = TARGET_SAMPLING_RATE * PLAYBACK_DURATION_S
        playback_waveform = np.concatenate((np.tile(normalized_signal, target_length // len(normalized_signal)), normalized_signal[:target_length % len(normalized_signal)])) if len(normalized_signal) < target_length else normalized_signal[:target_length]

        # 6. 构建并返回给前端的最终JSON
        response_to_frontend = {
            'waveform': playback_waveform.tolist(),
            'initialAnalysis': {k: (float(v) if v is not None and not np.isnan(v) else None) for k, v in dashboard_metrics.items()},
            'textReport': text_report,
            'fullAnalysis': full_api_data # 【新增】将完整数据也发给前端，用于后续聊天
        }
        
        return jsonify(response_to_frontend)

    except Exception as e:
        print(f"处理文件时出错: {e}")
        return jsonify({"error": f"处理文件时出现未知错误: {str(e)}"}), 500

# --- 聊天代理端点 ---
@app.route('/chat', methods=['POST'])
def chat_proxy():
    data = request.get_json()
    messages = data.get('messages')
    
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "服务器未配置API Key"}), 500

    deepseek_payload = {"model": "deepseek-chat", "messages": messages}
    
    print("--- Sending to DeepSeek API (Chat) ---")
    pprint.pprint(deepseek_payload)
    print("------------------------------------")

    headers = {'Authorization': f'Bearer {DEEPSEEK_API_KEY}', 'Content-Type': 'application/json'}

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=deepseek_payload)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"调用DeepSeek API失败: {e}"}), 500

# --- 启动服务器 ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)