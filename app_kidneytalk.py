import os
import requests
import numpy as np
import pprint
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.io import loadmat
from scipy.signal import resample
from dotenv import load_dotenv
import json
# --- 初始化与加载环境变量 ---
load_dotenv()
app = Flask(__name__)
CORS(app)
import re 
# --- 从环境变量中读取所有外部配置 ---
HEARTVOICE_API_URL = os.environ.get('HEARTVOICE_API_URL', "http://183.162.233.24:10081/HeartVoice")
MODEL_API_URL = os.environ.get('KIDNEYTALK_API_URL', "https://kidneytalk.bjmu.edu.cn/api/v1/chat/completions")
MODEL_NAME = "L-72B"

# --- 常量定义 ---
ORIGINAL_SAMPLING_RATE = 300
TARGET_SAMPLING_RATE = 100
PLAYBACK_DURATION_S = 300

# --- 端点 1: 快速分析接口 ---
@app.route('/analyze', methods=['POST'])
def analyze_ecg():
    """
    只负责快速的数值分析，并立即返回结果用于前端仪表盘展示。
    """
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
        
        # 3. 提取用于仪表盘的6个核心指标
        health_index = full_api_data.get('HealthIndex', {})
        dashboard_metrics = {
            'HR': full_api_data.get('Features', {}).get('HR'),
            'Pressure': health_index.get('Pressure'),
            'HRV': health_index.get('HRV'),
            'Emotion': health_index.get('Emotion'),
            'Fatigue': health_index.get('Fatigue'),
            'Vitality': health_index.get('Vitality')
        }
        
        # 4. 归一化并生成播放波形
        def normalize_signal(signal):
            return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        normalized_signal = normalize_signal(resampled_signal)
        target_length = TARGET_SAMPLING_RATE * PLAYBACK_DURATION_S
        playback_waveform = np.concatenate((np.tile(normalized_signal, target_length // len(normalized_signal)), normalized_signal[:target_length % len(normalized_signal)])) if len(normalized_signal) < target_length else normalized_signal[:target_length]
        
        # 5. 立即返回快速结果，不包含耗时的文本报告
        response_to_frontend = {
            'waveform': playback_waveform.tolist(),
            'initialAnalysis': {k: (float(v) if v is not None and not np.isnan(v) else None) for k, v in dashboard_metrics.items()},
            'fullAnalysis': full_api_data
        }
        return jsonify(response_to_frontend)

    except Exception as e:
        print(f"处理文件时出错: {e}")
        return jsonify({"error": f"处理文件时出现未知错误: {str(e)}"}), 500


def call_llm_api(messages):
    """一个通用的函数，用于调用大模型API，并伪装User-Agent。"""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.5
    }
    
    headers = {
        'Content-Type': 'application/json',
        # 【核心修改】添加一个伪装的User-Agent，模仿常用工具
        'User-Agent': 'PostmanRuntime/7.32.3' 
    }

    print(f"--- Sending to {MODEL_NAME} API with User-Agent ---")
    pprint.pprint(payload)
    print("-------------------------------------------------")

    response = requests.post(MODEL_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def clean_ai_response(text):
    """
    使用正则表达式移除AI回复中包含的 <think>...</think> 标签及其内容。
    """
    # re.DOTALL 标志让 '.' 可以匹配包括换行符在内的任意字符
    cleaned_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    # 移除可能存在的前后多余空格或换行
    return cleaned_text.strip()

# --- 端点 2: 异步AI报告生成接口 ---
@app.route('/generate-report', methods=['POST'])
def generate_report_endpoint():
    try:
        full_api_data = request.get_json().get('fullAnalysis')
        if not full_api_data:
            return jsonify({"error": "缺少分析数据"}), 400

        # 将Prompt格式化为单行长文本
        data_points = []
        for key, value in full_api_data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    data_points.append(f"{sub_key}: {sub_value}")
        data_summary_string = "; ".join(data_points)
        
        prompt = (
            "你是一名专业的心脏健康数据分析师HeartTalk。"
            "请根据以下提供的心电图（ECG）详细分析数据，为用户生成一份专业、简洁且通俗易懂的健康总结报告。"
            "报告应分点阐述，并给出一个总体的健康建议。请使用Markdown格式化你的回答。"
            f"分析数据摘要如下：{data_summary_string} "
            "请基于以上完整数据开始生成报告："
        )
        
        messages = [{"role": "user", "content": prompt}]
        api_response = call_llm_api(messages)
        report_text = api_response['choices'][0]['message']['content']
        cleaned_report_text = clean_ai_response(report_text)
        return jsonify({"textReport": cleaned_report_text})
    except Exception as e:
        return jsonify({"error": f"调用AI生成报告时出错: {str(e)}"}), 500

# --- 端点 3: 交互式聊天代理接口 ---
@app.route('/chat', methods=['POST'])
def chat_proxy():
    try:
        data = request.get_json()
        messages = data.get('messages')
        
        # 确保发送给API的content都是单行字符串
        for message in messages:
            if 'content' in message and isinstance(message['content'], str):
                message['content'] = message['content'].replace('\n', ' ')
        
        api_response = call_llm_api(messages)
        return jsonify(api_response)
    except Exception as e:
        return jsonify({"error": f"调用大模型API失败: {str(e)}"}), 500

# --- 启动服务器 ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)