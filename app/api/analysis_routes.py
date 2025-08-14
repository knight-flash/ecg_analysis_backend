import uuid
import time
import threading
import requests
from flask import Blueprint, request, jsonify
import json
# 从我们自己的模块中导入所需的内容
from app.utils.data_processor import process_ecg_signal_from_file
from app.state import SESSIONS
from app.config import HEARTVOICE_API_URL, TARGET_SAMPLING_RATE
from app.services.zhipuai_client import get_glm_response

# 创建一个名为 'analysis' 的蓝图
analysis_bp = Blueprint('analysis', __name__)


def _generate_report_and_update_status(session_id: str):
    """
    这是一个在后台线程中运行的函数。
    它负责调用LLM生成报告，并在完成后更新会话状态。
    """
    print(f"[{time.strftime('%H:%M:%S')}] 后台报告生成任务已启动，会话ID: {session_id}")
    
    # 使用 with 语句确保即使发生错误，也能安全地访问会话
    session = SESSIONS.get(session_id)
    if not session:
        print(f"[{time.strftime('%H:%M:%S')}] 错误：后台任务无法找到会话 {session_id}")
        return

    try:
        # 1. 准备用于生成报告的数据和Prompt
        full_api_data = session['full_analysis']
        # 在这里可以构建一个更详细的prompt，利用我们之前创建的知识库
        prompt = (
            "你是一名专业的心脏健康数据分析师HeartTalk。请根据以下提供的心电图（ECG）详细分析数据，"
            "为用户生成一份专业、简洁且通俗易懂的健康总结报告。"
            "报告应分点阐述，并给出一个总体的健康建议。请使用Markdown格式化你的回答。\n\n"
            f"--- 分析数据摘要 ---\n{json.dumps(full_api_data, indent=2, ensure_ascii=False)}"
        )
        
        # 2. 调用LLM服务
        report_data = get_glm_response(messages=[{"role": "user", "content": prompt}])
        report_text = report_data['choices'][0]['message']['content']
        
        # 3. 【关键】报告生成成功后，更新会话状态
        session['report'] = report_text
        session['status'] = 'ready' # 将状态更新为“已就绪”
        
        print(f"[{time.strftime('%H:%M:%S')}] 报告已生成，会话 {session_id} 状态更新为 'ready'")

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 后台报告生成失败 for session {session_id}: {e}")
        # 如果发生错误，也更新状态，方便前端处理
        session['status'] = 'error'
        session['report'] = f"AI报告生成失败，错误信息: {e}"


@analysis_bp.route('/analyze', methods=['POST'])
def analyze_ecg():
    """
    接收文件，进行分析，启动后台报告生成，并立即返回初始数据。
    """
    if 'file' not in request.files:
        return jsonify({"error": "未找到文件部分"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400

    try:
        # 步骤1: 调用数据处理模块处理文件
        resampled_signal, playback_waveform = process_ecg_signal_from_file(file.stream)
        
        # 步骤2: 调用外部HeartVoice API获取专业分析数据
        api_payload = {'ecgData': resampled_signal.tolist(), 'ecgSampleRate': TARGET_SAMPLING_RATE, 'method': 'FeatureDB'}
        response = requests.post(url=HEARTVOICE_API_URL, headers={'Content-Type': 'application/json'}, json=api_payload)
        response.raise_for_status()
        response_data_from_api = response.json()
        
        if response_data_from_api.get('code') != 200:
            raise Exception(f"HeartVoice API返回错误: {response_data_from_api.get('msg')}")
        
        full_api_data = response_data_from_api.get('data', {})
        
        # 步骤3: 创建会话并设置初始状态
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {
            'status': 'generating_report',  # 【关键点】初始状态
            'full_analysis': full_api_data,
            'report': None
        }
        
        # 步骤4: 启动后台线程来异步生成报告
        report_thread = threading.Thread(
            target=_generate_report_and_update_status,
            args=(session_id,)
        )
        report_thread.start() # 线程启动后，主程序继续执行，不会等待

        # 步骤5: 提取仪表盘所需指标并立即返回给前端
        health_index = full_api_data.get('HealthIndex', {})
        dashboard_metrics = {
            'HR': full_api_data.get('Features', {}).get('HR'),
            'Pressure': health_index.get('Pressure'),
            'HRV': health_index.get('HRV'),
            'Emotion': health_index.get('Emotion'),
            'Fatigue': health_index.get('Fatigue'),
            'Vitality': health_index.get('Vitality')
        }

        print(f"[{time.strftime('%H:%M:%S')}] 已为会话 {session_id} 返回初始响应，报告正在后台生成。")
        
        return jsonify({
            'session_id': session_id,
            'status': 'generating_report',
            'waveform': playback_waveform.tolist(),
            'initialAnalysis': {k: (float(v) if v is not None and not isinstance(v, str) else v) for k, v in dashboard_metrics.items()},
        })

    except Exception as e:
        print(f"处理文件时出错: {e}")
        return jsonify({"error": f"处理文件时出现未知错误: {str(e)}"}), 500


@analysis_bp.route('/session-status/<session_id>', methods=['GET'])
def get_session_status(session_id):
    """
    【新增】前端通过此接口轮询会话状态。
    """
    session = SESSIONS.get(session_id)
    if not session:
        return jsonify({"error": "会话不存在或已过期"}), 404
    
    print(f"[{time.strftime('%H:%M:%S')}] 前端查询会话 {session_id} 状态: {session['status']}")
    
    return jsonify({
        "session_id": session_id,
        "status": session['status'],
        "report": session.get('report') # 如果报告已生成，则一并返回
    })