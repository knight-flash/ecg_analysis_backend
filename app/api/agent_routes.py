from flask import Blueprint, request, jsonify
from app.services.zhipuai_client import get_glm_response
from app.toolkit.metric_tools import AVAILABLE_TOOLS
from app.state import SESSIONS
import json
from app.config import ZHIPU_API_TOKEN, GLM_API_URL, GLM_MODEL_NAME
import requests
from app.toolkit.knowledge import get_knowledge_for_prompt
agent_bp = Blueprint('agent', __name__)


@agent_bp.route('/agent', methods=['POST'])
def agent_endpoint():
    data = request.get_json()
    session_id, user_prompt = data.get('session_id'), data.get('prompt')

    if not all([session_id, user_prompt]): return jsonify({"error": "请求中缺少 session_id 或 prompt"}), 400
    if session_id not in SESSIONS: return jsonify({"error": "无效的 session_id"}), 400
    session = SESSIONS.get(session_id)
    if session['status'] != 'ready':
        return jsonify({
            "error": "报告仍在生成中，请稍等片刻后再进行问答。",
            "status": session['status']
        }), 422 # 422 Unprocessable Entity, 表示请求格式正确但服务器无法处理
    # 【修改】将工具定义回归到简单版本
    tools_schema = [
        {"type": "function", "function": {"name": "tool_get_full_analysis_report", "description": "当用户想要一份完整、详细的文本格式健康分析报告时调用。"}},
        {"type": "function", "function": {"name": "tool_reset_session", "description": "当用户想要清除数据并重新开始时调用。"}},
        {
            "type": "function",
            "function": {
                "name": "tool_get_specific_metric",
                "description": "在理解用户问题后，用于查询单个标准化健康指标的数值。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric_name": {
                            "type": "string",
                            "description": "从下面的知识库中选择的最匹配用户问题的、标准化的指标键名，例如 'HR', 'HRV'。"
                        }
                    },
                    "required": ["metric_name"]
                }
            }
        }
    ]

    # 【修改】构建包含知识库的、更强大的System Prompt
    knowledge_str = get_knowledge_for_prompt()
    system_prompt = (
        "你是一个顶级的健康数据分析助手。你的任务是：\n"
        "1. 理解用户的自然语言提问。\n"
        "2. 参考下面提供的“可用指标知识库”，分析出用户问题涉及到哪些具体的指标。\n"
        "3. 决定是否需要以及如何调用一个或多个工具来回答问题。\n"
        "4. 如果一个模糊问题（如“心脏稳定吗”）关联到多个指标，你应该为每个相关指标都发起一次独立的工具调用。\n\n"
        f"--- 可用指标知识库 ---\n{knowledge_str}\n"
        "--- End of Knowledge Base ---"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        glm_response = get_glm_response(messages=messages, tools=tools_schema)
        choice = glm_response['choices'][0]

        if choice['finish_reason'] == 'tool_calls':
            tool_calls = choice['message']['tool_calls']
            
            # --- 【新增】优化逻辑开始 ---
            
            # 1. 收集所有需要查询的指标名称，并自动去重
            metrics_to_query = set()
            for tool_call in tool_calls:
                if tool_call['function']['name'] == 'tool_get_specific_metric':
                    arguments = json.loads(tool_call['function']['arguments'])
                    metric_name = arguments.get('metric_name')
                    if metric_name:
                        metrics_to_query.add(metric_name)

            # 2. 如果识别出需要查询的指标，则统一调用一次查询逻辑
            if metrics_to_query:
                # 调用一个辅助函数来批量获取结果 (或者直接在此处实现)
                # 这里我们直接实现
                session_id = request.get_json().get('session_id')
                results = []
                for metric_name in sorted(list(metrics_to_query)): # 排序使输出稳定
                    # 直接调用工具函数，因为它已经包含了查询单项指标的逻辑
                    result_str = AVAILABLE_TOOLS['tool_get_specific_metric'](
                        session_id=session_id, 
                        metric_name=metric_name
                    )
                    results.append(result_str)
                
                final_response = f"根据您的提问，查询到以下最相关的指标信息：\n" + "\n".join(results)
                return jsonify({"response": final_response, "type": "tool_result"})

            # (可选) 处理其他类型的工具调用，例如报告生成
            elif tool_calls[0]['function']['name'] == 'tool_get_full_analysis_report':
                 result = AVAILABLE_TOOLS['tool_get_full_analysis_report'](session_id=session_id)
                 return jsonify({"response": result, "type": "tool_result"})
            
            # --- 优化逻辑结束 ---
        
        else:
            # 【修改】情况B: 上下文感知闲聊
            print("--- No tool called. Performing context-aware chat. ---")
            
            # 1. 构建包含健康数据的系统提示
            session_data = SESSIONS.get(session_id, {})
            full_analysis = session_data.get('full_analysis', {})
            # 为了简洁，我们只提取部分核心数据作为上下文
            health_index = full_analysis.get('HealthIndex', {})
            features = full_analysis.get('Features', {})
            context_summary = {**features, **health_index}
            
            system_prompt = f"你是一名专业的健康顾问。请根据以下用户的已知健康数据摘要，来回答用户的问题。摘要：{context_summary}"
            
            # 2. 发起第二次、不带工具的LLM调用
            contextual_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            chat_payload = {"model": GLM_MODEL_NAME, "messages": contextual_messages}
            
            chat_response = requests.post(GLM_API_URL, headers={'Authorization': f'Bearer {ZHIPU_API_TOKEN}'}, json=chat_payload)
            chat_response.raise_for_status()
            chat_data = chat_response.json()
            
            direct_answer = chat_data['choices'][0]['message']['content']
            return jsonify({"response": direct_answer, "type": "text"})

    except Exception as e:
        print(f"Agent-GLM交互出错: {e}")
        return jsonify({"error": f"与AI代理交互时出错: {e}"}), 500