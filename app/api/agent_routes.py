from flask import Blueprint, request, jsonify
from app.services.zhipuai_client import get_glm_response
from app.toolkit.metric_tools import AVAILABLE_TOOLS
from app.state import SESSIONS
import json
from app.config import ZHIPU_API_TOKEN, GLM_API_URL, GLM_MODEL_NAME
import requests

agent_bp = Blueprint('agent', __name__)

@agent_bp.route('/agent', methods=['POST'])
def agent_endpoint():
    """【修改】智能代理核心端点，功能已扩展。"""
    data = request.get_json()
    session_id, user_prompt = data.get('session_id'), data.get('prompt')

    if not all([session_id, user_prompt]): return jsonify({"error": "请求中缺少 session_id 或 prompt"}), 400
    if session_id not in SESSIONS: return jsonify({"error": "无效的 session_id"}), 400
    if not ZHIPU_API_TOKEN: return jsonify({"error": "服务器未配置ZHIPU_API_TOKEN"}), 500

    # 【修改】扩展了工具清单
    tools_schema = [
        # ... 原有的 tool_get_full_analysis_report schema ...
        {
            "type": "function",
            "function": {
                "name": "tool_get_full_analysis_report",
                "description": "当用户明确表示想要一份完整、详细的文本格式健康分析报告时，调用此工具。",
                
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tool_get_full_analysis_report",
                "description": "当用户想要一份完整、详细的文本格式健康分析报告时，调用此工具。",
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tool_reset_session",
                "description": "当用户想要清除、重置所有数据，或开始一次全新分析时调用此工具。",
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tool_get_specific_metric",
                "description": "当用户用口语化、模糊的语言询问其健康状态或具体指标时，调用此工具。例如'我心脏跳得稳不稳？'或'我最近是不是压力很大？'或'查下心率'。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "string",
                            "description": "用户的原始、完整的提问内容，例如'我心脏的节律怎么样？'。"
                        }
                    },
                    "required": ["user_query"]
                }
            }
        }
    ]

    messages = [{"role": "user", "content": user_prompt}]
    
    print(f"--- Sending to GLM Agent for session {session_id} ---")
    
    try:

        glm_response = get_glm_response(messages=messages, tools=tools_schema)
        choice = glm_response['choices'][0]
        
        if choice['finish_reason'] == 'tool_calls':
            # 情况A: 模型决定调用工具
            tool_call = choice['message']['tool_calls'][0]
            tool_name = tool_call['function']['name']
            
            if tool_name in AVAILABLE_TOOLS:
                tool_function = AVAILABLE_TOOLS[tool_name]
                # 【修改】解析工具参数
                arguments = json.loads(tool_call['function']['arguments'])
                # 使用kwargs传递参数
                tool_result = tool_function(session_id=session_id, **arguments)
                return jsonify({"response": tool_result, "type": "tool_result"})
            else:
                return jsonify({"error": f"模型意图调用一个未知的工具: {tool_name}"}), 500
        
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