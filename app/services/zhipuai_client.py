import requests
import pprint
from app.config import ZHIPU_API_TOKEN, GLM_API_URL, GLM_MODEL_NAME, GLM_RPM_LIMIT, GLM_TIME_WINDOW_SECONDS
from app.utils.request_controller import RequestController

# 【新增】在服务层初始化一个全局的速率控制器实例
# 所有对 get_glm_response 的调用都会共享这个控制器
glm_rate_limiter = RequestController(
    max_requests=GLM_RPM_LIMIT,
    per_seconds=GLM_TIME_WINDOW_SECONDS
)


def get_glm_response(messages: list, tools: list = None, tool_choice: str = "auto"):
    """
    向智谱AI GLM模型发起请求并获取响应（已集成速率控制）。
    """
    # 【新增】在发起任何请求前，先调用控制器等待一个“通行槽位”
    glm_rate_limiter.wait_for_slot()
    
    if not ZHIPU_API_TOKEN:
        raise ValueError("ZHIPU_API_TOKEN 未在环境中配置。")

    payload = {
        "model": GLM_MODEL_NAME,
        "messages": messages
    }
    
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    print("--- Sending Payload to GLM API ---")
    pprint.pprint(payload)
    print("------------------------------------")
    
    headers = {
        'Authorization': f'Bearer {ZHIPU_API_TOKEN}',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(GLM_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"调用GLM API时发生网络错误: {e}")
        raise