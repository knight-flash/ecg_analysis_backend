import requests
import pprint
from app.config import ZHIPU_API_TOKEN, GLM_API_URL, GLM_MODEL_NAME

def get_glm_response(messages: list, tools: list = None, tool_choice: str = "auto"):
    """
    向智谱AI GLM模型发起请求并获取响应。

    Args:
        messages (list): 对话消息列表。
        tools (list, optional): 可用工具的schema列表。Defaults to None.
        tool_choice (str, optional): 工具选择模式。Defaults to "auto".

    Returns:
        dict: 从API返回的JSON数据。
    """
    if not ZHIPU_API_TOKEN:
        # 在实际应用中，这里应该抛出更具体的异常
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
        response.raise_for_status()  # 如果请求失败 (非2xx状态码)，则会抛出HTTPError
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"调用GLM API时发生网络错误: {e}")
        # 在实际应用中，可以根据需要返回None或抛出自定义异常
        raise