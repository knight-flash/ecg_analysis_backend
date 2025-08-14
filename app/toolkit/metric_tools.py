import json
from app.state import SESSIONS
from app.services.zhipuai_client import get_glm_response
from app.toolkit.knowledge import get_knowledge_for_prompt
from app.config import ZHIPU_API_TOKEN, GLM_API_URL, GLM_MODEL_NAME
# (这里粘贴之前在 app.py 中定义的所有 tool_* 函数)
# 例如 tool_get_specific_metric, tool_reset_session 等
# 注意：原先的 tool_get_full_analysis_report 为方便演示，这里也只返回模拟内容
from app.toolkit.knowledge import get_knowledge_for_prompt, get_flat_knowledge_base
def tool_get_full_analysis_report(session_id: str) -> str:
    """
    工具函数：获取并生成完整的健康分析报告。
    它从会话中读取数据，然后调用GLM模型生成报告。
    """
    print(f"--- [Tool Executing] tool_get_full_analysis_report for session: {session_id} ---")
    
    # 从会话中获取数据
    session_data = SESSIONS.get(session_id)
    if not session_data or 'full_analysis' not in session_data:
        return "工具执行失败：未找到有效的分析数据。请先上传文件进行分析。"

    full_api_data = session_data['full_analysis']
    
    # 检查API Token
    if not ZHIPU_API_TOKEN:
        return "报告生成失败：服务器未配置ZHIPU_API_TOKEN。"

    # --- 调用LLM生成报告的核心逻辑 ---
    prompt = "你是一名专业的心脏健康数据分析师HeartTalk。请根据以下提供的心电图（ECG）详细分析数据，为用户生成一份专业、简洁且通俗易懂的健康总结报告。报告应分点阐述，并给出一个总体的健康建议。请使用Markdown格式化你的回答。\n\n--- 分析数据摘要 ---\n"
    
    for key, value in full_api_data.items():
        if isinstance(value, dict):
            prompt += f"\n**{key}**:\n"
            for sub_key, sub_value in value.items():
                prompt += f"- {sub_key}: {sub_value}\n"
    prompt += "\n请基于以上完整数据开始生成报告："

    try:

        report_data = get_glm_response(messages=[{"role": "user", "content": prompt}])
        
        content = report_data['choices'][0]['message']['content']
        cleaned_content = content.replace("<think>", "").replace("</think>", "").strip()
        print("--- [Tool Finished] Report generated successfully. ---")
        return cleaned_content

    except Exception as e:
        print(f"调用GLM生成报告时出错: {e}")
        return f"调用AI生成报告时出错: {e}"


def tool_reset_session(session_id: str) -> str:
    """【新增】工具函数：重置或清空当前会话数据。"""
    print(f"--- [Tool Executing] tool_reset_session for session: {session_id} ---")
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        print(f"--- Session {session_id} has been reset. ---")
        return "会话已成功重置。您可以上传新文件开始新的分析了。"
    return "操作失败：未找到需要重置的会话。"

def tool_get_specific_metric(session_id: str, user_query: str) -> str:
    """
    智能查询工具：接收用户的模糊问题，理解并映射到具体指标，然后查询并返回结果。
    """
    print(f"--- [Smart Tool Executing] for session: {session_id} with query: '{user_query}' ---")
    
    # --- 内部步骤1: 理解与映射 ---
    knowledge_prompt = get_knowledge_for_prompt()
    mapping_prompt = (
        "你是一个医疗数据映射专家。请根据用户的提问和下面提供的“可用指标知识库”，分析出用户最可能关心的指标是哪一个或哪几个。\n"
        "你的任务是：只返回一个包含相关指标键名（key）的JSON数组。例如 [\"HRV\", \"HR\"]。\n"
        "如果找不到任何相关指标，请返回一个空数组 []。\n\n"
        f"【用户提问】:\n{user_query}\n\n"
        f"【可用指标知识库】:\n{knowledge_prompt}\n\n"
        "请返回JSON数组:"
    )
    
    try:
        print("--- Calling LLM for metric mapping... ---")
        mapping_response = get_glm_response(messages=[{"role": "user", "content": mapping_prompt}])
        # 假设模型返回的 content 就是 JSON 字符串 "[\"HRV\", \"HR\"]"
        content_str = mapping_response['choices'][0]['message']['content']
        metric_keys_to_find = json.loads(content_str)
        
        if not isinstance(metric_keys_to_find, list) or not metric_keys_to_find:
            return "抱歉，根据您的描述，我未能确定您想查询的具体健康指标。您可以换个方式问问看？"
            
        print(f"--- LLM mapped query to keys: {metric_keys_to_find} ---")

    except Exception as e:
        print(f"LLM映射指标时出错: {e}")
        return "在理解您想查询的指标时遇到了点麻烦，请稍后再试。"

    # --- 内部步骤2: 查询并汇总数据 ---
    session_data = SESSIONS.get(session_id)
    if not session_data or 'full_analysis' not in session_data:
        return "工具执行失败：未找到分析数据。"
    
    full_data = session_data['full_analysis']
    flat_kb = get_flat_knowledge_base()
    
    results = []
    for key in metric_keys_to_find:
        key_upper = key.upper()
        # 在所有可能的子字典中查找
        value = full_data.get('ECGFeatures', {}).get(key_upper) or \
                full_data.get('PPGFeatures', {}).get(key_upper) or \
                full_data.get('HRVIndex', {}).get(key_upper) or \
                full_data.get('HealthIndex', {}).get(key_upper)

        if value is not None:
            metric_info = flat_kb.get(key, {})
            name_cn = metric_info.get("name_cn", key)
            results.append(f"- {name_cn} ({key}): **{value}**")
        else:
            results.append(f"- 未能在您的数据中找到指标 {key} 的值。")
            
    if not results:
        return "抱歉，虽然我理解了您想查询的指标，但在您的本次分析数据中并未找到它们。"

    final_response = f"根据您关于“{user_query}”的提问，我为您查询了以下最相关的指标：\n" + "\n".join(results)
    
    return final_response

# ... 如果有更多tool函数，都放在这里

# 工具箱的注册表
AVAILABLE_TOOLS = {
    "tool_get_full_analysis_report": tool_get_full_analysis_report,
    "tool_reset_session": tool_reset_session,
    "tool_get_specific_metric": tool_get_specific_metric,
}