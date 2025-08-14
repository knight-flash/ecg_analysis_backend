import json
import os

_knowledge_base_raw = None
_knowledge_base_flat = None

def _load_kb_if_needed():
    """内部函数，如果知识库未加载，则从JSON文件加载。"""
    global _knowledge_base_raw
    if _knowledge_base_raw is None:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(base_dir, '..', '..', 'metric_knowledge_base.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                _knowledge_base_raw = json.load(f)
        except FileNotFoundError:
            print("错误：metric_knowledge_base.json 文件未找到！")
            _knowledge_base_raw = {}

def get_flat_knowledge_base():
    """获取一个扁平化的知识库，方便按Key直接查找。"""
    global _knowledge_base_flat
    _load_kb_if_needed()
    if _knowledge_base_flat is None:
        _knowledge_base_flat = {}
        for category_data in _knowledge_base_raw.values():
            for metric_key, metric_info in category_data['metrics'].items():
                _knowledge_base_flat[metric_key] = metric_info
    return _knowledge_base_flat

def get_knowledge_for_prompt():
    """将知识库格式化为适合放入Prompt的字符串。"""
    _load_kb_if_needed()
    prompt_str = "可用指标知识库如下：\n"
    for category_key, category_data in _knowledge_base_raw.items():
        prompt_str += f"\n## {category_data['category_name']} ({category_key})\n"
        for metric_key, metric_data in category_data['metrics'].items():
            prompt_str += f"- {metric_key} ({metric_data['name_cn']}): {metric_data['description']}\n"
    return prompt_str