import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
# find_dotenv() 会自动寻找.env文件
from dotenv import find_dotenv
load_dotenv(find_dotenv())

# --- 从环境变量中读取所有外部配置 ---

# 心电分析API
HEARTVOICE_API_URL = os.environ.get('HEARTVOICE_API_URL', "http://183.162.233.24:10081/HeartVoice")

# 智谱AI GLM模型配置
ZHIPU_API_TOKEN = os.environ.get('ZHIPU_API_TOKEN') 
GLM_API_URL = os.environ.get('GLM_API_URL', "https://open.bigmodel.cn/api/paas/v4/chat/completions")
GLM_MODEL_NAME = os.environ.get('GLM_MODEL_NAME', "glm-4.5")

# --- 应用常量 ---
ORIGINAL_SAMPLING_RATE = 300
TARGET_SAMPLING_RATE = 100
PLAYBACK_DURATION_S = 300

GLM_RPM_LIMIT = 3  # RPM: Requests Per Minute
GLM_TIME_WINDOW_SECONDS = 60 # 时间窗口（秒）