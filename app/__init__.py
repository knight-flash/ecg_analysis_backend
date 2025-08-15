from flask import Flask
from flask_cors import CORS # 1. 导入CORS
# 注册蓝图
from .api.analysis_routes import analysis_bp
from .api.agent_routes import agent_bp
def create_app():
    """创建并配置Flask应用实例。"""
    app = Flask(__name__)
    
    # 从config模块加载配置
    app.config.from_object('app.config')
    
    
    CORS(app) 
    app.register_blueprint(analysis_bp)
    app.register_blueprint(agent_bp)
    
    return app