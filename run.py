from app import create_app

# 通过应用工厂创建app实例
app = create_app()

if __name__ == '__main__':
    # 从app.config中获取host和port会更灵活，此处为简化
    app.run(host='0.0.0.0', port=5001, debug=True)