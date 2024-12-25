# 使用官方的 Python 3.8 镜像作为基础镜像
FROM python:3.8

# 设置工作目录
WORKDIR /app

# 更新 pip 和 setuptools
RUN pip install --upgrade pip setuptools

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件到容器内
COPY . /app

# 安装 Python 依赖
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 暴露容器端口（如果需要）
EXPOSE 5000

# 设置环境变量（如果有必要）
ENV PYTHONPATH=/app

# 设置默认命令
CMD ["bash"]
