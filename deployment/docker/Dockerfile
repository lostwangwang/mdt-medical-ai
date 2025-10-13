# MDT医疗智能体系统 Docker配置
# 文件路径: deployment/docker/Dockerfile
# 作者: Tianyu (系统集成)
# 功能: 容器化部署配置

# 使用官方Python基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    curl \
    wget \
    unzip \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# 创建非root用户
RUN useradd --create-home --shell /bin/bash mdt_user
RUN chown -R mdt_user:mdt_user /app

# 复制项目文件
COPY --chown=mdt_user:mdt_user . .

# 创建必要的目录
RUN mkdir -p logs results data/processed data/examples \
    && chown -R mdt_user:mdt_user logs results data

# 切换到非root用户
USER mdt_user

# 设置健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "main.py", "--mode", "demo"]