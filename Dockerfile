# 说明：本机未安装 Docker，此文件未经实际构建验证（语法标准，可直接用于有 Docker 的环境）
FROM python:3.10-slim

WORKDIR /app

# 先装依赖再拷代码：代码改动时不会让依赖层缓存失效，构建更快
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 服务需要访问宿主机上的 Ollama，容器内请用 host.docker.internal
# 运行时：docker run -e OLLAMA_URL=http://host.docker.internal:11434 -p 8000:8000 rag-kb
ENV OLLAMA_URL=http://host.docker.internal:11434

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
