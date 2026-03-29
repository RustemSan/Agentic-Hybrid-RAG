FROM python:3.13-slim

WORKDIR /app

# Установка зависимостей
RUN pip install --no-cache-dir streamlit requests

# Копируем код
COPY app.py .

EXPOSE 8501

# Запуск streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]