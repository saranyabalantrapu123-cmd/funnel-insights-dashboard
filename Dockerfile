FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Bake the synthetic dataset into the image so the dashboard starts instantly.
RUN python scripts/generate_events.py --seed 42

EXPOSE 8501
# No key -> offline demo mode. Pass -e ANTHROPIC_API_KEY=... to enable the LLM layer.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
