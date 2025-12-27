FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/

# Default port for the Prediction API
EXPOSE 5000

# Entrypoint to run any script
ENTRYPOINT ["python"]

# Default command (can be overridden to run src/api/app.py)
CMD ["src/api/app.py"]