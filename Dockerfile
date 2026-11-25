FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY data ./data
COPY src ./src
COPY templates ./templates
COPY run.py ./run.py

RUN PYTHONPATH=src python -m smart_fertilizer.ml_pipeline

EXPOSE 5055
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "5055"]
