FROM python:3.10

WORKDIR /app

# --- ADD THIS LINE for tesseract OCR ---
RUN apt-get update && apt-get install -y tesseract-ocr
# ---------------------------------------

COPY requirements.txt .
RUN pip config set global.index-url https://pypi.org/simple
RUN pip install --default-timeout=2000 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
