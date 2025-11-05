FROM python:3.10

WORKDIR /app

# System dependencies for OCR and image processing
RUN apt-get update && apt-get install -y tesseract-ocr libglib2.0-0 libsm6 libxext6 libxrender-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
