from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel, pipeline
from PIL import Image, ImageOps, ImageEnhance
import easyocr
import pytesseract
import numpy as np
import os, re

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
donut_processor = DonutProcessor.from_pretrained("chinmays18/medical-prescription-ocr")
donut_model = VisionEncoderDecoderModel.from_pretrained("chinmays18/medical-prescription-ocr").to(device)
reader = easyocr.Reader(['en'])
humadex_pipe = pipeline("token-classification", model="HUMADEX/english_medical_ner", aggregation_strategy="simple")
medner_pipe = pipeline("token-classification", model="blaze999/Medical-NER", aggregation_strategy="simple")
biogpt_pipe = pipeline("text-generation", model="microsoft/BioGPT-Large-PubMedQA")

def advanced_preprocess(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((960,1280))
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(2)
    npimg = np.array(img)
    npimg = np.where(npimg < 128, 0, 255).astype(np.uint8)
    bin_img = Image.fromarray(npimg)
    return bin_img.convert("RGB")

def clean_text(text):
    text = text.replace('\n', ' ').replace('\f', ' ')
    text = re.sub(r'[^A-Za-z0-9\s\-/\(\)\.,:]', '', text)
    text = re.sub(' +', ' ', text)
    return text.strip()

def extract_drugs_and_dose(text):
    drugs = re.findall(r'(SYP|TAB|CAP|SYRUP|INJECTION|DROPS|INHALER|MEFTAL[- ]?P|CALPOL|DELCON|LEVOLIN)[\w\-\/\(\)]*', text, re.I)
    doses = re.findall(r'\d+(\.\d+)?\s*(ml|mg|g|mcg|tablet|cap|puff|dose|drops)', text, re.I)
    frequency = re.findall(r'(qc[h]?|q6h|tds|t[.]?d[.]?s[.]?|qds|b[.]?d[.]?|bd|sos|daily|once|twice|x\s*\d+d)', text, re.I)
    doses = set([d[0]+d[1] if d[0] else d[1] for d in doses])
    return set(drugs), doses, set(frequency)

@app.post("/api/prescription")
async def prescription(file: UploadFile = File(...)):
    # Save and preprocess image
    filepath = f"temp_{file.filename}"
    with open(filepath, "wb") as f:
        f.write(await file.read())
    img = advanced_preprocess(filepath)

    # OCR
    pixel_values = donut_processor(images=img, return_tensors="pt").pixel_values.to(device)
    task_prompt = "<s_ocr>"
    decoder_input_ids = donut_processor.tokenizer(task_prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = donut_model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
    donut_text = donut_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    easy_text = "\n".join([t[1] for t in reader.readtext(filepath)])
    tess_text = pytesseract.image_to_string(img)
    texts = [donut_text, easy_text, tess_text]
    best_text = max(texts, key=lambda t: len(set(t.strip().split())))
    cleaned = clean_text(best_text)
    humadex_ents = humadex_pipe(cleaned)
    medner_ents = medner_pipe(cleaned)
    regex_drugs, regex_doses, regex_freqs = extract_drugs_and_dose(cleaned)
    out_drugs = set([ent.get('word','') for ent in humadex_ents if 'DRUG' in ent.get('entity_group','').upper()]) | \
        set([ent.get('word','') for ent in medner_ents if 'DRUG' in ent.get('entity_group','').upper()]) | regex_drugs
    out_doses = set([ent.get('word','') for ent in humadex_ents if 'DOSE' in ent.get('entity_group','').upper() or 'DOSAGE' in ent.get('entity_group','').upper()]) | \
        set([ent.get('word','') for ent in medner_ents if 'DOSE' in ent.get('entity_group','').upper() or 'DOSAGE' in ent.get('entity_group','').upper()]) | regex_doses
    out_freqs = set([ent.get('word','') for ent in humadex_ents if 'FREQUENCY' in ent.get('entity_group','').upper()]) | \
        set([ent.get('word','') for ent in medner_ents if 'FREQUENCY' in ent.get('entity_group','').upper()]) | regex_freqs

    # Clean up temp
    os.remove(filepath)

    return {
        "ocr_text": cleaned,
        "drugs": list(out_drugs),
        "doses": list(out_doses),
        "frequencies": list(out_freqs),
    }

@app.post("/api/chat")
async def chat(message: str = Form(...)):
    # Query BioGPT
    result = biogpt_pipe(message, max_new_tokens=200)[0]["generated_text"]
    return {"response": result}

# Optionally, add D-ID . . .
