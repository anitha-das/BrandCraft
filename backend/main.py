from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from dotenv import load_dotenv
import os

import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

# -------------------------
# Load .env FIRST
# -------------------------
load_dotenv()
print("GROQ KEY =", os.getenv("GROQ_API_KEY"))

# -------------------------
# Import services AFTER env load
# -------------------------
from services.groq_service import (
    generate_brand_names,
    generate_marketing_content,
    analyze_sentiment,
    chat_with_ai,
    get_color_palette,
)

from services.sdxl_service import generate_logo_prompt
from services.ai_service import generate_logo_image

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="BizForge Backend", version="1.0")

# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
frontend_path = BASE_DIR / "frontend"
static_path = frontend_path / "static"

# -------------------------
# Mount Static Folder
# -------------------------
if static_path.exists():
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    print("✅ Static folder mounted:", static_path)
else:
    print("⚠️ Static folder not found:", static_path)

# -------------------------
# API ENDPOINTS
# -------------------------

@app.post("/api/generate-brand")
async def generate_brand_endpoint(request: dict):
    try:
        industry = request.get("industry")
        keywords = request.get("keywords")
        tone = request.get("tone", "modern")
        language = request.get("language", "en")

        if not industry or not keywords:
            raise HTTPException(status_code=400, detail="industry and keywords are required")

        result = await generate_brand_names(industry, keywords, tone, language)
        return {"success": True, "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-content")
async def generate_content_endpoint(request: dict):
    try:
        brand_name = request.get("brand_name")
        brand_description = request.get("brand_description")
        tone = request.get("tone", "professional")
        content_type = request.get("content_type", "product_description")
        language = request.get("language", "en")

        if not brand_name:
            raise HTTPException(status_code=400, detail="brand_name is required")

        if not brand_description:
            raise HTTPException(status_code=400, detail="brand_description is required")

        result = await generate_marketing_content(
            brand_name,
            content_type,
            tone,
            brand_description,
            language
        )

        return {"success": True, "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-sentiment")
async def analyze_sentiment_endpoint(request: dict):
    try:
        text = request.get("text")
        brand_tone = request.get("brand_tone", "Professional")

        if not text:
            raise HTTPException(status_code=400, detail="text is required")

        result = await analyze_sentiment(text, brand_tone)
        return {"success": True, "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/get-colors")
async def get_colors_endpoint(request: dict):
    try:
        tone = request.get("tone", "modern")
        industry = request.get("industry", "general")

        result = await get_color_palette(tone, industry)
        return {"success": True, "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_endpoint(request: dict):
    try:
        message = request.get("message")

        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        result = await chat_with_ai(message)
        return {"success": True, "data": {"content": result}}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-logo")
async def generate_logo_endpoint(request: dict):
    """
    This endpoint generates ONLY the prompt.
    """
    try:
        brand_name = request.get("brand_name")
        industry = request.get("industry")
        keywords = request.get("keywords")

        if not brand_name:
            raise HTTPException(status_code=400, detail="brand_name is required")
        if not industry:
            raise HTTPException(status_code=400, detail="industry is required")
        if not keywords:
            raise HTTPException(status_code=400, detail="keywords is required")

        result = await generate_logo_prompt(brand_name, industry, keywords)
        return {"success": True, "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-logo-image")
async def generate_logo_image_endpoint(request: dict):
    """
    This endpoint generates the IMAGE using SDXL.
    """
    try:
        logo_prompt = request.get("logo_prompt")

        if not logo_prompt:
            raise HTTPException(status_code=400, detail="logo_prompt is required")

        result = await generate_logo_image(logo_prompt)
        return {"success": True, "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transcribe-voice")
async def transcribe_voice(audio_file: UploadFile = File(...)):
    """
    Transcribe audio file using Google Speech-to-Text API (FREE)
    """
    try:
        audio_bytes = await audio_file.read()

        # Convert audio to wav using pydub
        audio = AudioSegment.from_file(BytesIO(audio_bytes))
        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)

        return {"success": True, "text": text}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcription failed: {str(e)}")


# -------------------------
# FRONTEND SERVING
# -------------------------

@app.get("/")
async def serve_home():
    file_path = frontend_path / "index.html"
    if file_path.exists():
        return FileResponse(file_path)
    return JSONResponse({"error": "index.html not found"}, status_code=404)


@app.get("/{path:path}")
async def catch_all(path: str):
    file_path = frontend_path / path
    if file_path.exists():
        return FileResponse(file_path)
    return FileResponse(frontend_path / "index.html")


@app.on_event("startup")
async def startup():
    print("\n🚀 BizForge Backend Started!")
    print("🌐 API running at: http://localhost:8000")
    print(f"📁 Frontend path: {frontend_path}")
    print(f"📁 Static path: {static_path}\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

