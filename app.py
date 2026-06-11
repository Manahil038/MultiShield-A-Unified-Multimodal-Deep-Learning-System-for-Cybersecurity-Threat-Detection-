import os
import sys
import time
import logging
import asyncio
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import io

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any

# Ensure project root is in path
sys.path.insert(0, os.path.abspath('.'))

from src.models.tabular_deep import DeepTabularMLP
from src.models.text_lstm import TextLSTM
from src.preprocessing.tabular_preprocessor import TabularPreprocessor
from src.preprocessing.text_preprocessor import TextPreprocessor, SimpleTokenizer

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MultiShieldBackend")

# Load configuration from environment variables with sensible defaults
CHECKPOINT_DIR = os.getenv("MULTISHIELD_CHECKPOINT_DIR", "experiments/checkpoints")
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")

app = FastAPI(
    title="MultiShield Threat Detection API",
    description="Unified Multimodal Deep Learning System for Cybersecurity Threat Detection",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and preprocessors (loaded once at startup)
models_db = {}

# Lifespan Context Manager / Startup Event
@app.on_event("startup")
async def startup_event():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing models and preprocessors on device: {device}")
    models_db["device"] = device
    
    try:
        # 1. TABULAR MODEL & PREPROCESSOR
        tab_prep_path = os.path.join(CHECKPOINT_DIR, "tabular_preprocessor.pkl")
        tab_ckpt_path = os.path.join(CHECKPOINT_DIR, "refined_tab_s42.pt")
        
        logger.info(f"Loading tabular preprocessor from {tab_prep_path}")
        with open(tab_prep_path, "rb") as f:
            models_db["tab_preprocessor"] = pickle.load(f)
            
        preprocessor = models_db["tab_preprocessor"]
        input_dim = len(preprocessor.numeric_cols) + sum(len(c) for c in preprocessor.encoder.categories_)
        logger.info(f"Tabular model input dimension dynamically resolved to: {input_dim}")
        
        tab_model = DeepTabularMLP(
            input_dim=input_dim,
            hidden_dim=256,
            num_blocks=3,
            num_classes=1,
            dropout_p=0.14
        )
        ckpt = torch.load(tab_ckpt_path, map_location=device)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        tab_model.load_state_dict(state_dict)
        tab_model.to(device)
        tab_model.eval()
        models_db["tab_model"] = tab_model
        
        # 2. FAKE NEWS MODEL & PREPROCESSORS
        news_prep_path = os.path.join(CHECKPOINT_DIR, "news_preprocessor.pkl")
        news_tok_path = os.path.join(CHECKPOINT_DIR, "news_tokenizer.pkl")
        news_ckpt_path = os.path.join(CHECKPOINT_DIR, "news_lstm.pt")
        
        logger.info(f"Loading news tokenizers/preprocessors")
        with open(news_prep_path, "rb") as f:
            models_db["news_preprocessor"] = pickle.load(f)
        with open(news_tok_path, "rb") as f:
            models_db["news_tokenizer"] = pickle.load(f)
            
        logger.info(f"Loading Fake News model from {news_ckpt_path}")
        news_model = TextLSTM(
            vocab_size=len(models_db["news_tokenizer"].word2idx),
            embedding_dim=128,
            hidden_dim=128,
            num_layers=2,
            num_classes=1,
            dropout_p=0.3
        )
        ckpt = torch.load(news_ckpt_path, map_location=device)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        news_model.load_state_dict(state_dict)
        news_model.to(device)
        news_model.eval()
        models_db["news_model"] = news_model

        # 3. PHISHING MODEL & PREPROCESSORS
        ph_prep_path = os.path.join(CHECKPOINT_DIR, "phishing_preprocessor.pkl")
        ph_tok_path = os.path.join(CHECKPOINT_DIR, "phishing_tokenizer.pkl")
        ph_ckpt_path = os.path.join(CHECKPOINT_DIR, "phishing_lstm.pt")
        
        logger.info(f"Loading phishing tokenizers/preprocessors")
        with open(ph_prep_path, "rb") as f:
            models_db["ph_preprocessor"] = pickle.load(f)
        with open(ph_tok_path, "rb") as f:
            models_db["ph_tokenizer"] = pickle.load(f)
            
        logger.info(f"Loading Phishing model from {ph_ckpt_path}")
        ph_model = TextLSTM(
            vocab_size=len(models_db["ph_tokenizer"].word2idx),
            embedding_dim=128,
            hidden_dim=128,
            num_layers=2,
            num_classes=1,
            dropout_p=0.3
        )
        ckpt = torch.load(ph_ckpt_path, map_location=device)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        ph_model.load_state_dict(state_dict)
        ph_model.to(device)
        ph_model.eval()
        models_db["ph_model"] = ph_model

        # 4. DEEPFAKE RESNET-18 MODEL
        resnet_ckpt_path = os.path.join(CHECKPOINT_DIR, "resnet_fft.pt")
        
        logger.info(f"Loading ResNet-18 model from {resnet_ckpt_path}")
        import torchvision.models as torchvision_models
        resnet_model = torchvision_models.resnet18()
        for param in resnet_model.parameters():
            param.requires_grad = False
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 1)
        ckpt = torch.load(resnet_ckpt_path, map_location=device)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        resnet_model.load_state_dict(state_dict)
        resnet_model.to(device)
        resnet_model.eval()
        models_db["resnet_model"] = resnet_model
        
        logger.info("All models and preprocessors successfully loaded.")
    except Exception as e:
        logger.critical(f"Failed to load models during startup: {str(e)}", exc_info=True)
        # We don't crash the server start immediately to allow diagnostics via /health endpoint
        models_db["error"] = str(e)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # We try to inspect request size / content length
    content_length = request.headers.get("content-length")
    req_size = f"{content_length} bytes" if content_length else "unknown size"
    
    # Process request
    response = await call_next(request)
    
    latency = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url.path} | "
        f"Size: {req_size} | "
        f"Status: {response.status_code} | "
        f"Latency: {latency:.4f}s"
    )
    return response

# No-cache middleware for static files (to prevent client browser caching during dev)
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.endswith((".html", ".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".ico")) or path == "/":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response




# Timeout handler wrapper
async def run_with_timeout(coro, timeout_sec: float = 30.0):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_sec)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The prediction took more than 30 seconds and timed out."
        )


# API Models Input Schema
class TabularInput(BaseModel):
    features: Dict[str, Any] = Field(..., description="UNSW-NB15 flow features dictionary")

class TextNewsInput(BaseModel):
    title: str = Field("", description="Fake News Title")
    text: str = Field(..., description="Fake News Body Text")

class TextPhishingInput(BaseModel):
    text: str = Field(..., description="Phishing Email Body Text")


# Health Check Endpoint
@app.get("/health")
async def health_check():
    if "error" in models_db:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": models_db["error"]}
        )
    
    active_models = [k for k in models_db.keys() if "model" in k]
    return {
        "status": "healthy",
        "device": models_db.get("device", "unknown"),
        "loaded_models": active_models,
        "timestamp": time.time()
    }


# Tabular Threat Detection Endpoint
@app.post("/api/predict/tabular")
async def predict_tabular(payload: TabularInput):
    if "tab_model" not in models_db:
        raise HTTPException(status_code=503, detail="Tabular anomaly detection model is not loaded.")
        
    async def inference():
        try:
            # Input validation: Check for empty payload
            if not payload.features:
                raise ValueError("Flow features dictionary cannot be empty.")
                
            preprocessor = models_db["tab_preprocessor"]
            model = models_db["tab_model"]
            device = models_db["device"]
            
            # Align features dictionary with columns expected by fit_transform
            input_data = {}
            for col in preprocessor.numeric_cols:
                val = payload.features.get(col, 0.0)
                try:
                    input_data[col] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    input_data[col] = 0.0
                    
            for col in preprocessor.categorical_cols:
                val = payload.features.get(col, "Unknown")
                input_data[col] = str(val) if val is not None else "Unknown"
                
            df = pd.DataFrame([input_data])
            
            # Preprocess
            features_scaled = preprocessor.transform(df)
            x = torch.FloatTensor(features_scaled).to(device)
            
            # Model Forward Pass
            with torch.no_grad():
                logits = model(x)
                prob = torch.sigmoid(logits).squeeze(-1).item()
                
            prediction = int(prob > 0.5)
            class_label = "Anomaly" if prediction == 1 else "Normal"
            confidence = prob if prediction == 1 else (1.0 - prob)
            
            return {
                "prediction": prediction,
                "label": class_label,
                "confidence": confidence,
                "raw_probability": prob
            }
        except Exception as e:
            logger.error(f"Tabular prediction error: {str(e)}", exc_info=True)
            raise ValueError(f"Error preprocessing or predicting: {str(e)}")
            
    try:
        result = await run_with_timeout(inference(), timeout_sec=30.0)
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Fake News Detection Endpoint
@app.post("/api/predict/news")
async def predict_news(payload: TextNewsInput):
    if "news_model" not in models_db:
        raise HTTPException(status_code=503, detail="Fake News LSTM model is not loaded.")
        
    async def inference():
        try:
            # Input validation
            title = payload.title.strip()
            body = payload.text.strip()
            if not body:
                raise ValueError("Body text cannot be empty.")
            if len(body) > 100000:
                raise ValueError("Body text exceeds maximum allowed size of 100,000 characters.")
                
            preprocessor = models_db["news_preprocessor"]
            tokenizer = models_db["news_tokenizer"]
            model = models_db["news_model"]
            device = models_db["device"]
            
            full_text = f"{title} [TITLE_END] {body}"
            
            # Preprocess & Clean
            cleaned_text = preprocessor.transform([full_text])
            
            # Tokenize
            seqs, lens = tokenizer.transform(cleaned_text, max_len=256)
            seqs, lens = seqs.to(device), lens.to(device)
            
            # Inference
            with torch.no_grad():
                logits = model(seqs, lens)
                prob = torch.sigmoid(logits).squeeze(-1).item()
                
            prediction = int(prob > 0.5)
            class_label = "Fake News" if prediction == 1 else "True News"
            confidence = prob if prediction == 1 else (1.0 - prob)
            
            return {
                "prediction": prediction,
                "label": class_label,
                "confidence": confidence,
                "raw_probability": prob
            }
        except Exception as e:
            logger.error(f"News prediction error: {str(e)}", exc_info=True)
            raise ValueError(f"Error during news text analysis: {str(e)}")
            
    try:
        result = await run_with_timeout(inference(), timeout_sec=30.0)
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Phishing Email Detection Endpoint
@app.post("/api/predict/phishing")
async def predict_phishing(payload: TextPhishingInput):
    if "ph_model" not in models_db:
        raise HTTPException(status_code=503, detail="Phishing Email LSTM model is not loaded.")
        
    async def inference():
        try:
            # Input validation
            body = payload.text.strip()
            if not body:
                raise ValueError("Email body text cannot be empty.")
            if len(body) > 100000:
                raise ValueError("Email text exceeds maximum allowed size of 100,000 characters.")
                
            preprocessor = models_db["ph_preprocessor"]
            tokenizer = models_db["ph_tokenizer"]
            model = models_db["ph_model"]
            device = models_db["device"]
            
            # Preprocess & Clean
            cleaned_text = preprocessor.transform([body])
            
            # Tokenize
            seqs, lens = tokenizer.transform(cleaned_text, max_len=256)
            seqs, lens = seqs.to(device), lens.to(device)
            
            # Inference
            with torch.no_grad():
                logits = model(seqs, lens)
                prob = torch.sigmoid(logits).squeeze(-1).item()
                
            prediction = int(prob > 0.5)
            class_label = "Phishing Email" if prediction == 1 else "Safe Email"
            confidence = prob if prediction == 1 else (1.0 - prob)
            
            return {
                "prediction": prediction,
                "label": class_label,
                "confidence": confidence,
                "raw_probability": prob
            }
        except Exception as e:
            logger.error(f"Phishing prediction error: {str(e)}", exc_info=True)
            raise ValueError(f"Error during phishing text analysis: {str(e)}")
            
    try:
        result = await run_with_timeout(inference(), timeout_sec=30.0)
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Deepfake Image Detection Endpoint
@app.post("/api/predict/deepfake")
async def predict_deepfake(file: UploadFile = File(...)):
    if "resnet_model" not in models_db:
        raise HTTPException(status_code=503, detail="Deepfake ResNet classification model is not loaded.")
        
    async def inference():
        try:
            # Input validation: check file format
            extension = file.filename.split(".")[-1].lower()
            if extension not in ("jpg", "jpeg", "png"):
                raise ValueError("Invalid file format. Only JPG, JPEG, and PNG images are supported.")
                
            img_bytes = await file.read()
            if len(img_bytes) == 0:
                raise ValueError("Uploaded image file is empty.")
            if len(img_bytes) > 10 * 1024 * 1024: # 10MB limit
                raise ValueError("Image file size exceeds the 10MB limit.")
                
            # Try loading image using PIL
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                raise ValueError("Failed to decode image. File might be corrupted or malformed.")
                
            model = models_db["resnet_model"]
            device = models_db["device"]
            
            # Evaluation transforms matching training pipeline
            import torchvision.transforms as T
            eval_tf = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize([0.5]*3, [0.5]*3)
            ])
            
            img_orig = eval_tf(img).unsqueeze(0).to(device)
            img_flipped = eval_tf(img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
            
            # Inference with Test-Time Augmentation (TTA)
            with torch.no_grad():
                logits_orig = model(img_orig)
                logits_flipped = model(img_flipped)
                
                prob_orig = torch.sigmoid(logits_orig).squeeze(-1).item()
                prob_flipped = torch.sigmoid(logits_flipped).squeeze(-1).item()
                
                # Average probability across TTA views
                prob = (prob_orig + prob_flipped) / 2.0
                
            # Calibrated decision threshold (0.4) to catch high-quality GAN fakes
            decision_threshold = 0.4
            prediction = int(prob > decision_threshold)
            class_label = "Deepfake Face" if prediction == 1 else "Real Face"
            confidence = prob if prediction == 1 else (1.0 - prob)
            
            return {
                "prediction": prediction,
                "label": class_label,
                "confidence": confidence,
                "raw_probability": prob,
                "raw_prob_original": prob_orig,
                "raw_prob_flipped": prob_flipped
            }
        except Exception as e:
            logger.error(f"Deepfake prediction error: {str(e)}", exc_info=True)
            raise ValueError(f"Error during deepfake face analysis: {str(e)}")
            
    try:
        result = await run_with_timeout(inference(), timeout_sec=30.0)
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Mount static directory to serve HTML, CSS, JS dashboard interface
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    logger.warning("Static directory not found. Please place static files in the static/ folder.")


# Fallback for root route in case static files mount fails
@app.get("/")
async def root():
    return HTMLResponse(
        content="""
        <html>
            <head><title>MultiShield Threat Detection API</title></head>
            <body>
                <h1>MultiShield threat detection API is running</h1>
                <p>Please place frontend files in the <code>static/</code> directory to view the dashboard.</p>
                <p>API documentation is available at <a href="/docs">/docs</a>.</p>
            </body>
        </html>
        """
    )


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting MultiShield server on {HOST}:{PORT}")
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
