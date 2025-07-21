from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "/opt/huggingface_models/all-MiniLM-L6-v2"
SVM_MODEL_FILE = os.path.join(os.path.dirname(__file__), "model", "svm_model.pkl")

# Load models once at startup
try:
    logger.info("Loading transformer model...")
    embedder = SentenceTransformer(MODEL_PATH)

    logger.info("Loading SVM classifier...")
    classifier = joblib.load(SVM_MODEL_FILE)

    logger.info("Models loaded successfully.")
except Exception as e:
    logger.critical(f"Model loading failed: {e}")
    raise RuntimeError(f"Model loading failed: {e}")

# Create FastAPI app
app = FastAPI()

# Request body schema
class HeadlineRequest(BaseModel):
    headlines: list[str]

@app.get("/status")
def get_status():
    return {"status": "OK"}

@app.post("/score_headlines")
def score_headlines(req: HeadlineRequest):
    try:
        logger.info("Received request to score headlines.")
        embeddings = embedder.encode(req.headlines)
        predictions = classifier.predict(embeddings)
        return {"labels": predictions.tolist()}
    except Exception as e:
        logger.error(f"Failed to score headlines: {e}")
        raise HTTPException(status_code=500, detail=str(e))
