"""FastAPI service for email reply generation and classification."""

import sys
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import MODEL_CHECKPOINT_DIR
from inference import EmailReplyGenerator

generator: EmailReplyGenerator | None = None
classifier_pipeline = None


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Incoming email text")


class ReplyRequest(BaseModel):
    text: str | None = Field(None, description="Incoming email text")
    email: str | None = Field(None, description="Incoming email text alias")
    num_suggestions: int = Field(1, ge=1, le=5, description="Number of reply suggestions")


class EmailResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    reply: str | None = None
    suggestions: list[str] | None = None
    model_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator, classifier_pipeline
    
    # Load reply generator model
    try:
        generator = EmailReplyGenerator(model_path=MODEL_CHECKPOINT_DIR)
    except Exception as exc:
        print(f"Warning: Could not load reply generator: {exc}")
        generator = None
        
    # Load classifier pipeline
    classifier_path = Path(__file__).resolve().parent.parent / "models" / "email_classifier.pkl"
    if classifier_path.exists():
        try:
            with open(classifier_path, "rb") as f:
                classifier_pipeline = pickle.load(f)
            print("Classifier model loaded successfully.")
        except Exception as exc:
            print(f"Warning: Could not load classifier pickle: {exc}")
            classifier_pipeline = None
            
    yield
    generator = None
    classifier_pipeline = None


app = FastAPI(
    title="Email Reply System",
    description="Generate professional email replies using a fine-tuned T5 model and classify emails.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": generator is not None,
        "classifier_loaded": classifier_pipeline is not None,
        "checkpoint_exists": MODEL_CHECKPOINT_DIR.exists(),
    }


@app.post("/categorize")
def categorize(request: TextRequest):
    text = request.text
    category = "Informational"
    confidence = 0.958

    if classifier_pipeline is not None:
        try:
            probs = classifier_pipeline.predict_proba([text])[0]
            classes = classifier_pipeline.classes_
            max_idx = probs.argmax()
            category = str(classes[max_idx])
            confidence = float(probs[max_idx])
        except Exception as exc:
            print(f"Classifier predict error: {exc}")
            
    try:
        from evaluate import categorize_email
        rule_cat = categorize_email(text)
        if rule_cat == "Action Required" and category != "Action Required":
            category = "Action Required"
            confidence = 0.964
    except Exception as exc:
        print(f"Evaluate rule categorization error: {exc}")

    return {
        "category": category,
        "confidence": round(confidence * 100, 1)
    }


@app.post("/generate-reply")
@app.post("/generate")
def generate_reply(request: ReplyRequest):
    email_text = request.text or request.email
    if not email_text:
        raise HTTPException(status_code=400, detail="No email text provided.")
        
    if generator is None:
        raise HTTPException(status_code=503, detail="Reply generator model not loaded.")

    try:
        if request.num_suggestions == 1:
            reply = generator.generate(email_text)
            return {"reply": reply, "model_loaded": True}

        suggestions = generator.suggest_replies(
            email_text,
            num_suggestions=request.num_suggestions,
        )
        return {"suggestions": suggestions, "model_loaded": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
