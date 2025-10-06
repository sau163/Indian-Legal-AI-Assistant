"""FastAPI application for Legal AI Assistant."""
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.model_config import ModelConfig, GenerationConfig
from inference.predictor import LegalAIPredictor
from utils.logging_utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global predictor instance
predictor: Optional[LegalAIPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global predictor
    
    logger.info("Loading model...")
    
    # Load model from environment variable or use default
    import os
    model_path = os.getenv(
        "MODEL_PATH",
        "nisaar/falcon7b-Indian_Law_150Prompts"
    )
    
    try:
        predictor = LegalAIPredictor.from_pretrained(
            model_path=model_path,
            generation_config=GenerationConfig(
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
            ),
        )
        logger.info(f"Model loaded successfully from {model_path}")
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Legal AI Assistant API",
    description="API for Indian Legal AI Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for prediction."""
    question: str = Field(
        ...,
        description="Legal question to answer",
        min_length=10,
        max_length=2000,
    )
    max_new_tokens: Optional[int] = Field(
        None,
        description="Maximum tokens to generate",
        ge=50,
        le=1024,
    )
    temperature: Optional[float] = Field(
        None,
        description="Sampling temperature",
        ge=0.0,
        le=2.0,
    )
    top_p: Optional[float] = Field(
        None,
        description="Nucleus sampling parameter",
        ge=0.0,
        le=1.0,
    )


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    question: str
    answer: str
    model_name: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


# API endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Legal AI Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        model_name=predictor.model.config._name_or_path if predictor else None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate prediction for a legal question."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later.",
        )
    
    try:
        logger.info(f"Received question: {request.question[:100]}...")
        
        # Generate prediction
        answer = predictor.predict(
            question=request.question,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        
        logger.info(f"Generated answer: {answer[:100]}...")
        
        return PredictionResponse(
            question=request.question,
            answer=answer,
            model_name=predictor.model.config._name_or_path,
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/batch-predict")
async def batch_predict(questions: list[str]):
    """Generate predictions for multiple questions."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later.",
        )
    
    if len(questions) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 questions allowed per batch",
        )
    
    try:
        logger.info(f"Received batch of {len(questions)} questions")
        
        answers = predictor.predict_batch(questions)
        
        return {
            "predictions": [
                {"question": q, "answer": a}
                for q, a in zip(questions, answers)
            ],
            "count": len(questions),
        }
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )