"""Pydantic schemas for API."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""
    
    question: str = Field(
        ...,
        description="Legal question to answer",
        min_length=10,
        max_length=2000,
        example="What are the fundamental rights under Article 21?",
    )
    max_new_tokens: Optional[int] = Field(
        512,
        description="Maximum tokens to generate",
        ge=50,
        le=2048,
    )
    temperature: Optional[float] = Field(
        0.7,
        description="Sampling temperature (0.0 = deterministic, higher = more random)",
        ge=0.0,
        le=2.0,
    )
    top_p: Optional[float] = Field(
        0.9,
        description="Nucleus sampling parameter",
        ge=0.0,
        le=1.0,
    )
    top_k: Optional[int] = Field(
        50,
        description="Top-k sampling parameter",
        ge=0,
        le=100,
    )
    repetition_penalty: Optional[float] = Field(
        1.1,
        description="Repetition penalty",
        ge=1.0,
        le=2.0,
    )
    
    @validator('question')
    def validate_question(cls, v):
        """Validate question field."""
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What is Public Interest Litigation in Indian law?",
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction."""
    
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    model_name: str = Field(..., description="Name of the model used")
    citations: Optional[List[str]] = Field(
        None,
        description="Extracted legal citations",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata",
    )
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What is PIL?",
                "answer": "Public Interest Litigation (PIL) is...",
                "model_name": "mistral-7b-legal",
                "citations": ["AIR 1981 SC 149"],
                "metadata": {"tokens_generated": 156},
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""
    
    questions: List[str] = Field(
        ...,
        description="List of questions",
        min_items=1,
        max_items=10,
    )
    max_new_tokens: Optional[int] = Field(512, ge=50, le=2048)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    
    @validator('questions')
    def validate_questions(cls, v):
        """Validate questions list."""
        if not v:
            raise ValueError("Questions list cannot be empty")
        
        # Validate each question
        cleaned = []
        for q in v:
            q = q.strip()
            if not q:
                raise ValueError("Question cannot be empty")
            if len(q) < 10:
                raise ValueError(f"Question too short: {q}")
            cleaned.append(q)
        
        return cleaned
    
    class Config:
        schema_extra = {
            "example": {
                "questions": [
                    "What is Article 21?",
                    "What is PIL?",
                    "What is FIR?",
                ],
                "temperature": 0.7,
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""
    
    predictions: List[Dict[str, str]] = Field(
        ...,
        description="List of predictions",
    )
    count: int = Field(..., description="Number of predictions")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {"question": "Q1", "answer": "A1"},
                    {"question": "Q2", "answer": "A2"},
                ],
                "count": 2,
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Name of loaded model")
    gpu_available: Optional[bool] = Field(None, description="Whether GPU is available")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "mistral-7b-legal",
                "gpu_available": True,
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Prediction failed",
                "detail": "Input too long",
                "code": "INPUT_TOO_LONG",
            }
        }


class ModelInfo(BaseModel):
    """Model information schema."""
    
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    parameters: Optional[str] = Field(None, description="Number of parameters")
    description: Optional[str] = Field(None, description="Model description")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "mistral-7b-legal",
                "type": "Mistral-7B",
                "parameters": "7B",
                "description": "Fine-tuned for Indian legal queries",
            }
        }