"""
FastAPI Backend Server for Academic Burnout Prevention System
Provides REST API endpoints for burnout prediction and advisory services
"""

import os
import sys
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.predict import BurnoutPredictor, get_predictor
from backend.advisor import BurnoutAdvisor, get_advisor

# ============================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================

class JournalEntry(BaseModel):
    """Schema for a journal entry input"""
    text: str = Field(
        ..., 
        min_length=10, 
        max_length=5000,
        description="The journal entry text to analyze",
        example="I've been feeling overwhelmed with assignments lately and can't seem to catch up."
    )


class BatchJournalEntries(BaseModel):
    """Schema for batch journal entries"""
    entries: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of journal entries to analyze (max 10)"
    )


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    label: str
    label_id: int
    confidence: float
    probabilities: dict
    risk_level: str
    text_preview: str


class AdviceResponse(BaseModel):
    """Schema for advice response"""
    burnout_level: str
    severity_score: float
    summary: str
    recommendations: list
    emergency_resources: Optional[dict]
    follow_up: str
    quick_tip: str


class FullAnalysisResponse(BaseModel):
    """Schema for full analysis (prediction + advice)"""
    timestamp: str
    prediction: dict
    advice: dict


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str
    model_loaded: bool
    timestamp: str
    version: str


# ============================================
# APPLICATION SETUP
# ============================================

# Global instances
predictor: BurnoutPredictor = None
advisor: BurnoutAdvisor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global predictor, advisor
    
    print("\n🚀 Starting Academic Burnout Prevention API...")
    
    # Initialize predictor and advisor
    predictor = get_predictor()
    advisor = get_advisor()
    
    print("✅ API Server is ready!")
    print("📚 Documentation available at: http://localhost:8000/docs")
    
    yield
    
    print("\n👋 Shutting down API server...")


# Create FastAPI app
app = FastAPI(
    title="Academic Burnout Prevention API",
    description="""
    🎓 **Academic Burnout Prevention and Rule-Based Advisory System**
    
    This API uses a fine-tuned BERT model to analyze student journal entries 
    and detect signs of burnout. It provides:
    
    - **Burnout Detection**: Classifies entries as Healthy, Stressed, or Burnout
    - **Risk Assessment**: Provides confidence scores and risk levels
    - **Personalized Advice**: Rule-based recommendations tailored to the detected state
    - **Emergency Resources**: Crisis support information when burnout is detected
    
    ## How to Use
    
    1. Submit a journal entry to `/predict` for classification
    2. Use `/advice` to get recommendations based on a prediction
    3. Use `/analyze` for a complete analysis (prediction + advice)
    
    ---
    *Built with ❤️ for student mental health*
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🎓 Welcome to the Academic Burnout Prevention API",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict - Analyze a journal entry",
            "advice": "POST /advice - Get recommendations",
            "analyze": "POST /analyze - Full analysis",
            "batch": "POST /predict/batch - Batch predictions"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy",
        "model_loaded": predictor.is_loaded if predictor else False,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_burnout(entry: JournalEntry):
    """
    Analyze a journal entry for burnout indicators
    
    - **text**: The journal entry text (10-5000 characters)
    
    Returns the predicted burnout level, confidence score, and risk assessment.
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure the model is trained and available."
        )
    
    # Get prediction
    result = predictor.predict(entry.text)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # Add risk level
    result["risk_level"] = predictor.get_risk_level(result)
    
    return result


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(entries: BatchJournalEntries):
    """
    Analyze multiple journal entries at once (max 10)
    
    - **entries**: List of journal entry texts
    
    Returns predictions for each entry.
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded."
        )
    
    results = []
    for text in entries.entries:
        result = predictor.predict(text)
        if "error" not in result:
            result["risk_level"] = predictor.get_risk_level(result)
        results.append(result)
    
    return {
        "count": len(results),
        "predictions": results
    }


@app.post("/advice", response_model=AdviceResponse, tags=["Advisory"])
async def get_advice_endpoint(entry: JournalEntry):
    """
    Get personalized recommendations based on journal entry analysis
    
    - **text**: The journal entry text
    
    Returns tailored advice and recommendations based on the detected burnout level.
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded."
        )
    
    # First get prediction
    prediction = predictor.predict(entry.text)
    
    if "error" in prediction:
        raise HTTPException(status_code=500, detail=prediction["error"])
    
    # Get advice
    advice = advisor.get_recommendations(prediction)
    advice["quick_tip"] = advisor.get_quick_tip(prediction["label_id"])
    
    return advice


@app.post("/analyze", response_model=FullAnalysisResponse, tags=["Analysis"])
async def full_analysis(entry: JournalEntry):
    """
    Complete analysis: prediction + personalized advice
    
    - **text**: The journal entry text
    
    Returns both the burnout prediction and tailored recommendations.
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded."
        )
    
    # Get prediction
    prediction = predictor.predict(entry.text)
    
    if "error" in prediction:
        raise HTTPException(status_code=500, detail=prediction["error"])
    
    prediction["risk_level"] = predictor.get_risk_level(prediction)
    
    # Get advice
    advice = advisor.get_recommendations(prediction)
    advice["quick_tip"] = advisor.get_quick_tip(prediction["label_id"])
    
    return {
        "timestamp": datetime.now().isoformat(),
        "prediction": prediction,
        "advice": advice
    }


@app.get("/labels", tags=["Information"])
async def get_labels():
    """Get information about the classification labels"""
    return {
        "labels": {
            0: {
                "name": "Healthy",
                "description": "Student shows signs of good mental health and balanced academic life",
                "color": "green"
            },
            1: {
                "name": "Stressed",
                "description": "Student shows signs of academic stress that should be addressed",
                "color": "yellow"
            },
            2: {
                "name": "Burnout",
                "description": "Student shows severe signs of burnout requiring immediate attention",
                "color": "red"
            }
        }
    }


@app.get("/resources", tags=["Information"])
async def get_resources():
    """Get mental health resources and support information"""
    return {
        "crisis_lines": [
            {"name": "National Suicide Prevention Lifeline", "number": "988"},
            {"name": "Crisis Text Line", "text": "Text HOME to 741741"},
            {"name": "SAMHSA National Helpline", "number": "1-800-662-4357"}
        ],
        "general_resources": [
            {
                "name": "Campus Counseling Services",
                "description": "Free confidential counseling for enrolled students"
            },
            {
                "name": "Student Health Center",
                "description": "Physical and mental health services"
            },
            {
                "name": "Academic Advising",
                "description": "Help with course load and academic planning"
            }
        ],
        "self_help": [
            "Regular exercise and physical activity",
            "Consistent sleep schedule (7-9 hours)",
            "Balanced nutrition and hydration",
            "Social connections and support",
            "Time management and organization",
            "Mindfulness and relaxation techniques"
        ]
    }


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🎓 ACADEMIC BURNOUT PREVENTION API SERVER")
    print("="*60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
