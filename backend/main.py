"""
FastAPI Backend Server for Academic Burnout Prevention System
Provides REST API endpoints for burnout prediction and advisory services
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.predict import BurnoutPredictor, get_predictor
from backend.advisor import BurnoutAdvisor, get_advisor

# ============================================
# LOGGING CONFIGURATION
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# ============================================
# CUSTOM EXCEPTIONS
# ============================================

class OracleException(Exception):
    """Base exception for the Burnout Oracle"""
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ModelNotLoadedException(OracleException):
    """Exception when model is not loaded"""
    def __init__(self):
        super().__init__(
            message="The Oracle's wisdom (model) hath not been loaded! Please ensure the model is trained and available.",
            status_code=503,
            details={"error_code": "MODEL_NOT_LOADED", "suggestion": "Train the model or check the model path"}
        )


class PredictionException(OracleException):
    """Exception during prediction"""
    def __init__(self, original_error: str):
        super().__init__(
            message=f"The Oracle encountered an error whilst divining thy fate: {original_error}",
            status_code=500,
            details={"error_code": "PREDICTION_FAILED", "original_error": original_error}
        )


class InvalidInputException(OracleException):
    """Exception for invalid input"""
    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=400,
            details={"error_code": "INVALID_INPUT"}
        )


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
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Journal entry cannot be empty or contain only whitespace")
        
        text = v.strip()
        
        # Check for gibberish (too many special characters)
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.3:
            raise ValueError("Thy entry contains too many special characters! Please write in plain English.")
        
        # Check for keyboard mashing (words that are too long without spaces)
        words = text.split()
        if len(words) == 0:
            raise ValueError("Please write some actual words, noble scholar!")
        
        # Check average word length - gibberish tends to have very long "words"
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length > 15:
            raise ValueError("Thy words art suspiciously long! Please write normally, not in keyboard-smashing tongue.")
        
        # Check for very long words (keyboard mashing)
        max_word_length = max(len(word) for word in words)
        if max_word_length > 25:
            raise ValueError(f"Hark! A word with {max_word_length} characters? Please write actual words, noble scholar!")
        
        # Check if text has at least some common English patterns
        common_words = {'i', 'im', 'my', 'me', 'the', 'a', 'an', 'is', 'am', 'are', 'was', 'were', 'be', 
                       'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                       'could', 'should', 'can', 'cant', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                       'and', 'but', 'or', 'so', 'just', 'not', 'no', 'yes', 'this', 'that', 'it',
                       'feel', 'feeling', 'felt', 'tired', 'stressed', 'happy', 'sad', 'okay', 'good',
                       'bad', 'help', 'need', 'want', 'today', 'yesterday', 'school', 'work', 'class'}
        
        lower_words = [word.lower().strip('.,!?;\'"()[]{}') for word in words]
        common_count = sum(1 for word in lower_words if word in common_words)
        
        # If no common words found and text is long, it's probably gibberish
        if common_count == 0 and len(words) > 3:
            raise ValueError("The Oracle cannot understand thy tongue! Please write in English.")
        
        # Check for repeated character patterns (like 'asdfasdfasdf')
        if len(text) > 20:
            # Check if any 4-character pattern repeats more than 3 times
            for i in range(len(text) - 4):
                pattern = text[i:i+4].lower()
                if text.lower().count(pattern) > 3 and pattern.isalpha():
                    raise ValueError("Repeated patterns detected! Please write a genuine entry.")
        
        return text


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


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    success: bool = False
    error: str
    error_code: str
    details: Optional[dict] = None
    timestamp: str
    suggestion: Optional[str] = None


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
    
    logger.info("🚀 Starting Academic Burnout Prevention API...")
    
    try:
        # Initialize predictor and advisor
        predictor = get_predictor()
        advisor = get_advisor()
        
        if predictor and predictor.is_loaded:
            logger.info("✅ Model loaded successfully!")
        else:
            logger.warning("⚠️ Model not loaded - some features may be unavailable")
        
        logger.info("✅ API Server is ready!")
        logger.info("📚 Documentation available at: http://localhost:8000/docs")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {str(e)}")
        logger.error(traceback.format_exc())
    
    yield
    
    logger.info("👋 Shutting down API server...")


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
# EXCEPTION HANDLERS
# ============================================

@app.exception_handler(OracleException)
async def oracle_exception_handler(request: Request, exc: OracleException):
    """Handle custom Oracle exceptions"""
    logger.error(f"OracleException: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.message,
            "error_code": exc.details.get("error_code", "UNKNOWN_ERROR"),
            "details": exc.details,
            "timestamp": datetime.now().isoformat(),
            "suggestion": exc.details.get("suggestion")
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with friendly messages"""
    errors = exc.errors()
    error_messages = []
    
    for error in errors:
        field = ".".join(str(loc) for loc in error["loc"])
        msg = error["msg"]
        error_messages.append(f"{field}: {msg}")
    
    logger.warning(f"Validation error: {error_messages}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Thy input hath failed the validation rites!",
            "error_code": "VALIDATION_ERROR",
            "details": {
                "errors": error_messages,
                "raw_errors": [{"field": ".".join(str(loc) for loc in e["loc"]), "message": e["msg"], "type": e["type"]} for e in errors]
            },
            "timestamp": datetime.now().isoformat(),
            "suggestion": "Please check thy input and ensure all fields meet the requirements"
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": str(exc.detail),
            "error_code": f"HTTP_{exc.status_code}",
            "details": None,
            "timestamp": datetime.now().isoformat(),
            "suggestion": None
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected exceptions"""
    logger.error(f"Unexpected error: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "The Oracle hath encountered an unforeseen catastrophe!",
            "error_code": "INTERNAL_SERVER_ERROR",
            "details": {"exception_type": type(exc).__name__},
            "timestamp": datetime.now().isoformat(),
            "suggestion": "Please try again later or contact the royal administrators"
        }
    )


# ============================================
# HELPER FUNCTIONS
# ============================================

def check_model_loaded():
    """Check if the model is loaded and raise exception if not"""
    if not predictor or not predictor.is_loaded:
        raise ModelNotLoadedException()


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "success": True,
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
    # Check model is loaded
    check_model_loaded()
    
    try:
        # Get prediction
        result = predictor.predict(entry.text)
        
        if "error" in result:
            raise PredictionException(result["error"])
        
        # Add risk level
        result["risk_level"] = predictor.get_risk_level(result)
        
        logger.info(f"Prediction successful: {result['label']} (confidence: {result['confidence']:.2%})")
        return result
        
    except OracleException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise PredictionException(str(e))


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(entries: BatchJournalEntries):
    """
    Analyze multiple journal entries at once (max 10)
    
    - **entries**: List of journal entry texts
    
    Returns predictions for each entry.
    """
    # Check model is loaded
    check_model_loaded()
    
    results = []
    errors = []
    
    for idx, text in enumerate(entries.entries):
        try:
            result = predictor.predict(text)
            if "error" not in result:
                result["risk_level"] = predictor.get_risk_level(result)
                result["success"] = True
            else:
                result["success"] = False
                errors.append({"index": idx, "error": result["error"]})
            results.append(result)
        except Exception as e:
            logger.warning(f"Batch prediction error for entry {idx}: {str(e)}")
            results.append({
                "success": False,
                "error": str(e),
                "text_preview": text[:50] + "..." if len(text) > 50 else text
            })
            errors.append({"index": idx, "error": str(e)})
    
    logger.info(f"Batch prediction: {len(results) - len(errors)}/{len(results)} successful")
    
    return {
        "success": len(errors) == 0,
        "count": len(results),
        "successful": len(results) - len(errors),
        "failed": len(errors),
        "predictions": results,
        "errors": errors if errors else None
    }


@app.post("/advice", response_model=AdviceResponse, tags=["Advisory"])
async def get_advice_endpoint(entry: JournalEntry):
    """
    Get personalized recommendations based on journal entry analysis
    
    - **text**: The journal entry text
    
    Returns tailored advice and recommendations based on the detected burnout level.
    """
    # Check model is loaded
    check_model_loaded()
    
    try:
        # First get prediction
        prediction = predictor.predict(entry.text)
        
        if "error" in prediction:
            raise PredictionException(prediction["error"])
        
        # Get advice
        advice = advisor.get_recommendations(prediction)
        advice["quick_tip"] = advisor.get_quick_tip(prediction["label_id"])
        
        logger.info(f"Advice generated for: {prediction['label']}")
        return advice
        
    except OracleException:
        raise
    except Exception as e:
        logger.error(f"Advice generation failed: {str(e)}")
        raise PredictionException(str(e))


@app.post("/analyze", response_model=FullAnalysisResponse, tags=["Analysis"])
async def full_analysis(entry: JournalEntry):
    """
    Complete analysis: prediction + personalized advice
    
    - **text**: The journal entry text
    
    Returns both the burnout prediction and tailored recommendations.
    """
    # Check model is loaded
    check_model_loaded()
    
    try:
        # Get prediction
        prediction = predictor.predict(entry.text)
        
        if "error" in prediction:
            raise PredictionException(prediction["error"])
        
        prediction["risk_level"] = predictor.get_risk_level(prediction)
        
        # Get advice
        advice = advisor.get_recommendations(prediction)
        advice["quick_tip"] = advisor.get_quick_tip(prediction["label_id"])
        
        logger.info(f"Full analysis completed: {prediction['label']}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "advice": advice
        }
        
    except OracleException:
        raise
    except Exception as e:
        logger.error(f"Full analysis failed: {str(e)}")
        raise PredictionException(str(e))


@app.get("/labels", tags=["Information"])
async def get_labels():
    """Get information about the classification labels"""
    return {
        "success": True,
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
        "success": True,
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
    
    logger.info("=" * 60)
    logger.info("🎓 ACADEMIC BURNOUT PREVENTION API SERVER")
    logger.info("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )