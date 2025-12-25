"""
Prediction Module for Academic Burnout Detection
Loads the fine-tuned BERT model and makes predictions on new journal entries
"""

import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from typing import Dict, Tuple
import torch.nn.functional as F

# ============================================
# CONFIGURATION
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_burnout_model")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer_config")
MAX_LENGTH = 128

# Label mapping
LABEL_MAP = {
    0: "Healthy",
    1: "Stressed",
    2: "Burnout"
}

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BurnoutPredictor:
    """
    Class to handle burnout prediction from student journal entries
    """
    
    def __init__(self):
        """Initialize the predictor by loading the model and tokenizer"""
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the fine-tuned BERT model and tokenizer
        Returns True if successful, False otherwise
        """
        try:
            print("📚 Loading model and tokenizer...")
            
            # Check if model exists
            if not os.path.exists(MODEL_PATH):
                print(f"❌ Model not found at {MODEL_PATH}")
                print("   Please run training first: python training/train_bert.py")
                return False
            
            # Load tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
            
            # Load model
            self.model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
            self.model.to(DEVICE)
            self.model.eval()
            
            self.is_loaded = True
            print(f"✅ Model loaded successfully on {DEVICE}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def predict(self, text: str) -> Dict:
        """
        Predict the burnout level for a given journal entry
        
        Args:
            text: The journal entry text to analyze
            
        Returns:
            Dictionary containing:
            - label: Predicted label (Healthy, Stressed, Burnout)
            - label_id: Numeric label (0, 1, 2)
            - confidence: Confidence score (0-1)
            - probabilities: Probability for each class
        """
        if not self.is_loaded:
            if not self.load_model():
                return {"error": "Model not loaded"}
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get probabilities using softmax
                probabilities = F.softmax(logits, dim=1)
                
                # Get predicted class
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # Get all probabilities
                probs_dict = {
                    LABEL_MAP[i]: round(probabilities[0][i].item(), 4)
                    for i in range(len(LABEL_MAP))
                }
            
            return {
                "label": LABEL_MAP[predicted_class],
                "label_id": predicted_class,
                "confidence": round(confidence, 4),
                "probabilities": probs_dict,
                "text_preview": text[:100] + "..." if len(text) > 100 else text
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict burnout levels for multiple journal entries
        
        Args:
            texts: List of journal entry texts
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]
    
    def get_risk_level(self, prediction: Dict) -> str:
        """
        Convert prediction to risk level description
        """
        label_id = prediction.get("label_id", 0)
        confidence = prediction.get("confidence", 0)
        
        if label_id == 2:  # Burnout
            if confidence > 0.8:
                return "🔴 CRITICAL - Immediate intervention recommended"
            else:
                return "🔴 HIGH RISK - Burnout indicators detected"
        elif label_id == 1:  # Stressed
            if confidence > 0.8:
                return "🟡 MODERATE RISK - Significant stress detected"
            else:
                return "🟡 ELEVATED - Some stress indicators present"
        else:  # Healthy
            return "🟢 LOW RISK - Healthy mental state indicated"


# Global predictor instance
_predictor = None


def get_predictor() -> BurnoutPredictor:
    """Get or create the global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = BurnoutPredictor()
        _predictor.load_model()
    return _predictor


def predict_burnout(text: str) -> Dict:
    """
    Convenience function to predict burnout from text
    
    Args:
        text: Journal entry text
        
    Returns:
        Prediction dictionary
    """
    predictor = get_predictor()
    return predictor.predict(text)


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    # Test the predictor
    print("\n" + "="*60)
    print("🧪 BURNOUT PREDICTOR - TEST MODE")
    print("="*60 + "\n")
    
    predictor = BurnoutPredictor()
    
    if predictor.load_model():
        # Test cases
        test_entries = [
            "I'm feeling great today! Finished all my work and relaxed with friends.",
            "I have so many deadlines coming up. I'm getting really stressed out.",
            "I can't take this anymore. I'm exhausted and nothing seems to matter.",
        ]
        
        print("\n📝 Test Predictions:\n" + "-"*50)
        
        for entry in test_entries:
            result = predictor.predict(entry)
            risk = predictor.get_risk_level(result)
            
            print(f"\n📄 Text: \"{entry[:60]}...\"")
            print(f"   Prediction: {result['label']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Risk Level: {risk}")
            print(f"   Probabilities: {result['probabilities']}")
    else:
        print("⚠️ Could not load model. Please train the model first.")
