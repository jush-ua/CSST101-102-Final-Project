"""
BERT Fine-Tuning Script for Academic Burnout Detection
This script trains a BERT model to classify student journal entries
into three categories: Healthy (0), Stressed (1), Burnout (2)
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset

from dataset.split_data import get_train_val_data, LABEL_MAP

# ============================================
# CONFIGURATION
# ============================================

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "best_burnout_model")
TOKENIZER_SAVE_PATH = os.path.join(BASE_DIR, "models", "tokenizer_config")
LOGS_PATH = os.path.join(BASE_DIR, "training", "logs")

# Model Configuration
MODEL_NAME = "bert-base-uncased"  # Pre-trained BERT model
NUM_LABELS = 3  # Healthy, Stressed, Burnout
MAX_LENGTH = 128  # Maximum token length

# Training Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================
# DATA PREPARATION
# ============================================

def prepare_datasets(tokenizer):
    """
    Load data and convert to HuggingFace Dataset format
    """
    print("📂 Loading and preparing datasets...")
    
    train_df, val_df = get_train_val_data()
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']].reset_index(drop=True))
    
    def tokenize_function(examples):
        """Tokenize the text data"""
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=None
        )
    
    # Tokenize datasets
    print("🔤 Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


# ============================================
# METRICS
# ============================================

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for the model
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_confusion_matrix(trainer, val_dataset):
    """
    Generate and save confusion matrix visualization
    """
    print("📊 Generating confusion matrix...")
    
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=list(LABEL_MAP.values()),
        yticklabels=list(LABEL_MAP.values())
    )
    plt.title('Burnout Detection - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save the plot
    os.makedirs(LOGS_PATH, exist_ok=True)
    plt.savefig(os.path.join(LOGS_PATH, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix saved to {LOGS_PATH}")


# ============================================
# TRAINING
# ============================================

def train_model():
    """
    Main training function
    """
    print("\n" + "="*60)
    print("🎓 ACADEMIC BURNOUT DETECTION - BERT FINE-TUNING")
    print("="*60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Device: {DEVICE}")
    print(f"🤖 Base Model: {MODEL_NAME}")
    print("="*60 + "\n")
    
    # Create directories
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(TOKENIZER_SAVE_PATH, exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)
    
    # Load tokenizer
    print("📚 Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(tokenizer)
    
    # Load pre-trained BERT model
    print("🧠 Loading pre-trained BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=LABEL_MAP,
        label2id={v: k for k, v in LABEL_MAP.items()}
    )
    model.to(DEVICE)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(LOGS_PATH, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(LOGS_PATH, "tensorboard"),
        logging_steps=10,
        report_to=["tensorboard"],
        save_total_limit=2,
        seed=42
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    print("\n🏋️ Starting training...")
    print("-"*40)
    
    train_result = trainer.train()
    
    print("\n" + "-"*40)
    print("✅ Training completed!")
    
    # Evaluate
    print("\n📊 Evaluating model...")
    eval_results = trainer.evaluate()
    
    print("\n📈 EVALUATION RESULTS:")
    print(f"   Accuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"   Precision: {eval_results['eval_precision']:.4f}")
    print(f"   Recall:    {eval_results['eval_recall']:.4f}")
    print(f"   F1 Score:  {eval_results['eval_f1']:.4f}")
    
    # Generate confusion matrix
    plot_confusion_matrix(trainer, val_dataset)
    
    # Save the best model
    print("\n💾 Saving model and tokenizer...")
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)
    
    print(f"✅ Model saved to: {MODEL_SAVE_PATH}")
    print(f"✅ Tokenizer saved to: {TOKENIZER_SAVE_PATH}")
    
    # Save training summary
    summary_path = os.path.join(LOGS_PATH, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("ACADEMIC BURNOUT DETECTION - TRAINING SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Device: {DEVICE}\n\n")
        f.write("Hyperparameters:\n")
        f.write(f"  - Batch Size: {BATCH_SIZE}\n")
        f.write(f"  - Learning Rate: {LEARNING_RATE}\n")
        f.write(f"  - Epochs: {NUM_EPOCHS}\n")
        f.write(f"  - Max Length: {MAX_LENGTH}\n\n")
        f.write("Results:\n")
        f.write(f"  - Accuracy: {eval_results['eval_accuracy']:.4f}\n")
        f.write(f"  - Precision: {eval_results['eval_precision']:.4f}\n")
        f.write(f"  - Recall: {eval_results['eval_recall']:.4f}\n")
        f.write(f"  - F1 Score: {eval_results['eval_f1']:.4f}\n")
    
    print(f"📝 Training summary saved to: {summary_path}")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    
    return model, tokenizer


if __name__ == "__main__":
    train_model()
