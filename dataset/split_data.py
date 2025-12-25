"""
Data Splitting Script for Student Journals Dataset
This script loads the CSV and creates train/validation splits
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), "student_journals.csv")
TRAIN_RATIO = 0.8  # 80% training, 20% validation
RANDOM_STATE = 42  # For reproducibility

# Label mapping
LABEL_MAP = {
    0: "Healthy",
    1: "Stressed", 
    2: "Burnout"
}


def load_data():
    """Load the student journals dataset"""
    df = pd.read_csv(DATA_PATH)
    print(f"📊 Loaded {len(df)} journal entries")
    print(f"📈 Label distribution:\n{df['label'].value_counts().sort_index()}")
    return df


def split_data(df, train_ratio=TRAIN_RATIO, random_state=RANDOM_STATE):
    """
    Split data into training and validation sets
    Uses stratified splitting to maintain label distribution
    """
    train_df, val_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        stratify=df['label']  # Maintains proportion of each class
    )
    
    print(f"\n✅ Data split complete:")
    print(f"   Training samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")
    
    return train_df, val_df


def get_train_val_data():
    """
    Main function to get train and validation datasets
    Returns: (train_df, val_df)
    """
    df = load_data()
    train_df, val_df = split_data(df)
    return train_df, val_df


def analyze_data():
    """Analyze the dataset and print statistics"""
    df = load_data()
    
    print("\n" + "="*50)
    print("📋 DATASET ANALYSIS")
    print("="*50)
    
    # Label distribution
    print("\n🏷️ Label Distribution:")
    for label, name in LABEL_MAP.items():
        count = len(df[df['label'] == label])
        percentage = (count / len(df)) * 100
        print(f"   {label} ({name}): {count} samples ({percentage:.1f}%)")
    
    # Text length statistics
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    
    print("\n📝 Text Statistics:")
    print(f"   Average characters: {df['text_length'].mean():.1f}")
    print(f"   Average words: {df['word_count'].mean():.1f}")
    print(f"   Min words: {df['word_count'].min()}")
    print(f"   Max words: {df['word_count'].max()}")
    
    return df


if __name__ == "__main__":
    # Run analysis when script is executed directly
    analyze_data()
    
    print("\n" + "="*50)
    print("🔄 PERFORMING TRAIN/VALIDATION SPLIT")
    print("="*50)
    
    train_df, val_df = get_train_val_data()
    
    print("\n📊 Training set label distribution:")
    print(train_df['label'].value_counts().sort_index())
    
    print("\n📊 Validation set label distribution:")
    print(val_df['label'].value_counts().sort_index())
