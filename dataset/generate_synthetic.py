"""
🏭 The Oracle's Data Forge V2 (Slang Edition)
=============================================
Generates synthetic student journal entries for training.
Creates 1000 rows of REALISTIC student data with casual language.
"""

import pandas as pd
import random
import os

# --- LABEL 0: HEALTHY (Chill, Good, Okay) ---
healthy_phrases = [
    "I feel great", "im good", "doing okay", "finished my hw",
    "ready for the test", "chilling", "watching netflix", "slept well",
    "easy day", "balanced life", "feeling solid", "all good",
    "love this class", "happy", "not stressed"
]

# --- LABEL 1: STRESSED (Tired, Rushing, Anxious) ---
stressed_phrases = [
    "im tired", "so tired", "need sleep", "too much homework",
    "stressed out", "rushing", "deadline soon", "cant focus",
    "coffee addict", "head hurts", "up all night", "grinding",
    "busy week", "anxious", "scared of failing"
]

# --- LABEL 2: BURNOUT (Dead, Done, Quit) ---
burnout_phrases = [
    "i give up", "im done", "cant do this anymore", "crying",
    "hate my life", "failed everything", "empty", "numb",
    "want to drop out", "broken", "no energy", "help me",
    "collapsing", "nothing matters", "so depressed"
]

# --- THE MIXER ---
data = []

print("🏭 Starting Factory V2 (Slang Edition)...")

# Generate 1000 entries (More data = Smarter AI)
for _ in range(1000):
    label = random.choice([0, 1, 2])
    
    if label == 0:
        base = random.choice(healthy_phrases)
    elif label == 1:
        base = random.choice(stressed_phrases)
    else:
        base = random.choice(burnout_phrases)
    
    # Chaos Factor: Add randomness to make it robust
    if random.random() > 0.5:
        # Add punctuation sometimes
        text = base + "."
    elif random.random() > 0.8:
        # Add emphasis
        text = base + "!!!"
    else:
        # Raw text
        text = base

    # Lowercase everything (Helps the AI match patterns)
    text = text.lower()
    
    data.append({"text": text, "label": label})

# --- EXPORT ---
df = pd.DataFrame(data)

# Save to the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "student_journals.csv")

df.to_csv(output_path, index=False)
print(f"✅ Generated 1000 rows of REALISTIC student data.")
print(f"📁 Saved to: {output_path}")

# Show distribution
print("\n📊 Label Distribution:")
print(df['label'].value_counts().sort_index())
print("\n🔍 Sample entries:")
print(df.sample(5))
