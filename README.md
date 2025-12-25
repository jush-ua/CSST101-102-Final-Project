# � Ye Olde Academic Burnout Prevention & Rule-Based Advisory System ⚔️

*Hear ye, hear ye! A most wondrous contraption of machine learning sorcery that doth analyze the journal entries of weary scholars to detect signs of academic burnout and bestow upon them personalized counsel for mental wellness!* 📜✨

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![BERT](https://img.shields.io/badge/Model-BERT-orange.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)

---

## 📜 Table of Contents

- [The Grand Overview](#-the-grand-overview)
- [Enchanted Features](#-enchanted-features)
- [The Castle Structure](#-the-castle-structure)
- [Summoning the Dependencies](#-summoning-the-dependencies)
- [How to Embark Upon This Quest](#-how-to-embark-upon-this-quest)
- [The Sacred API Scrolls](#-the-sacred-api-scrolls)
- [Training Thy Model](#-training-thy-model)
- [The Three Classifications](#-the-three-classifications)
- [Join the Fellowship](#-join-the-fellowship)

---

## 🎯 The Grand Overview

Hark! Academic burnout doth plague many a scholar in these troubled times, marked by exhaustion of the soul, cynicism most foul, and diminished efficacy in one's studies. This mystical system employs the ancient arts of Natural Language Processing (NLP) to:

1. ⚔️ **Detect** - Uncover the level of burnout from student journal entries
2. 🏷️ **Classify** - Sort mental states into three sacred categories: Healthy, Stressed, or Burnout
3. 💡 **Provide** - Bestow personalized, rule-based counsel upon each weary soul

This grand apparatus combines a **fine-tuned BERT model** (a most learned oracle) for accurate text classification with a **rule-based advisory engine** that delivereth actionable wisdom unto thee!

---

## ✨ Enchanted Features

- 🧠 **BERT-Based Classification** - A transformer most wise, fine-tuned for burnout detection
- 📊 **Three-Level Classification** - Healthy, Stressed, and Burnout states
- 💡 **Rule-Based Advisor** - Personalized recommendations bestowed upon each pilgrim
- 🚀 **FastAPI Backend** - A RESTful messenger swift as a falcon
- 📈 **Confidence Scores** - Probability distribution across all classifications
- 🆘 **Emergency Resources** - Aid for those in dire straits
- 📝 **Batch Processing** - Analyze multiple entries in a single incantation

---

## 🏰 The Castle Structure

```
Burnout_Advisor_Project/
│
├── 📂 dataset/                  # The Grand Library of Knowledge
│   ├── student_journals.csv     # Ancient scrolls of training data
│   └── split_data.py            # The Scroll Divider
│
├── 📂 models/                   # The Vault of Trained Minds
│   ├── best_burnout_model/      # Thy finest trained BERT
│   └── tokenizer_config/        # The Dictionary of Understanding
│
├── 📂 backend/                  # The Royal Messenger Service
│   ├── main.py                  # The FastAPI Gateway
│   ├── predict.py               # The Oracle Module
│   └── advisor.py               # The Wise Counselor
│
├── 📂 training/                 # The Training Grounds
│   └── train_bert.py            # The Knight's Training Regimen
│
├── requirements.txt             # The Spellbook of Dependencies
└── README.md                    # This Very Scroll Thou Art Reading
```

---

## 📦 Summoning the Dependencies

### ⚠️ Prerequisites (Tools Thou Must Possess)

Before embarking upon this noble quest, ensure thy machine possesses:

- 🐍 **Python 3.8 or higher** - The serpent language
- 📦 **pip** - The package summoner
- 💻 **A terminal** - Thy command throne

### 🧙‍♂️ Step I: Navigate to the Sacred Project Directory

Open thy terminal (the PowerShell or Command Prompt) and venture forth:

```bash
cd "path\to\Burnout_Advisor_Project"
```

### 🔮 Step II: Summon the Magical Dependencies

Invoke this incantation to install all required enchantments:

```bash
pip install -r requirements.txt
```

*Lo and behold! The following arcane libraries shall be summoned unto thy machine:*

| 📚 Library | 🎯 Purpose |
|------------|-----------|
| `torch` | The Deep Learning Flame 🔥 |
| `transformers` | The BERT Summoning Circle 🤖 |
| `datasets` | The Data Loading Wizardry 📊 |
| `pandas` | The Data Manipulation Arts 🐼 |
| `scikit-learn` | Machine Learning Utilities ⚙️ |
| `fastapi` | The Swift API Framework 🚀 |
| `uvicorn` | The Server Steed 🐎 |
| `seaborn` | Visualization Sorcery 📈 |

*Patience, noble scholar! This process may taketh a few minutes...* ⏳

---

## 🗡️ How to Embark Upon This Quest

### 📖 Quest I: Train the Oracle (The BERT Model)

First, thou must train the mystical BERT oracle upon the ancient scrolls of student journals:

```bash
python training/train_bert.py
```

*This sacred ritual shall:*
- 📜 Load and prepare the dataset scrolls
- 🏋️ Fine-tune BERT through rigorous training
- 💾 Save thy best model unto `models/best_burnout_model/`
- 📊 Generate training metrics and a confusion matrix

**Behold! The Expected Output:**
```
🎓 ACADEMIC BURNOUT DETECTION - BERT FINE-TUNING
============================================================
⏰ Started at: 2025-12-25 10:00:00
💻 Device: cuda (or cpu)
🤖 Base Model: bert-base-uncased
...
📈 EVALUATION RESULTS:
   Accuracy:  0.9200
   Precision: 0.9180
   Recall:    0.9200
   F1 Score:  0.9185
```

*Rejoice! Thy model hath been trained!* 🎉

---

### 🏃 Quest II: Awaken the API Server

Once thy model is trained, summon the FastAPI server to life:

```bash
cd backend
python main.py
```

Or invoke uvicorn directly with this incantation:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**🏰 Access Points to Thy Kingdom:**

| 🚪 Portal | 🔗 URL | 📋 Description |
|-----------|--------|----------------|
| 🏠 API Root | http://localhost:8000 | The main gate |
| 📚 Swagger Docs | http://localhost:8000/docs | Interactive scrolls |
| 📖 ReDoc | http://localhost:8000/redoc | Alternative documentation |

---

### 🧪 Quest III: Test Thy Creation

**Using the cURL Messenger:**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "I have been feeling overwhelmed with assignments and cannot sleep properly."}'
```

**Using the Python Familiar:**

```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={"text": "I can't take this anymore. Everything feels hopeless."}
)

result = response.json()
print(f"🏷️ Prediction: {result['prediction']['label']}")
print(f"📊 Confidence: {result['prediction']['confidence']:.2%}")
print(f"💬 Summary: {result['advice']['summary']}")
```

---

## 🗺️ Quick Start Guide (For the Impatient Knight) ⚡

```bash
# Step 1: Navigate to the project
cd Burnout_Advisor_Project

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Train the model
python training/train_bert.py

# Step 4: Start the server
cd backend
python main.py

# Step 5: Visit http://localhost:8000/docs in thy browser! 🎉
```

---

## 📚 The Sacred API Scrolls

### 🗡️ Endpoints of Power

| ⚔️ Method | 🚪 Endpoint | 📜 Description |
|-----------|-------------|----------------|
| GET | `/` | The welcome proclamation |
| GET | `/health` | Check if the oracle liveth |
| POST | `/predict` | Divine the burnout level |
| POST | `/predict/batch` | Batch divinations (max 10) |
| POST | `/advice` | Receive wise counsel |
| POST | `/analyze` | Full prophecy (predict + advice) |
| GET | `/labels` | The classification codex |
| GET | `/resources` | Mental health scrolls |

### 📜 Example Request/Response

**POST /analyze** *(The Full Prophecy)*

Request:
```json
{
  "text": "I've been staying up all night studying and still feel like I'm failing."
}
```

Response:
```json
{
  "timestamp": "2025-12-25T10:30:00",
  "prediction": {
    "label": "Stressed",
    "label_id": 1,
    "confidence": 0.82,
    "probabilities": {
      "Healthy": 0.08,
      "Stressed": 0.82,
      "Burnout": 0.10
    },
    "risk_level": "🟡 MODERATE RISK - Significant stress detected"
  },
  "advice": {
    "burnout_level": "STRESSED",
    "severity_score": 5.5,
    "summary": "Thy entry showeth signs of academic stress. 'Tis common and manageable.",
    "recommendations": [...],
    "follow_up": "We recommend journaling daily this week."
  }
}
```

---

## 🏋️ Training Thy Model

### 📜 The Sacred Dataset Format

The training scrolls (`dataset/student_journals.csv`) must contain two columns:

| 📝 text | 🏷️ label |
|---------|----------|
| "I'm feeling great today!" | 0 |
| "Too much homework, feeling stressed" | 1 |
| "I can't take this anymore" | 2 |

### ⚙️ The Arcane Hyperparameters

| 🔧 Parameter | 📊 Value |
|--------------|----------|
| Base Model | bert-base-uncased |
| Max Length | 128 tokens |
| Batch Size | 8 |
| Learning Rate | 2e-5 |
| Epochs | 10 |
| Early Stopping | 3 epochs patience |

### 💡 Words of Wisdom for Training

- 📚 **More Data**: Gather more journal entries for greater accuracy
- ⚡ **GPU**: Training proceedeth faster with a CUDA-enabled GPU
- 🎛️ **Hyperparameters**: Adjust the learning rate and batch size as needed
- ⚖️ **Class Balance**: Ensure similar quantities of samples per class

---

## 🏷️ The Three Classifications

*The oracle shall sort all souls into one of three categories:*

| 🔢 Label ID | 📛 Name | 📋 Description | 🔍 Indicators |
|-------------|---------|----------------|---------------|
| 0 | **🟢 Healthy** | A sound mind and spirit | Positive outlook, balanced life, restful slumber |
| 1 | **🟡 Stressed** | Burdened but manageable | Worry, pressure, troubled sleep, yet recoverable |
| 2 | **🔴 Burnout** | Severe exhaustion of the soul | Complete exhaustion, hopelessness, physical ailments |

### ⚠️ Risk Levels

- 🟢 **LOW RISK**: Thy mental state is most healthy, noble scholar!
- 🟡 **MODERATE/ELEVATED**: Signs of stress detected, take heed!
- 🔴 **HIGH/CRITICAL**: Burnout most severe! Seek aid forthwith!

---

## 🤝 Join the Fellowship

*Contributions from fellow knights and scholars art most welcome!*

1. 📚 **Add Training Data**: More diverse journal entries improve thy oracle's wisdom
2. 💡 **Improve Recommendations**: Enhance the rule-based advisory chambers
3. 🎨 **Build Frontend**: Craft a user-friendly web interface for the masses
4. 🌍 **Add Languages**: Enable multilingual burnout detection
5. 📱 **Mobile App**: Develop iOS/Android applications for scholars on the go

---

## ⚠️ A Most Important Disclaimer

*Hear this warning well, noble reader!*

This system is designed as an **educational tool** and **early warning system**. It is **NOT** a replacement for the counsel of professional healers and mental health practitioners! 🏥

If thou or someone thou knowest is experiencing severe burnout or mental health crisis:
- 🏛️ Contact thy campus counseling center
- 📞 Call the **988 Suicide & Crisis Lifeline**
- 💬 Text **HOME** to **741741** (Crisis Text Line)
- 🆘 Seek professional help immediately!

*Remember: Asking for help is a sign of courage, not weakness!* 💪

---

## 📜 License

This project is for educational purposes within the realm of academia.

---

## 👥 The Noble Authors

*This grand work was crafted by:*

🎓 **CSST 101** - Artificial Intelligence Project  
🏛️ **Laguna State Polytechnic University (LSPU)**  
📅 3rd Year, 1st Semester  

---

## 🙏 Final Words

*"Take care of thy mind, for it is the castle from which all thy battles are fought."*

---

*Forged with ❤️ and ⚔️ for the mental health and wellness of scholars everywhere!*

```
   ⚔️  STAY STRONG, NOBLE SCHOLARS!  ⚔️
        _____
       |     |
       | 🎓  |
       |_____|
         |||
         |||
    _____|_|_____
   |             |
   |   YOU ARE   |
   |   WORTHY!   |
   |_____________|
```

