# 🎓 Academic Burnout Prevention & Rule-Based Advisory System

*A machine learning-powered system that analyzes student journal entries to detect signs of academic burnout and provides personalized mental wellness recommendations.*

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![BERT](https://img.shields.io/badge/Model-BERT-orange.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)

---

## 📜 Table of Contents

- [Overview](#-overview)
- [Technology Stack](#-technology-stack)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Getting Started](#-getting-started)
- [Quick Start Guide](#-quick-start-guide)
- [Interactive Chat Interface](#-interactive-chat-interface)
- [API Documentation](#-api-documentation)
- [Model Training](#-model-training)
- [Classification Categories](#-classification-categories)
- [Contributing](#-contributing)
- [Git & Git LFS Guide for Contributors](#-git--git-lfs-guide-for-contributors)

---

## 🎯 Overview

Academic burnout is a significant challenge faced by college students, characterized by emotional exhaustion, cynicism, and reduced academic performance. This system leverages Natural Language Processing (NLP) to:

1. ⚔️ **Detect** - Identify burnout levels from student journal entries
2. 🏷️ **Classify** - Categorize mental states into three levels: Healthy, Stressed, or Burnout
3. 💡 **Provide** - Deliver personalized, rule-based recommendations

The system combines a **fine-tuned BERT model** for accurate text classification with a **rule-based advisory engine** that provides actionable guidance to students.

---

## 🛠️ Technology Stack

*Tools and technologies powering this system:*

### 🐍 Core Language

| 🔧 Tool | 📋 Description |
|---------|----------------|
| ![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white) | Primary programming language |

### 🧠 Machine Learning & AI

| 🔧 Tool | 📋 Description |
|---------|----------------|
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) | Deep learning framework |
| ![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-FFD21E?style=for-the-badge) | Hugging Face library for BERT and transformer models |
| ![BERT](https://img.shields.io/badge/BERT-base--uncased-orange?style=for-the-badge) | Pre-trained transformer model fine-tuned for burnout detection |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) | Machine learning utilities for metrics and evaluation |

### 🚀 Backend & API Framework

| 🔧 Tool | 📋 Description |
|---------|----------------|
| ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white) | Modern, high-performance web framework for building APIs |
| ![Uvicorn](https://img.shields.io/badge/Uvicorn-0.22+-499848?style=for-the-badge) | ASGI server for running FastAPI applications |
| ![Pydantic](https://img.shields.io/badge/Pydantic-2.0+-E92063?style=for-the-badge&logo=pydantic&logoColor=white) | Data validation and settings management |

### 📊 Data Processing & Visualization

| 🔧 Tool | 📋 Description |
|---------|----------------|
| ![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white) | Data manipulation and analysis library |
| ![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white) | Numerical computing library |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C?style=for-the-badge) | Data visualization library |
| ![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-7DB0BC?style=for-the-badge) | Statistical data visualization |

### 🗃️ Data & Model Storage

| 🔧 Tool | 📋 Description |
|---------|----------------|
| ![HuggingFace](https://img.shields.io/badge/🤗_Datasets-2.12+-FFD21E?style=for-the-badge) | Dataset loading and processing library |
| ![Git LFS](https://img.shields.io/badge/Git_LFS-3.0+-F05032?style=for-the-badge&logo=git&logoColor=white) | Large file storage for model files |
| ![Safetensors](https://img.shields.io/badge/Safetensors-0.4+-FF6F00?style=for-the-badge) | Safe and efficient model weight storage format |

### 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    🏰 BURNOUT ADVISOR SYSTEM                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   📝 User   │───▶│  🚀 FastAPI │───▶│  🧠 BERT    │         │
│  │   Input     │    │   Backend   │    │   Model     │         │
│  └─────────────┘    └──────┬──────┘    └──────┬──────┘         │
│                            │                  │                 │
│                            ▼                  ▼                 │
│                     ┌─────────────┐    ┌─────────────┐         │
│                     │  💡 Rule    │◀───│ 🔮 Predict  │         │
│                     │   Advisor   │    │   Module    │         │
│                     └──────┬──────┘    └─────────────┘         │
│                            │                                    │
│                            ▼                                    │
│                     ┌─────────────┐                            │
│                     │  📋 JSON    │                            │
│                     │  Response   │                            │
│                     └─────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- 🧠 **BERT-Based Classification** - Fine-tuned transformer model for accurate burnout detection
- 📊 **Three-Level Classification** - Categorizes states as Healthy, Stressed, or Burnout
- 💡 **Rule-Based Advisor** - Provides personalized recommendations based on classification
- 🚀 **FastAPI Backend** - Fast, modern RESTful API implementation
- 💬 **Interactive Chat** - Command-line interface for direct interaction
- 📈 **Confidence Scores** - Displays probability distribution across all classifications
- 🆘 **Emergency Resources** - Includes mental health resources for critical cases
- 📝 **Batch Processing** - Analyze multiple journal entries in a single request

---

## 🏰 Project Structure

```
Burnout_Advisor_Project/
│
├── 📂 dataset/                  # Training Data
│   ├── student_journals.csv     # Journal entries dataset
│   └── split_data.py            # Data splitting utility
│
├── 📂 models/                   # Trained Models
│   ├── best_burnout_model/      # Fine-tuned BERT model
│   └── tokenizer_config/        # Tokenizer configuration files
│
├── 📂 backend/                  # API Server
│   ├── main.py                  # FastAPI application entry point
│   ├── predict.py               # Prediction module
│   └── advisor.py               # Rule-based advisory engine
│
├── 📂 training/                 # Model Training
│   └── train_bert.py            # BERT fine-tuning script
│
├── chat.py                      # 💬 Interactive Chat Interface
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 📦 Installation

### ⚠️ Prerequisites

Before installation, ensure you have the following:

- 🐍 **Python 3.8 or higher**
- 📦 **pip** - Python package manager
- 💻 **Terminal/Command Prompt**
- 📁 **Git LFS** - For downloading large model files (recommended)

### Step 1: Clone the Repository

If you want to use the pre-trained model, ensure Git LFS is installed first:

```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone the repository (LFS files will be downloaded automatically)
git clone https://github.com/jush-ua/CSST101-102-Final-Project.git
cd CSST101-102-Final-Project
```

*If you already have the repository but are missing model files:*
```bash
git lfs pull
```

### Step 2: Navigate to the Project Directory

```bash
cd "path/to/Burnout_Advisor_Project"
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

*The following libraries will be installed:*

| 📚 Library | 🎯 Purpose |
|------------|-----------|
| `torch` | Deep learning framework |
| `transformers` | BERT and transformer models |
| `datasets` | Data loading utilities |
| `pandas` | Data manipulation |
| `scikit-learn` | Machine learning utilities |
| `fastapi` | Web API framework |
| `uvicorn` | ASGI server |
| `seaborn` | Data visualization |

*Note: Installation may take several minutes depending on your internet connection.*

### 🔧 Common Installation Issues

<details>
<summary><b>❓ Model file is only a few KB instead of ~440MB</b></summary>

This means Git LFS didn't download the actual model. Run:
```bash
git lfs install
git lfs pull
```
</details>

<details>
<summary><b>❓ "Git LFS not found" error</b></summary>

Install Git LFS first:
- **Windows:** `winget install GitHub.GitLFS` or download from [git-lfs.com](https://git-lfs.com)
- **Mac:** `brew install git-lfs`
- **Linux:** `sudo apt install git-lfs`

Then run: `git lfs install`
</details>

<details>
<summary><b>❓ CUDA/GPU errors on Windows</b></summary>

If you don't have a GPU, PyTorch will use CPU automatically. If you have GPU issues:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
</details>

<details>
<summary><b>❓ Permission denied errors</b></summary>

Try running your terminal as administrator, or use:
```bash
pip install --user -r requirements.txt
```
</details>

---

## 🧠 Pre-Trained Model

*This repository includes a pre-trained BERT model stored via Git LFS.*

### 📊 Model Performance

The model was trained on **1,000 synthetic journal entries** and achieved the following metrics:

| 📈 Metric | 🏆 Score |
|-----------|----------|
| **Accuracy** | 100.00% |
| **Precision** | 100.00% |
| **Recall** | 100.00% |
| **F1 Score** | 100.00% |

### 🗂️ Model Files (Stored via Git LFS)

| 📁 File | 📏 Size | 📋 Description |
|---------|---------|----------------|
| `model.safetensors` | ~440 MB | Trained BERT model weights |
| `training_args.bin` | ~5 KB | Training configuration |

*If you prefer to train your own model, see [Model Training](#-model-training) below.*

---

## 🗡️ Getting Started

### Step 1: Train the Model (Optional)

If you want to train the model yourself instead of using the pre-trained version:

```bash
python training/train_bert.py
```

*This process will:*
- 📜 Load and prepare the dataset
- 🏋️ Fine-tune the BERT model
- 💾 Save the best model to `models/best_burnout_model/`
- 📊 Generate training metrics and confusion matrix

**Expected Output:**
```
🎓 ACADEMIC BURNOUT DETECTION - BERT FINE-TUNING
============================================================
⏰ Started at: 2025-12-27 14:16:19
💻 Device: cuda (or cpu)
🤖 Base Model: bert-base-uncased
...
📈 EVALUATION RESULTS:
   Accuracy:  1.0000
   Precision: 1.0000
   Recall:    1.0000
   F1 Score:  1.0000
```

---

### Step 2: Start the API Server

Launch the FastAPI server:

```bash
cd backend
python main.py
```

Or run directly with uvicorn:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Available Endpoints:**

| 🚪 Endpoint | 🔗 URL | 📋 Description |
|-----------|--------|----------------|
| 🏠 API Root | http://localhost:8000 | Welcome message |
| 📚 Swagger Docs | http://localhost:8000/docs | Interactive API documentation |
| 📖 ReDoc | http://localhost:8000/redoc | Alternative API documentation |

---

### Step 3: Test the API

**Using cURL:**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "I have been feeling overwhelmed with assignments and cannot sleep properly."}'
```

**Using Python:**

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

## 🗺️ Quick Start Guide

### 🚀 One-Click Startup (Recommended)

The easiest way to run the project is using the provided startup scripts:

**Windows:**
```bash
# Double-click run.bat or run from terminal:
.\run.bat
```

**Linux/Mac:**
```bash
# Make the script executable (first time only)
chmod +x run.sh

# Run the startup script
./run.sh
```

The script will:
- ✅ Check Python installation
- ✅ Install dependencies if needed
- ✅ Start the API server automatically

### 📝 Manual Setup

If you prefer manual setup:

```bash
# Step 1: Navigate to the project
cd Burnout_Advisor_Project

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Train the model (optional - pre-trained model included)
python training/train_bert.py

# Step 4: Start the server
cd backend
python main.py

# Step 5: Open http://localhost:8000/docs in your browser
```

### 🌐 Accessing the Application

Once the server is running:

| Interface | URL | Description |
|-----------|-----|-------------|
| 🎨 **Web Frontend** | Open `frontend/index.html` in browser | Beautiful medieval-themed chat interface |
| 📚 **API Docs** | http://localhost:8000/docs | Interactive Swagger documentation |
| 💻 **CLI Chat** | `python chat.py` (in new terminal) | Command-line chat interface |

---

## 💬 Interactive Chat Interface

*A command-line interface for direct interaction with the system.*

### Starting the Chat

First, ensure the API server is running in a separate terminal:

```bash
# Terminal 1: Start the server
cd Burnout_Advisor_Project
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then, launch the chat interface in another terminal:

```bash
# Terminal 2: Start the chat
cd Burnout_Advisor_Project
python chat.py
```

### Available Commands

| ⌨️ Command | 📋 Description |
|------------|----------------|
| *Type your entry* | Submit a journal entry for analysis |
| `help` | Display help information |
| `clear` | Clear the terminal screen |
| `quit` / `exit` | Exit the chat interface |

### Example Conversation

```
📝 Share your thoughts:
> I've been feeling overwhelmed with assignments and can't seem to catch up

🔮 Analyzing your entry...

══════════════════════════════════════════════════════════════════════
🔮 ANALYSIS RESULT:
══════════════════════════════════════════════════════════════════════

🟡 Mental State: **STRESSED**
📊 Confidence: 58.2%
⚠️  Risk Level: 🟡 ELEVATED - Some stress indicators present

📈 Probability Distribution:
   Healthy    [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 7.7%
   Stressed   [█████████████████░░░░░░░░░░░░░] 58.2%
   Burnout    [██████████░░░░░░░░░░░░░░░░░░░░] 34.0%

💬 Your entry shows signs of academic stress. This is common and manageable.

💡 TOP RECOMMENDATIONS:
   ⚡ Immediate Action: Address Current Stressors
   😴 Rest & Recovery: Prioritize Sleep and Rest
   📅 Time Management: Reorganize Your Schedule

💡 Tip: Step outside for 5 minutes. Fresh air can help reset your mindset.
══════════════════════════════════════════════════════════════════════
```

---

## 📚 API Documentation

### Available Endpoints

| ⚔️ Method | 🚪 Endpoint | 📜 Description |
|-----------|-------------|----------------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check endpoint |
| POST | `/predict` | Get burnout prediction |
| POST | `/predict/batch` | Batch predictions (max 10 entries) |
| POST | `/advice` | Get personalized recommendations |
| POST | `/analyze` | Full analysis (prediction + advice) |
| GET | `/labels` | Get classification labels |
| GET | `/resources` | Get mental health resources |

### Example Request/Response

**POST /analyze** *(Full Analysis)*

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
    "summary": "Your entry shows signs of academic stress. This is common and manageable.",
    "recommendations": [...],
    "follow_up": "We recommend journaling daily this week."
  }
}
```

---

## 🏋️ Model Training

### Dataset Format

The training dataset (`dataset/student_journals.csv`) must contain two columns:

| 📝 text | 🏷️ label |
|---------|----------|
| "I'm feeling great today!" | 0 |
| "Too much homework, feeling stressed" | 1 |
| "I can't take this anymore" | 2 |

### Training Hyperparameters

| 🔧 Parameter | 📊 Value |
|--------------|----------|
| Base Model | bert-base-uncased |
| Max Length | 128 tokens |
| Batch Size | 8 |
| Learning Rate | 2e-5 |
| Epochs | 10 |
| Early Stopping | 3 epochs patience |

### Training Tips

- 📚 **More Data**: Collect additional journal entries for improved accuracy
- ⚡ **GPU**: Training is significantly faster with a CUDA-enabled GPU
- 🎛️ **Hyperparameters**: Adjust learning rate and batch size based on your dataset
- ⚖️ **Class Balance**: Ensure similar sample quantities for each classification

---

## 🏷️ Classification Categories

*The system classifies entries into one of three categories:*

| 🔢 Label ID | 📛 Name | 📋 Description | 🔍 Indicators |
|-------------|---------|----------------|---------------|
| 0 | **🟢 Healthy** | Stable mental state | Positive outlook, work-life balance, adequate sleep |
| 1 | **🟡 Stressed** | Elevated but manageable stress | Worry, academic pressure, sleep disturbances |
| 2 | **🔴 Burnout** | Severe exhaustion | Complete exhaustion, hopelessness, physical symptoms |

### Risk Levels

- 🟢 **LOW RISK**: Healthy mental state, continue current practices
- 🟡 **MODERATE/ELEVATED**: Stress indicators present, preventive action recommended
- 🔴 **HIGH/CRITICAL**: Burnout detected, professional support strongly recommended

---

## 🤝 Contributing

*Contributions are welcome! Here are some ways you can help:*

1. 📚 **Add Training Data**: Contribute diverse journal entries to improve model accuracy
2. 💡 **Improve Recommendations**: Enhance the rule-based advisory system
3. 🎨 **Build Frontend**: Develop a web-based user interface
4. 🌍 **Add Languages**: Implement multilingual burnout detection
5. 📱 **Mobile App**: Create iOS/Android applications

---

## 🏰 Git & Git LFS Guide for Contributors

*Follow these steps to properly contribute to this project.*

### Prerequisites for Contributors

| 🔧 Tool | 📥 Installation | 📋 Purpose |
|---------|----------------|------------|
| **Git** | [git-scm.com](https://git-scm.com/downloads) | Version control |
| **Git LFS** | [git-lfs.github.com](https://git-lfs.github.com/) | Large file storage |

### Step 1: Install Git LFS

```bash
# Windows (via Git Bash or PowerShell)
git lfs install

# macOS (via Homebrew)
brew install git-lfs
git lfs install

# Ubuntu/Debian
sudo apt-get install git-lfs
git lfs install
```

*Verify installation:*
```bash
git lfs version
# Expected output: git-lfs/3.x.x (...)
```

### Step 2: Clone the Repository

```bash
# Clone with automatic LFS file download
git clone https://github.com/jush-ua/CSST101-102-Final-Project.git

# Navigate to the project directory
cd CSST101-102-Final-Project
```

### Step 3: Pulling Latest Changes

*If you already have the repository cloned:*

```bash
# Navigate to the project directory
cd CSST101-102-Final-Project

# Pull latest changes
git pull origin main

# Download any updated LFS files
git lfs pull
```

*If you have local changes that conflict:*
```bash
# Stash your changes temporarily
git stash

# Pull latest updates
git pull origin main
git lfs pull

# Restore your changes
git stash pop
```

### Step 4: Downloading LFS Files

*If model files appear as text pointers instead of actual files:*
```bash
# Download all LFS files
git lfs pull

# Or download specific files
git lfs pull --include="models/**"
```

*Verify LFS files are properly downloaded:*
```bash
# Check LFS file status
git lfs ls-files

# Verify file sizes (model.safetensors should be ~440 MB)
dir models\best_burnout_model\  # Windows
ls -la models/best_burnout_model/  # macOS/Linux
```

### Step 5: Making Contributions

```bash
# 1. Create a new branch
git checkout -b feature/your-feature-name

# 2. Make your changes
# ... edit files ...

# 3. Stage your changes
git add .

# 4. Commit with a descriptive message
git commit -m "✨ Add: Brief description of changes"

# 5. Push to GitHub
git push origin feature/your-feature-name
```

### Step 6: Working with Large Files

*Git LFS automatically tracks these file types:*

| 📁 Extension | 📋 File Type |
|--------------|--------------|
| `*.safetensors` | Model weights (SafeTensors format) |
| `*.bin` | PyTorch model binaries |
| `*.pt` | PyTorch tensors |
| `*.pth` | PyTorch checkpoints |

*To track additional large file types:*
```bash
# Track a new file type
git lfs track "*.csv"

# Track a specific file
git lfs track "path/to/large_file.zip"

# Commit the updated .gitattributes
git add .gitattributes
git commit -m "📦 Track: Add new file type to LFS"
```

### Step 7: Submitting a Pull Request

1. 🌐 Go to the [GitHub Repository](https://github.com/jush-ua/CSST101-102-Final-Project)
2. 🔔 Click **"Compare & pull request"**
3. 📝 Describe your changes:
   - What changes were made
   - Why the changes were necessary
   - Testing performed
4. ✅ Submit for review

### Commit Message Convention

| 🏷️ Prefix | 📋 Usage |
|-----------|----------|
| `✨ Add:` | New features or files |
| `🔧 Fix:` | Bug fixes |
| `📝 Docs:` | Documentation updates |
| `🎨 Style:` | Code formatting (no logic changes) |
| `♻️ Refactor:` | Code restructuring |
| `🧪 Test:` | Adding or updating tests |
| `📦 Track:` | Git LFS tracking changes |
| `🚀 Deploy:` | Deployment-related changes |

*Example commit messages:*
```bash
git commit -m "✨ Add: New stress-related phrases to training data"
git commit -m "🔧 Fix: Resolve encoding issue in journal parser"
git commit -m "📝 Docs: Update README with API examples"
```

### Troubleshooting

**Issue: "Encountered X file(s) that should have been pointers"**
```bash
git lfs migrate import --include="*.safetensors,*.bin,*.pt,*.pth" --everything
git push --force-with-lease
```

**Issue: "Smudge error" when pulling**
```bash
git lfs fetch --all
git lfs checkout
```

**Issue: Push rejected due to file size**
```bash
git lfs track "path/to/large-file.ext"
git add .gitattributes
git add path/to/large-file.ext
git commit -m "📦 Track: Add large file to LFS"
git push
```

---

## ⚠️ Important Disclaimer

This system is designed as an **educational tool** and **early warning system**. It is **NOT** a replacement for professional mental health services.

If you or someone you know is experiencing severe burnout or a mental health crisis:
- 🏛️ Contact your campus counseling center
- 📞 Call the **988 Suicide & Crisis Lifeline**
- 💬 Text **HOME** to **741741** (Crisis Text Line)
- 🆘 Seek professional help immediately

*Remember: Seeking help is a sign of strength, not weakness.*

---

## 📜 License

This project is for educational purposes within an academic setting.

---

## 👥 Authors

**CSST 101/102** - Artificial Intelligence Final Project  
**Laguna State Polytechnic University (LSPU)**  
3rd Year, 1st Semester

---

## 🏆 Contributors

### Project Lead & Developer

| 👤 Name | 🎭 Role | 📋 Contributions |
|---------|---------|------------------|
| **Urrea** | Project Lead & Backend Developer | System architecture, BERT model training, FastAPI backend, rule-based advisory engine |

### Team Members

| 👤 Name | 🎭 Role | 📋 Contributions |
|---------|---------|------------------|
| **Urrea** | Backend Developer | Server architecture, API endpoints, model integration |
| **Bauyon** | Frontend Developer | User interface design |
| **Pagalanan** | Documentation | Project documentation and guides |

---

## 🙏 Acknowledgments

*"Your mental health is a priority. Take care of yourself."*

---

*Built with care for student mental health and wellness.*

