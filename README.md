# � Ye Olde Academic Burnout Prevention & Rule-Based Advisory System ⚔️

*Hear ye, hear ye! A most wondrous contraption of machine learning sorcery that doth analyze the journal entries of weary scholars to detect signs of academic burnout and bestow upon them personalized counsel for mental wellness!* 📜✨

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![BERT](https://img.shields.io/badge/Model-BERT-orange.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)

---

## 📜 Table of Contents

- [The Grand Overview](#-the-grand-overview)
- [The Arcane Tech Stack](#-the-arcane-tech-stack)
- [Enchanted Features](#-enchanted-features)
- [The Castle Structure](#-the-castle-structure)
- [Summoning the Dependencies](#-summoning-the-dependencies)
- [How to Embark Upon This Quest](#-how-to-embark-upon-this-quest)
- [Quick Start Guide](#-quick-start-guide-for-the-impatient-knight-)
- [The Interactive Oracle Chat](#-the-interactive-oracle-chat)
- [The Sacred API Scrolls](#-the-sacred-api-scrolls)
- [Training Thy Model](#-training-thy-model)
- [The Three Classifications](#-the-three-classifications)
- [Join the Fellowship](#-join-the-fellowship)
- [The Contributor's Guide to Git & Git LFS](#-the-contributors-guide-to-git--git-lfs)

---

## 🎯 The Grand Overview

Hark! Academic burnout doth plague many a scholar in these troubled times, marked by exhaustion of the soul, cynicism most foul, and diminished efficacy in one's studies. This mystical system employs the ancient arts of Natural Language Processing (NLP) to:

1. ⚔️ **Detect** - Uncover the level of burnout from student journal entries
2. 🏷️ **Classify** - Sort mental states into three sacred categories: Healthy, Stressed, or Burnout
3. 💡 **Provide** - Bestow personalized, rule-based counsel upon each weary soul

This grand apparatus combines a **fine-tuned BERT model** (a most learned oracle) for accurate text classification with a **rule-based advisory engine** that delivereth actionable wisdom unto thee!

---

## 🛠️ The Arcane Tech Stack

*Behold the mystical tools and enchantments that power this grand apparatus!* ⚗️✨

### 🐍 Core Language

| 🔧 Tool | 📋 Description |
|---------|----------------|
| ![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white) | The serpent tongue in which all spells are written |

### 🧠 Machine Learning & AI Sorcery

| 🔧 Tool | 📋 Description |
|---------|----------------|
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) | The deep learning forge where models are crafted 🔥 |
| ![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-FFD21E?style=for-the-badge) | The sacred library of BERT and other wise oracles 🤖 |
| ![BERT](https://img.shields.io/badge/BERT-base--uncased-orange?style=for-the-badge) | The all-knowing transformer, fine-tuned for burnout detection 📚 |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) | Utilities for metrics, splitting, and evaluation ⚙️ |

### 🚀 Backend & API Framework

| 🔧 Tool | 📋 Description |
|---------|----------------|
| ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white) | The swift falcon that carries REST messages 🦅 |
| ![Uvicorn](https://img.shields.io/badge/Uvicorn-0.22+-499848?style=for-the-badge) | The ASGI steed that gallops with lightning speed ⚡ |
| ![Pydantic](https://img.shields.io/badge/Pydantic-2.0+-E92063?style=for-the-badge&logo=pydantic&logoColor=white) | The guardian of data validation and schemas 🛡️ |

### 📊 Data Manipulation & Visualization

| 🔧 Tool | 📋 Description |
|---------|----------------|
| ![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white) | The data wrangler, master of tables and scrolls 🐼 |
| ![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white) | The mathematical foundation of all computations 🔢 |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C?style=for-the-badge) | The artist that paints charts and graphs 🎨 |
| ![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-7DB0BC?style=for-the-badge) | The beautifier of statistical visualizations 📈 |

### 🗃️ Data & Model Storage

| 🔧 Tool | 📋 Description |
|---------|----------------|
| ![HuggingFace](https://img.shields.io/badge/🤗_Datasets-2.12+-FFD21E?style=for-the-badge) | The data loading wizardry from HuggingFace 📦 |
| ![Git LFS](https://img.shields.io/badge/Git_LFS-3.0+-F05032?style=for-the-badge&logo=git&logoColor=white) | The vault keeper for large model files 🏰 |
| ![Safetensors](https://img.shields.io/badge/Safetensors-0.4+-FF6F00?style=for-the-badge) | The secure format for storing model weights 🔐 |

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

## ✨ Enchanted Features

- 🧠 **BERT-Based Classification** - A transformer most wise, fine-tuned for burnout detection
- 📊 **Three-Level Classification** - Healthy, Stressed, and Burnout states
- 💡 **Rule-Based Advisor** - Personalized recommendations bestowed upon each pilgrim
- 🚀 **FastAPI Backend** - A RESTful messenger swift as a falcon
- 💬 **Interactive Chat** - Converse with the Oracle in thy terminal!
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
├── chat.py                      # 💬 The Interactive Oracle Chat
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
- 📁 **Git LFS** - For fetching the large model files (optional but recommended)

### 🧙‍♂️ Step I: Clone the Repository with Git LFS

If thou wishest to use the pre-trained model, ensure Git LFS is installed:

```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone the repository (LFS files will be fetched automatically)
git clone https://github.com/ItSnOtNoOkIeBeAr/Academic-Burnout-Prevention-and-Rule-Based-Advisory-System-for-College-Students.git
cd Academic-Burnout-Prevention-and-Rule-Based-Advisory-System-for-College-Students
```

*If thou already hast the repository but lacketh the model files:*
```bash
git lfs pull
```

### 🧙‍♂️ Step II: Navigate to the Sacred Project Directory

Open thy terminal (the PowerShell or Command Prompt) and venture forth:

```bash
cd "path\to\Burnout_Advisor_Project"
```

### 🔮 Step III: Summon the Magical Dependencies

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

## 🧠 About the Pre-Trained Model

*This repository includes a pre-trained BERT model stored via Git LFS!* 📦

### 📊 Model Performance

The Oracle hath been trained upon **1000 synthetic journal entries** and achieved these most glorious metrics:

| 📈 Metric | 🏆 Score |
|-----------|----------|
| **Accuracy** | 100.00% ✨ |
| **Precision** | 100.00% ✨ |
| **Recall** | 100.00% ✨ |
| **F1 Score** | 100.00% ✨ |

*A perfect score! The Oracle hath achieved enlightenment!* 🧙‍♂️🔮

### 🗂️ Model Files (Stored via Git LFS)

| 📁 File | 📏 Size | 📋 Description |
|---------|---------|----------------|
| `model.safetensors` | ~440 MB | The trained BERT weights |
| `training_args.bin` | ~5 KB | Training configuration |

*If thou preferest to train thy own model, see [Training Thy Model](#-training-thy-model) below!*

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

*Rejoice! Thy model hath achieved perfection!* 🎉✨

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

## � The Interactive Oracle Chat

*For those who wish to converse directly with the Oracle!* 🔮

### 🗣️ Starting the Chat Interface

First, ensure the API server is running in a separate terminal:

```bash
# Terminal 1: Start the server
cd Burnout_Advisor_Project
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then, in another terminal, launch the interactive chat:

```bash
# Terminal 2: Start the chat
cd Burnout_Advisor_Project
python chat.py
```

### 🎮 Chat Commands

| ⌨️ Command | 📋 Description |
|------------|----------------|
| *Type thy feelings* | Share thy thoughts and receive wisdom |
| `help` | Display guidance for the weary |
| `clear` | Clear the terminal screen |
| `quit` / `exit` | Depart from the Oracle's presence |

### 📸 Example Conversation

```
📝 Share thy thoughts, noble scholar:
> I've been feeling overwhelmed with assignments and can't seem to catch up

🔮 The Oracle is divining thy mental state...

══════════════════════════════════════════════════════════════════════
🔮 THE ORACLE SPEAKS:
══════════════════════════════════════════════════════════════════════

🟡 Thy Mental State: **STRESSED**
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

💡 Tip: Step outside for 5 minutes. Fresh air can reset your mind.
══════════════════════════════════════════════════════════════════════
```

---

## �📚 The Sacred API Scrolls

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

## 🏰 The Contributor's Guide to Git & Git LFS

*Hark, noble contributor! Follow these sacred scrolls to properly contribute to this grand apparatus!* 📜⚔️

### 📋 Prerequisites for Contributors

Before thy contribution, ensure thou hast installed:

| 🔧 Tool | 📥 Installation | 📋 Purpose |
|---------|----------------|------------|
| **Git** | [git-scm.com](https://git-scm.com/downloads) | Version control sorcery |
| **Git LFS** | [git-lfs.github.com](https://git-lfs.github.com/) | Large file storage vault |

### 🚀 Step I: Installing Git LFS

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

*Verify thy installation:*
```bash
git lfs version
# Should display: git-lfs/3.x.x (...)
```

### 📦 Step II: Clone the Repository

```bash
# Clone with LFS files automatically
git clone https://github.com/ItSnOtNoOkIeBeAr/Academic-Burnout-Prevention-and-Rule-Based-Advisory-System-for-College-Students.git

# Navigate into the castle
cd Academic-Burnout-Prevention-and-Rule-Based-Advisory-System-for-College-Students
```

### 🔄 Step III: For Existing Users - Pulling Latest Changes

*If thou already hast the repository cloned and wish to receive the latest updates:*

```bash
# Navigate to thy project directory
cd Academic-Burnout-Prevention-and-Rule-Based-Advisory-System-for-College-Students

# Fetch and pull the latest changes from the main branch
git pull origin main

# If LFS files were updated, ensure they are downloaded
git lfs pull
```

*If thou hast made local changes that conflict:*
```bash
# Stash thy changes temporarily
git stash

# Pull the latest updates
git pull origin main
git lfs pull

# Restore thy changes
git stash pop
```

*To update thy local branch with the latest from main:*
```bash
# Switch to thy feature branch
git checkout feature/thy-branch-name

# Merge latest changes from main
git fetch origin
git merge origin/main

# Pull any new LFS files
git lfs pull
```

### 🔍 Step IV: Pulling LFS Files

*If thou hast cloned but the model files appear as pointers:*
```bash
# Fetch all LFS files
git lfs pull

# Or fetch specific files
git lfs pull --include="models/**"
```

*Verify LFS files are downloaded correctly:*
```bash
# Check LFS file status
git lfs ls-files

# Check if files are actual content (not pointers)
# The model.safetensors should be ~440 MB, not a few bytes
dir models\best_burnout_model\  # Windows
ls -la models/best_burnout_model/  # macOS/Linux
```

### ⚔️ Step V: Making Thy Contribution

```bash
# 1. Create a new branch for thy quest
git checkout -b feature/thy-noble-contribution

# 2. Make thy changes to the code
# ... edit files ...

# 3. Stage thy changes
git add .

# 4. Commit with a descriptive message
git commit -m "✨ Add: Brief description of thy noble deed"

# 5. Push to GitHub
git push origin feature/thy-noble-contribution
```

### 🏰 Step V: Pushing Large Files with Git LFS

*The vault (Git LFS) tracketh these file types automatically:*

| 📁 Extension | 📋 File Type |
|--------------|--------------|
| `*.safetensors` | Model weights (SafeTensors format) |
| `*.bin` | PyTorch model binaries |
| `*.pt` | PyTorch tensors |
| `*.pth` | PyTorch checkpoints |

*If thou needest to track additional large files:*
```bash
# Track a new file type (e.g., large CSV files)
git lfs track "*.csv"

# Or track a specific file
git lfs track "path/to/large_file.zip"

# This updates .gitattributes - commit it!
git add .gitattributes
git commit -m "📦 Track: Add new file type to LFS"
```

*Verify what LFS is tracking:*
```bash
git lfs ls-files
```

### 🚀 Step VI: Pushing to GitHub

```bash
# Push thy branch (LFS files are handled automatically)
git push origin feature/thy-noble-contribution

# If pushing large files for the first time, thou may need:
git lfs push origin feature/thy-noble-contribution --all
```

### 🔮 Step VII: Creating a Pull Request

1. 🌐 Go to the [GitHub Repository](https://github.com/ItSnOtNoOkIeBeAr/Academic-Burnout-Prevention-and-Rule-Based-Advisory-System-for-College-Students)
2. 🔔 Click **"Compare & pull request"** for thy branch
3. 📝 Fill out the PR template with:
   - What changes thou hast made
   - Why these changes benefit the realm
   - Any testing thou hast performed
4. ✅ Submit and await review from the Council!

### 📜 Git Commit Message Convention

*Follow this sacred format for commit messages:*

| 🏷️ Prefix | 📋 Usage |
|-----------|----------|
| `✨ Add:` | New features or files |
| `🔧 Fix:` | Bug fixes |
| `📝 Docs:` | Documentation updates |
| `🎨 Style:` | Code formatting (no logic change) |
| `♻️ Refactor:` | Code restructuring |
| `🧪 Test:` | Adding or updating tests |
| `📦 Track:` | Git LFS tracking changes |
| `🚀 Deploy:` | Deployment related changes |

*Example commit messages:*
```bash
git commit -m "✨ Add: New stress-related phrases to training data"
git commit -m "🔧 Fix: Resolve encoding issue in journal parser"
git commit -m "📝 Docs: Update README with API examples"
```

### ⚠️ Common Issues & Solutions

**Issue: "Encountered X file(s) that should have been pointers"**
```bash
# Fix LFS pointer issues
git lfs migrate import --include="*.safetensors,*.bin,*.pt,*.pth" --everything
git push --force-with-lease
```

**Issue: "Smudge error" when pulling**
```bash
# Clear LFS cache and re-pull
git lfs fetch --all
git lfs checkout
```

**Issue: Push rejected due to file size**
```bash
# Ensure LFS is tracking the file
git lfs track "path/to/large-file.ext"
git add .gitattributes
git add path/to/large-file.ext
git commit -m "📦 Track: Add large file to LFS"
git push
```

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

## 🏆 The Guild of Creators

*Hark! These noble souls hath contributed their talents to forge this grand apparatus:*

### ⚔️ Project Architect & Lead Developer

| 👤 Name | 🎭 Role | 📋 Contributions |
|---------|---------|------------------|
| **Urrea** | 🏰 Project Creator & Backend Developer | Crafted the entire system from the ground up, designed the architecture, implemented the BERT model training, FastAPI backend, and rule-based advisory engine |

### 🛡️ The Fellowship

| 👤 Name | 🎭 Role | 📋 Contributions |
|---------|---------|------------------|
| **Urrea** | ⚙️ Backend Developer | Server architecture, API endpoints, model integration, and the Oracle's wisdom |
| **Bauyon** | 🎨 Frontend Developer | User interface and experience design |
| **Pagalanan** | 📜 Documentation | Scrolls, guides, and sacred texts |

---

### 🎖️ Special Recognition

*This project was conceived, designed, and crafted by* ***Urrea*** *— the mastermind behind the Oracle's creation.* 🧙‍♂️✨

*From the initial spark of inspiration to the final incantation, Urrea hath poured countless hours of dedication into bringing this burnout detection system to life. Truly, a knight of code most valiant!* ⚔️🏰

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

