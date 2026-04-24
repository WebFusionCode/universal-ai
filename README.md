# 🤖 Universal AI Platform

A comprehensive, production-ready AutoML platform that automatically trains, evaluates, and explains machine learning models across **image classification**, **tabular data**, and **time-series forecasting**. Built with FastAPI and React for seamless model training, prediction, and interpretation.

---

## ✨ Features

### 🎯 **Multi-Domain ML Support**
- **Image Classification**: CNN, MobileNet, ResNet18, EfficientNet, ViT, UNet
- **Tabular ML**: Automated hyperparameter tuning with Optuna, classification & regression
- **Time-Series Forecasting**: ARIMA, Prophet, LSTM models with temporal feature engineering
- **Audio & Video AI**: Extensible framework for multimodal AI tasks

### 🚀 **Advanced AutoML Capabilities**
- **Intelligent Problem Detection**: Automatically detects classification vs. regression vs. time-series
- **Dynamic Hyperparameter Tuning**: Optuna-based optimization with time constraints
- **Parallel Model Training**: joblib-based parallelism for multi-model ensemble training
- **Live Training Progress**: WebSocket-enabled real-time training updates and logs
- **Robust Data Preprocessing**: Smart feature engineering, datetime handling, missing value imputation

### 📊 **Model Explainability**
- **SHAP Analysis**: Feature importance and model interpretation
- **Training Metrics**: Comprehensive metrics (accuracy, F1, AUC, RMSE, MAE, etc.)
- **Experiment Tracking**: Full experiment history with metadata and model artifacts
- **Leaderboard**: Automatic model ranking and performance comparison

### 🛡️ **Production-Grade Robustness**
- **Safe Dependency Handling**: Graceful fallbacks for optional dependencies
- **Comprehensive Error Handling**: Detailed error messages and recovery mechanisms
- **Data Validation**: Smart CSV parsing, schema validation, edge case handling
- **Model Persistence**: joblib-based model serialization with versioning

---

## 📋 Tech Stack

### Backend
- **Framework**: FastAPI (async HTTP + WebSocket)
- **ML Libraries**: scikit-learn, XGBoost, CatBoost, Optuna, Prophet, statsmodels
- **Computer Vision**: PyTorch, timm, torchvision
- **Data Processing**: pandas, numpy
- **Model Management**: joblib, pickle
- **Monitoring**: MongoDB (experiments tracking)

### Frontend
- **UI Framework**: React 18+
- **Styling**: Tailwind CSS, custom components
- **HTTP Client**: Axios
- **UI Components**: Shadcn UI (accordion, dialog, tabs, etc.)
- **State Management**: React hooks, context API

### Infrastructure
- **Server**: Uvicorn (ASGI)
- **Build**: npm, craco
- **Database**: MongoDB (optional for experiments)

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- Node.js 16+
- npm or yarn

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Build
npm run build
```

---

## 🚀 Quick Start

### Start Backend Server

```bash
cd backend
python main.py
```

The backend API will be available at `http://localhost:8000`

**API Endpoints:**
- `POST /train` - Train a new model
- `POST /predict` - Make predictions
- `GET /models` - List trained models
- `WebSocket /ws/progress` - Live training progress

### Start Frontend Server

```bash
cd frontend
npm start
```

The frontend will be available at `http://localhost:3000`

---

## 📚 Usage Examples

### Training a Model

**Tabular Data (Classification)**
```bash
curl -X POST http://localhost:8000/train \
  -F "file=@data.csv" \
  -F "target_column=category" \
  -F "dataset_type=tabular" \
  -F "model_name=auto"
```

**Image Classification**
```bash
curl -X POST http://localhost:8000/train \
  -F "file=@images.zip" \
  -F "model_name=auto"
```

### Making Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_data.csv" \
  -F "target_column=category"
```

### API Response Format

```json
{
  "status": "completed",
  "best_model": "XGBClassifier",
  "score": 0.95,
  "loss": 0.05,
  "problem_type": "classification",
  "dataset_rows": 1000,
  "train_time": 45.2,
  "leaderboard": [
    {"model": "XGBClassifier", "score": 0.95},
    {"model": "RandomForestClassifier", "score": 0.93}
  ]
}
```

---

## 📁 Project Structure

```
universal-ai/
├── backend/
│   ├── main.py                 # FastAPI application & endpoints
│   ├── requirements.txt         # Python dependencies
│   ├── database.py             # MongoDB integration
│   ├── models/                 # Trained model storage
│   ├── experiments/            # Experiment tracking
│   ├── explain/                # Model explanations (SHAP, metrics)
│   ├── uploads/                # User-uploaded files
│   └── generated_pipeline.py   # Auto-generated ML pipelines
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── pages/              # Route pages (Train, Predict, etc.)
│   │   ├── components/         # React components
│   │   ├── services/           # API service layer
│   │   ├── hooks/              # Custom React hooks
│   │   ├── lib/                # Utilities (API, utils)
│   │   └── ui/                 # Shadcn UI components
│   ├── package.json
│   └── tailwind.config.js
│
└── README.md                   # This file
```

---

## 🔧 Configuration

### Backend Configuration (main.py)

```python
# WebSocket settings
WS_CLIENTS = set()

# Model storage paths
MODEL_FOLDER = "backend/models"
IMAGE_DATASET_FOLDER = "backend/uploads/images"
UPLOAD_FOLDER = "backend/uploads"

# Training constraints
MAX_TRAINING_TIME = 300  # seconds
CV_SPLITS = 5  # cross-validation folds
```

### Environment Variables

Create `.env` file in backend:
```
MONGO_URI=mongodb://localhost:27017
DATABASE_NAME=universal_ai
SECRET_KEY=your_secret_key
```

---

## 🎯 Key Features Explained

### Intelligent Problem Type Detection
The platform automatically detects:
- **Classification**: Categorical targets with <50 unique values
- **Regression**: Numeric continuous targets
- **Time-Series**: Temporal data with datetime columns

```python
detected_type, metadata = detect_dataset_type(df)
# Returns: ("classification" | "regression" | "timeseries" | "image")
```

### Optuna-Based Hyperparameter Tuning
Automatic search for optimal hyperparameters within time constraints:
- Dynamic trial allocation based on remaining time
- Pruning of unpromising trials
- Parallel execution with joblib

### Live Training Progress
WebSocket connection for real-time updates:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/progress');
ws.onmessage = (event) => {
  console.log('Training update:', event.data);
};
```

### Robust Data Preprocessing
- Automatic datetime feature engineering (year, month, day extraction)
- Smart missing value handling (forward-fill for time-series, mean imputation for tabular)
- Categorical encoding (pd.get_dummies for tabular, Label encoding for targets)
- Feature scaling (StandardScaler for image models)

---

## 📊 Model Explanations

### Feature Importance (SHAP)
```json
{
  "feature_importance": {
    "age": 0.25,
    "income": 0.18,
    "credit_score": 0.15
  },
  "top_features": ["age", "income", "credit_score"]
}
```

### Training Metrics
```json
{
  "accuracy": 0.95,
  "precision": 0.94,
  "recall": 0.96,
  "f1_score": 0.95,
  "roc_auc": 0.98,
  "training_time": 45.2
}
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Model Training Fails**
- Check CSV format and ensure target column exists
- Ensure dataset has at least 20 samples
- Check for missing dependencies: `pip install -r requirements.txt`

**2. WebSocket Connection Error**
- Ensure backend server is running
- Check firewall/proxy settings
- Verify CORS configuration in FastAPI

**3. GPU Not Detected (PyTorch)**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Install CUDA toolkit if needed.

**4. Optuna Import Error**
```bash
pip install optuna
```

---

## 🤝 Contributing

We welcome contributions! Follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 API Documentation

### Interactive API Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Core Endpoints

#### POST /train
Train a new model on uploaded data
- **Parameters**: file, target_column, dataset_type, model_name, params
- **Returns**: {status, best_model, score, loss, problem_type, dataset_rows, train_time, leaderboard}

#### POST /predict
Make predictions on new data
- **Parameters**: file, target_column
- **Returns**: {predictions, confidence, metadata}

#### GET /models
List all trained models
- **Returns**: List of model metadata

#### GET /explain/{model_id}
Get model explanations
- **Returns**: {feature_importance, shap_values, metrics}

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙋 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation at `/docs`
- Review API docs at `http://localhost:8000/docs`

---

## 🚀 Roadmap

- [ ] Distributed training with Ray
- [ ] Model versioning and registry
- [ ] A/B testing framework
- [ ] Real-time model monitoring
- [ ] Support for NLP tasks
- [ ] Enhanced model interpretability tools
- [ ] Production deployment guides (Docker, Kubernetes)

---

## 🎉 Acknowledgments

Built with ❤️ using:
- FastAPI for robust async APIs
- React for intuitive UX
- scikit-learn, PyTorch, and community ML libraries
- Optuna for intelligent hyperparameter optimization

**Start building powerful ML models today! 🚀**
