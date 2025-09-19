# Idealize Datathon: Ultimate Ensemble Model for Survival Prediction

## 🏆 Project Overview

This repository contains a comprehensive machine learning solution for the **Idealize Datathon** competition, focused on predicting patient survival status with maximum F1 score performance.

## 📊 Competition Goal

Predict patient survival status using advanced ensemble modeling techniques to achieve the highest possible F1 score.

## 🚀 Solution Approach

Our solution combines multiple state-of-the-art techniques:

### 🔧 Comprehensive Feature Engineering
- **Time-based features**: Treatment duration, time to treatment, diagnosis patterns
- **Health indices**: BMI, comorbidity scores, health risk factors
- **Interaction features**: Cross-feature relationships and polynomial terms
- **Aggregation features**: Group-based statistics and derived metrics
- **134+ engineered features** in total

### 🤖 Advanced Ensemble Strategy
- **Base Models**: LightGBM, XGBoost, and CatBoost
- **Stacking Architecture**: Level-1 meta-learner for intelligent blending
- **Cross-Validation**: 10-fold stratified approach for robust validation
- **Class Balancing**: Optimized weights for imbalanced data

### ⚖️ F1 Score Optimization
- **Dynamic Threshold Tuning**: Systematic search for optimal classification threshold
- **Out-of-Fold Predictions**: Unbiased performance estimation
- **Individual vs Ensemble**: Comprehensive model comparison

## 📁 Repository Structure

```
Idealize-Datathon/
├── idealize_datathon_analysis.ipynb    # Main analysis notebook
├── 041435.py                           # Original Python script
├── train.csv                          # Training dataset
├── test.csv                           # Test dataset  
├── sample_submission.csv              # Submission format
├── .gitignore                         # Git exclusions
└── README.md                          # This file
```

## 🛠️ Setup and Installation

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
```

### Required Libraries
```bash
pip install numpy pandas scikit-learn
pip install lightgbm xgboost catboost
pip install matplotlib seaborn tqdm
pip install joblib
```

### For GPU Acceleration (Optional)
```bash
# For LightGBM GPU support
pip install lightgbm[gpu]

# For XGBoost GPU support  
pip install xgboost[gpu]

# For CatBoost GPU support
pip install catboost[gpu]
```

## 🚀 Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/mehara-rothila/Idealize-Datathon.git
cd Idealize-Datathon
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt  # If available
# Or install packages individually as listed above
```

3. **Run the analysis:**
```bash
jupyter notebook idealize_datathon_analysis.ipynb
```

4. **Execute all cells** to reproduce the complete pipeline

## 📈 Results

### Model Performance
- **🔥 LightGBM**: Individual F1 performance
- **⚡ XGBoost**: Individual F1 performance  
- **🐱 CatBoost**: Individual F1 performance
- **🔗 Stacked Ensemble**: Superior combined performance

### Generated Files
- `submission.csv`: Final competition predictions
- `final_lgbm_model.pkl`: Saved model for compliance

## 🧪 Methodology

### 1. Data Preprocessing
- Missing value handling
- Date feature extraction
- Memory optimization

### 2. Feature Engineering
- Health risk indicators
- Treatment pattern analysis
- Patient demographic interactions
- Polynomial feature combinations

### 3. Model Training
- Stratified cross-validation
- Early stopping mechanisms
- Hyperparameter optimization
- Class weight balancing

### 4. Ensemble Strategy
- Level-0: Base model predictions
- Level-1: Meta-learner stacking
- Optimal threshold selection

### 5. Evaluation
- F1 score optimization
- Cross-validation consistency
- Model interpretability analysis

## 📊 Key Features

- **Robust Pipeline**: End-to-end automated workflow
- **Scalable Architecture**: Handles large datasets efficiently
- **Model Interpretability**: Feature importance analysis
- **Reproducible Results**: Fixed random seeds and versioning
- **Competition Ready**: Direct submission file generation

## 🤝 Contributing

This is a competition entry repository. For suggestions or improvements:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed explanation

## 📜 License

This project is for educational and competition purposes. Please respect the competition rules and data usage guidelines.

## 🏆 Competition Details

- **Event**: Idealize Datathon
- **Task**: Binary Classification (Survival Prediction)
- **Metric**: F1 Score
- **Approach**: Ensemble Learning with Advanced Feature Engineering

## 📞 Contact

For questions about this implementation:
- Repository: [github.com/mehara-rothila/Idealize-Datathon](https://github.com/mehara-rothila/Idealize-Datathon)
- Issues: Use GitHub Issues for technical questions

---

**⚡ Ready to predict survival outcomes with state-of-the-art machine learning!**