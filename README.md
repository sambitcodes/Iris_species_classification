# 🌸 IRIS Flower Classification Project

A comprehensive machine learning project for classifying IRIS flower species using multiple algorithms with an interactive Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Dataset](#dataset)
- [Results](#results)
- [Screenshots](#screenshots)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a complete machine learning pipeline for the famous IRIS flower dataset. It includes:
- Exploratory Data Analysis (EDA)
- Multiple classification algorithms
- Hyperparameter tuning
- Model evaluation and comparison
- Interactive web application for predictions

## ✨ Features

### 🔬 Machine Learning Models
- **Logistic Regression**: Linear classification with regularization
- **Decision Tree**: Tree-based classification with pruning
- **Naive Bayes**: Probabilistic classifier (Gaussian, Bernoulli, Multinomial)
- **Bagging Classifier**: Ensemble method with bootstrap aggregation
- **XGBoost**: Gradient boosting framework
- **AdaBoost**: Adaptive boosting ensemble
- **MLP Classifier**: Multi-layer perceptron neural network

### 📊 Interactive Web App
- **Prediction Tab**:
  - Feature input using interactive sliders
  - Model selection dropdown
  - Real-time species prediction
  - Confidence scores for all classes
  - Model comparison across all algorithms
  - Classification report and confusion matrix
  - 3D cluster visualization with decision boundaries

- **Exploration Tab**:
  - Multiple plot types (scatter, box, violin, histogram, 3D)
  - Interactive feature selection
  - Correlation heatmap
  - Dataset statistics

## 📁 Project Structure

```
IRIS-Flower-Classification/
│
├── notebooks/                          # Jupyter Notebooks
│   ├── 01_Dataset_Generation_EDA.ipynb           # Data loading, EDA, cleaning
│   ├── 02_Logistic_Regression_Classifier.ipynb   # Logistic Regression model
│   ├── 03_Decision_Tree_Classifier.ipynb         # Decision Tree model
│   ├── 04_Naive_Bayes_Classifier.ipynb           # Naive Bayes variants
│   ├── 05_Bagging_Classifier.ipynb               # Bagging ensemble
│   ├── 06_XGBoost_Classifier.ipynb               # XGBoost model
│   ├── 07_AdaBoost_Classifier.ipynb              # AdaBoost ensemble
│   └── 08_MLP_Classifier.ipynb                   # Neural network
│
├── models/                            # Trained models and data
│   ├── cleaned_iris_dataset.csv                  # Cleaned dataset
│   ├── logistic_regression_model.pkl             # Trained LR model
│   ├── decision_tree_model.pkl                   # Trained DT model
│   ├── naive_bayes_model.pkl                     # Trained NB model
│   ├── bagging_model.pkl                         # Trained Bagging model
│   ├── xgboost_model.pkl                         # Trained XGBoost model
│   ├── adaboost_model.pkl                        # Trained AdaBoost model
│   └── mlp_model.pkl                             # Trained MLP model
│
├── app.py                             # Streamlit web application
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── .gitignore                         # Git ignore file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/IRIS-Flower-Classification.git
cd IRIS-Flower-Classification
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run Jupyter notebooks** (to train models)
```bash
jupyter notebook
```
Execute notebooks in order (01 through 08) to:
- Prepare the dataset
- Train all models
- Generate model files in `models/` directory

5. **Launch the Streamlit app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 💻 Usage

### Training Models

Navigate to the `notebooks/` directory and run notebooks sequentially:

```bash
cd notebooks
jupyter notebook
```

1. **01_Dataset_Generation_EDA.ipynb**: 
   - Loads IRIS dataset
   - Performs EDA
   - Cleans and exports data

2. **02-08 (Model notebooks)**:
   - Train specific model
   - Hyperparameter tuning
   - Evaluation metrics
   - Export trained model

### Running the Web App

```bash
streamlit run app.py
```

**Prediction Workflow:**
1. Adjust feature sliders (sepal/petal dimensions)
2. Select a model from dropdown
3. Click "Predict Species"
4. View prediction, confidence scores, and comparisons
5. Explore confusion matrix and classification report
6. Visualize data in 3D

**Exploration Workflow:**
1. Select plot type
2. Choose features to visualize
3. Click "Explore Data"
4. Interact with plots (zoom, pan, rotate for 3D)

## 🤖 Models

### Model Performance Summary

| Model | Test Accuracy | Training Time | Best Use Case |
|-------|--------------|---------------|---------------|
| Logistic Regression | ~97% | Fast | Linear relationships |
| Decision Tree | ~95% | Fast | Interpretability |
| Naive Bayes | ~96% | Very Fast | Probabilistic inference |
| Bagging | ~97% | Moderate | Reduce variance |
| XGBoost | ~97% | Moderate | Complex patterns |
| AdaBoost | ~96% | Moderate | Boosting weak learners |
| MLP | ~97% | Slower | Non-linear patterns |

### Hyperparameter Tuning

Each model notebook includes:
- **GridSearchCV** or **RandomizedSearchCV**
- Cross-validation (5-fold)
- Best parameter selection
- Performance comparison

## 📊 Dataset

**IRIS Flower Dataset** (Fisher, 1936)

- **Samples**: 150 (50 per class)
- **Features**: 4
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Classes**: 3
  - Setosa (0)
  - Versicolor (1)
  - Virginica (2)

**Source**: UCI Machine Learning Repository / Scikit-learn

## 📈 Results

### Key Findings

1. **Feature Importance**:
   - Petal length and petal width are most discriminative
   - Strong correlation between petal dimensions
   - Setosa is linearly separable from other species

2. **Model Insights**:
   - All models achieve >95% accuracy
   - Ensemble methods (Bagging, XGBoost) slightly outperform
   - Simple models (Logistic Regression) perform competitively
   - MLP requires more tuning but captures non-linear patterns

3. **Classification Patterns**:
   - Setosa: 100% correctly classified
   - Versicolor: Minor confusion with Virginica
   - Virginica: Some overlap with Versicolor

## 🖼️ Screenshots

### Prediction Interface
![Prediction Tab](screenshots/prediction_tab.png)

### Data Exploration
![Exploration Tab](screenshots/exploration_tab.png)

### Model Comparison
![Model Comparison](screenshots/model_comparison.png)

## 🛠️ Technologies Used

- **Python 3.8+**
- **Scikit-learn**: ML algorithms and preprocessing
- **XGBoost**: Gradient boosting
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive plots
- **Streamlit**: Web application framework
- **Jupyter**: Interactive notebooks

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 👨‍💻 Author

**Your Name**
- GitHub: [@sambitcodes](https://github.com/sambitcodes)
- LinkedIn: [Sambit Chakraborty](https://www.linkedin.com/in/samchak/)

## 🙏 Acknowledgments

- Ronald Fisher for the IRIS dataset
- UCI Machine Learning Repository
- Scikit-learn documentation
- Streamlit community
- XGBoost developers

## 📧 Contact

For questions or suggestions, please open an issue or contact:
- Email: sambitmaths123@gmail.com
- Project Link: https://github.com/sambitcodes/Iris_species_classification

---

**⭐ If you find this project useful, please consider giving it a star!**
