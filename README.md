# 📊 Student Performance Prediction - ML Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive machine learning solution for predicting student academic performance using advanced regression algorithms and data preprocessing techniques.

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline to predict student exam scores based on various demographic and academic factors. The system leverages multiple machine learning algorithms including CatBoost, XGBoost, and traditional scikit-learn models to achieve optimal prediction accuracy.

### Key Features

- 🔍 **Comprehensive EDA**: Detailed exploratory data analysis with visualization
- 🛠️ **Modular Architecture**: Well-structured, reusable components
- 📈 **Multiple ML Models**: Evaluation of 8+ regression algorithms
- ⚙️ **Automated Pipeline**: Seamless data ingestion, transformation, and model training
- 🎨 **Custom Exception Handling**: Robust error management system
- 📝 **Logging Framework**: Comprehensive logging for debugging and monitoring

## 🏗️ Project Structure

```
ML_PROJECT/
│
├── artifacts/                  # Stored models and preprocessed data
├── notebook/
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb
│   ├── 2. MODEL TRAINING.ipynb
│   └── data/                   # Raw dataset
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Data loading and splitting
│   │   ├── data_transformation.py  # Feature engineering & preprocessing
│   │   └── model_trainer.py        # Model training and evaluation
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py       # Training pipeline orchestration
│   │   └── predict_pipeline.py     # Prediction pipeline
│   │
│   ├── exception.py            # Custom exception handling
│   ├── logger.py              # Logging configuration
│   └── utils.py               # Utility functions
│
├── requirements.txt           # Project dependencies
├── setup.py                  # Package configuration
└── README.md                 # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/RAJVEER42/ML_PROJECT.git
   cd ML_PROJECT
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Training the Model

```python
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Initialize components
data_ingestion = DataIngestion()
data_transformation = DataTransformation()
model_trainer = ModelTrainer()

# Execute pipeline
train_data, test_data = data_ingestion.initiate_data_ingestion()
train_arr, test_arr = data_transformation.initiate_data_transformation(train_data, test_data)
model_trainer.initiate_model_trainer(train_arr, test_arr)
```

### Making Predictions

```python
from src.pipeline.predict_pipeline import PredictPipeline

# Initialize prediction pipeline
predict_pipeline = PredictPipeline()

# Make predictions
predictions = predict_pipeline.predict(input_data)
```

## 📊 Machine Learning Models

The project evaluates the following regression models:

- **Linear Regression**
- **Lasso Regression**
- **Ridge Regression**
- **K-Neighbors Regressor**
- **Decision Tree**
- **Random Forest Regressor**
- **XGBoost Regressor**
- **CatBoost Regressor**
- **AdaBoost Regressor**

Model selection is based on comprehensive evaluation metrics including R² score, RMSE, and MAE.

## 🔧 Technologies & Libraries

- **Core ML**: scikit-learn, XGBoost, CatBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Serialization**: dill
- **Others**: Custom logging and exception handling

## 📈 Dataset

The project uses a student performance dataset containing features such as:
- Demographic information
- Parental education level
- Study habits
- Previous test scores
- Other relevant academic indicators

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ End-to-end ML project development
- ✅ Object-oriented programming in ML
- ✅ Custom exception handling and logging
- ✅ Data preprocessing and feature engineering
- ✅ Model evaluation and selection
- ✅ Pipeline creation for scalability

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Rajveer**

- GitHub: [@RAJVEER42](https://github.com/RAJVEER42)
- Email: irajveer.bishnoi2310@gmail.com

## 🙏 Acknowledgments

- scikit-learn documentation
- CatBoost and XGBoost communities
- Open source ML community

## 📞 Contact

For any queries or suggestions, please reach out:
- 📧 Email: irajveer.bishnoi2310@gmail.com
- 💼 GitHub: [RAJVEER42](https://github.com/RAJVEER42)

---

⭐ If you find this project helpful, please consider giving it a star!

**Made with ❤️ by Rajveer**