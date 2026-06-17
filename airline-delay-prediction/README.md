# ✈️ Airline Delay Prediction

> Machine learning ensemble model predicting flight delays using Random Forests, Logistic Regression, and Neural Networks on a 530K+ flight dataset, achieving 66.9% accuracy.

## 📊 Overview

This project trains and ensembles multiple ML models to predict whether a flight will be delayed based on airline, route, timing, and seasonal features. It demonstrates end-to-end ML workflow from feature engineering to model evaluation.

## 🛠️ Tools & Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

## 📁 Project Structure

```
airline-delay-prediction/
│
├── airline_delay_prediction.py   # Main ML pipeline script
├── airline_delay_data.csv        # Sample dataset
├── airline_delay_analysis.png    # Visualizations
└── README.md
```

## 🔍 Key Steps

1. **Data Generation** — Simulated 530K+ flight records with airline, route, timing, and delay features
2. **Feature Engineering** — Encoded categorical variables, created peak season and red-eye flags
3. **Model Training** — Logistic Regression, Random Forest, Ensemble (Voting Classifier)
4. **Evaluation** — Accuracy, classification report, confusion matrix, feature importance
5. **Visualization** — Model comparison, feature importance, delay distribution

## 📈 Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | Baseline |
| Random Forest | Best single model |
| Ensemble | Combined prediction |
| **Final (full dataset)** | **66.9%** |

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python airline_delay_prediction.py
```

## 📬 Contact

**Visalakshi Polepalli** — [LinkedIn](https://linkedin.com/in/visalakshi-polepalli17)
