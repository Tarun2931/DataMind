<div align="center">

# 🧠 DataMind
### AI-Powered Dataset Analyzer & ML Pipeline


<p align="center">
  <b>Upload a CSV. Get a trained ML model. Ask AI anything about your data.</b><br/>
  No code required.
</p>

<br/>

</div>

---

## 📌 What is DataMind?

**DataMind** is an end-to-end Machine Learning web application that automates the complete ML workflow for tabular datasets. Upload a CSV, select your target column — and DataMind handles everything else.

From preprocessing to model comparison, EDA to predictions, and an AI chatbot that explains results in plain language — DataMind is your intelligent data science assistant.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Auto Problem Detection** | Automatically identifies Regression or Classification |
| ⚙️ **Smart Preprocessing** | Handles missing values, outliers, skewness, encoding |
| 📊 **Full EDA Report** | Statistical summaries, skewness, outliers, missing values |
| 🤖 **13 Models Compared** | Trains all models and picks the best automatically |
| 📈 **Rich Visualizations** | Histogram, Boxplot, Scatter, Heatmap, Countplot |
| 🎯 **Model Evaluation** | CV Score, Feature Importance, Confusion Matrix, ROC-AUC |
| ⚡ **Live Prediction** | Enter values and get instant predictions |
| 💬 **AI Chatbot** | LangChain + Groq powered assistant for your dataset |
| 🔄 **Raw vs Processed** | Compare data distributions before and after preprocessing |

---

## 🖼️ Screenshots

> *Add your screenshots here*

| Home | Results | EDA Report |
|---|---|---|
| ![Home]() | ![Results]() | ![EDA]() |

| Insights | AI Chatbot | ROC Curve |
|---|---|---|
| ![Insights]() | ![Chatbot]() | ![ROC]() |

---

## 🧩 Architecture

```
DataMind/
│
├── back.py                        # FastAPI — all routes & app logic
│
├── ml_models/
│   ├── problem_detection.py       # Auto detect regression / classification
│   ├── preprocessing.py           # Full preprocessing pipeline
│   ├── classification_models.py   # 6 classifiers + CV + feature importance
│   ├── regression_models.py       # 7 regressors + CV + feature importance
│   ├── exploratory_data_analysis.py # EDA functions
│   ├── prediction.py              # Inference on new input
│   └── llm.py                     # LangChain + Groq chatbot
│
├── graphs/
│   ├── histogram.py
│   ├── boxplot.py
│   ├── scatterplot.py
│   ├── heatmap.py
│   ├── countplot.py
│   ├── roc_curve.py
│   ├── residual.py
│   └── actual_vs_predicted.py
│
├── templates/
│   ├── index.html
│   ├── insights.html
│   ├── chatbot.html
│   ├── roc_curve.html
│   ├── residual.html
│   ├── actual_vs_predicted.html
│   └── eda.html
│
├── static/
│   └── graphs/                    # Generated graph images saved here
│
└── requirements.txt
```

---

## 🤖 ML Models Included

### Classification
| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Interpretable |
| **Random Forest** | Usually best 🏆 |
| XGBoost | High performance |
| SVC | Kernel-based |
| Gaussian Naive Bayes | Probabilistic |

### Regression
| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | L2 regularization |
| Lasso Regression | L1 regularization |
| Decision Tree | Non-linear |
| **Random Forest** | Robust ensemble |
| **XGBoost** | Usually best 🏆 |
| SVR | Kernel-based |

---

## ⚙️ Preprocessing Pipeline

```
Raw CSV
   │
   ├── 1. Remove Duplicates
   ├── 2. Handle Missing Values     (mean/median for numerical, mode for categorical)
   ├── 3. Encode Features           (OrdinalEncoder, LabelEncoder, OneHotEncoder)
   ├── 4. Handle Outliers           (IQR Clipping)
   ├── 5. Fix Skewness              (Yeo-Johnson Transform)
   └── 6. Remove Low Variance       (VarianceThreshold)
          │
          ▼
   Preprocessed Data → Model Training
```

---

## 📊 Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1 Score
- Cross Validation Mean & Std (5-Fold)
- Confusion Matrix
- ROC-AUC Curve

### Regression
- RMSE, MAE, R² Score
- Cross Validation Mean & Std (5-Fold)
- Actual vs Predicted Plot
- Residual Plot

---



## 📋 Requirements

```
fastapi
uvicorn
jinja2
python-multipart
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib
langchain
groq
python-dotenv
```

---

## 🔄 How It Works

```
1. Upload CSV  →  Select Target Column  →  Click Analyze
       │
       ▼
2. Auto Problem Detection (Regression / Classification)
       │
       ▼
3. Preprocessing Pipeline runs automatically
       │
       ▼
4. EDA Report generated (stats, missing, skew, outliers)
       │
       ▼
5. All models trained & compared
       │
       ▼
6. Best model selected automatically
       │
       ├── View Feature Importance
       ├── View Confusion Matrix / ROC Curve
       ├── View Residual / Actual vs Predicted Plot
       ├── Make Predictions
       └── Ask AI Chatbot anything about your results
```

---

## 💬 AI Chatbot

DataMind includes an **AI-powered Data Scientist chatbot** built with:

- **LangChain** — for prompt management and chaining
- **Groq API** — ultra-fast LLM inference

The chatbot has full context of your:
- Dataset summary & EDA results
- Model performance metrics
- Feature importance rankings
- Problem type and best model

**Example questions you can ask:**
```
"Why did Random Forest perform better than XGBoost?"
"Which features are most important for prediction?"
"What does the cross-validation score tell me?"
"How can I improve my model accuracy?"
"Explain the confusion matrix results"
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI, Python |
| **ML** | Scikit-learn, XGBoost |
| **AI Chatbot** | LangChain, Groq API |
| **Visualization** | Matplotlib, Seaborn |
| **Frontend** | HTML, CSS, Bootstrap 5 |
| **Templating** | Jinja2 |
| **Data** | Pandas, NumPy |

---

## 📈 Future Roadmap

- [ ] Unsupervised Learning (KMeans Clustering, Anomaly Detection)
- [ ] Deep Learning support (CNN for image classification)
- [ ] Model export & download (.pkl)
- [ ] PDF report generation
- [ ] User authentication & history
- [ ] Dataset auto-cleaning suggestions from AI

---

## 🙌 Contributing

Contributions are welcome! Feel free to:

1. Fork the repo
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by **[Your Name](https://github.com/yourusername)**

⭐ **Star this repo if you found it useful!** ⭐

</div>
