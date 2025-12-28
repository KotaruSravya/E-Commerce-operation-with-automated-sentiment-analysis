# E-Commerce Sentiment Analysis System

Automated sentiment analysis system for Elevate Retail Solutions using Support Vector Machine (SVM) to classify product reviews into Positive, Negative, and Neutral categories.

## 📁 Required Files

### Core Python Files
1. **`ecommerce_sentiment_svm.py`** (201 lines)
   - Main training and evaluation script
   - Dataset loading and preprocessing
   - TF-IDF feature extraction
   - SVM model training
   - Model evaluation with metrics

2. **`flask_sentiment_app.py`** (100 lines)
   - Flask web application
   - Interactive web interface for sentiment analysis
   - Real-time review analysis
   - Aspect detection (battery, performance, shipping, design, audio)

3. **`ecommerce_sentiment_analysis.ipynb`** (219 lines)
   - Jupyter notebook for interactive analysis
   - Step-by-step workflow demonstration
   - Data exploration and visualization
   - Model comparison (SVM vs Logistic Regression)

### Data Files
4. **`data/ecommerce_reviews.csv`** (4.1 KB)
   - Curated sample dataset (60 reviews)
   - Columns: `review`, `sentiment`

5. **`/Users/pavan/Downloads/train.csv`** (168 MB, 3.6M rows)
   - Primary training dataset
   - Automatically loaded if available
   - Columns: `rating` (1=Negative, 2=Positive), `title`, `review`

### Environment
6. **`.venv/`** directory
   - Python virtual environment
   - Required packages: `pandas`, `numpy`, `scikit-learn`, `nltk`, `flask`

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd /Users/pavan/python
source .venv/bin/activate
```

### 2. Run Training Script
```bash
python ecommerce_sentiment_svm.py
```

### 3. Start Web Application
```bash
FLASK_APP=flask_sentiment_app flask run
```
Then open: `http://127.0.0.1:5000`

### 4. Use Jupyter Notebook
```bash
jupyter notebook ecommerce_sentiment_analysis.ipynb
```

## 📊 Features

- **Sentiment Classification**: Positive, Negative, Neutral
- **Aspect Detection**: Identifies mentions of battery, performance, shipping, design, audio
- **Modern Web Interface**: Clean, responsive UI with color-coded results
- **High Accuracy**: ~87.5% accuracy on large dataset
- **Real-time Analysis**: Instant sentiment prediction for any review

## 🔧 Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- nltk
- flask

## 📈 Model Performance

- **Dataset**: 20,063 reviews (from train.csv)
- **Accuracy**: 87.5%
- **Algorithm**: Linear SVM with TF-IDF features
- **Features**: Bigram TF-IDF vectors with stopword removal

## 🎯 Use Cases

- Analyze customer feedback in real-time
- Identify product strengths and weaknesses
- Monitor customer satisfaction trends
- Automate review categorization
- Extract actionable insights from reviews

