"""
Flask front end for the Elevate Retail sentiment analysis demo.

Run with:
    cd /Users/pavan/python
    source .venv/bin/activate
    flask --app flask_sentiment_app run --reload
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template_string, request

from ecommerce_sentiment_svm import (
    ASPECT_KEYWORDS,
    build_vectorizer,
    ensure_stopwords,
    load_dataset,
    prepare_features,
    train_svm,
)

DATA_PATH = Path(__file__).with_name("data").joinpath("ecommerce_reviews.csv")

# Train the TF-IDF + SVM pipeline once at startup so requests just reuse it.
# Using smaller sample (2000 per class) for faster Flask startup
_dataset = load_dataset(DATA_PATH, max_per_class=2000)
_stop_words = ensure_stopwords()
_vectorizer = build_vectorizer(_stop_words)
_X_train, _X_test, _y_train, _y_test = prepare_features(_dataset, _vectorizer)
_svm_model = train_svm(_X_train, _y_train)

app = Flask(__name__)

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>E-Commerce Sentiment Analysis | Elevate Retail Solutions</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 2rem 1rem;
      color: #333;
    }
    .container {
      max-width: 800px;
      margin: 0 auto;
      background: white;
      border-radius: 16px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      overflow: hidden;
    }
    .header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 2rem;
      text-align: center;
    }
    .header h1 {
      font-size: 2rem;
      margin-bottom: 0.5rem;
      font-weight: 700;
    }
    .header p {
      opacity: 0.9;
      font-size: 1rem;
    }
    .content {
      padding: 2rem;
    }
    .form-group {
      margin-bottom: 1.5rem;
    }
    label {
      display: block;
      font-weight: 600;
      margin-bottom: 0.5rem;
      color: #555;
      font-size: 0.95rem;
    }
    textarea {
      width: 100%;
      min-height: 180px;
      padding: 1rem;
      font-size: 1rem;
      border: 2px solid #e0e0e0;
      border-radius: 8px;
      font-family: inherit;
      resize: vertical;
      transition: border-color 0.3s;
    }
    textarea:focus {
      outline: none;
      border-color: #667eea;
      box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    .btn {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border: none;
      padding: 0.875rem 2rem;
      font-size: 1rem;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      transition: transform 0.2s, box-shadow 0.2s;
      width: 100%;
    }
    .btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    .btn:active {
      transform: translateY(0);
    }
    .result {
      margin-top: 2rem;
      padding: 1.5rem;
      border-radius: 12px;
      background: #f8f9fa;
      border-left: 4px solid;
      animation: slideIn 0.3s ease-out;
    }
    @keyframes slideIn {
      from { opacity: 0; transform: translateY(-10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .result.positive { border-color: #10b981; background: #ecfdf5; }
    .result.negative { border-color: #ef4444; background: #fef2f2; }
    .result.neutral { border-color: #6b7280; background: #f3f4f6; }
    .sentiment-badge {
      display: inline-block;
      padding: 0.5rem 1rem;
      border-radius: 20px;
      font-weight: 600;
      font-size: 0.95rem;
      margin-bottom: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .sentiment-badge.positive {
      background: #10b981;
      color: white;
    }
    .sentiment-badge.negative {
      background: #ef4444;
      color: white;
    }
    .sentiment-badge.neutral {
      background: #6b7280;
      color: white;
    }
    .result-title {
      font-size: 0.9rem;
      color: #6b7280;
      margin-bottom: 0.5rem;
      font-weight: 500;
    }
    .aspects {
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px solid rgba(0,0,0,0.1);
    }
    .aspects-title {
      font-size: 0.9rem;
      color: #6b7280;
      margin-bottom: 0.75rem;
      font-weight: 500;
    }
    .aspect-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    .aspect-tag {
      display: inline-block;
      padding: 0.4rem 0.875rem;
      background: white;
      border: 1px solid #e0e0e0;
      border-radius: 20px;
      font-size: 0.85rem;
      color: #555;
      font-weight: 500;
    }
    .no-aspects {
      color: #9ca3af;
      font-size: 0.9rem;
      font-style: italic;
    }
    .stats {
      margin-top: 2rem;
      padding: 1rem;
      background: #f8f9fa;
      border-radius: 8px;
      font-size: 0.85rem;
      color: #6b7280;
      text-align: center;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>📊 E-Commerce Sentiment Analysis</h1>
      <p>Analyze product reviews and identify customer sentiment using AI</p>
    </div>
    <div class="content">
      <form method="post">
        <div class="form-group">
          <label for="review">Product Review</label>
          <textarea id="review" name="review" placeholder="Enter or paste a product review here...&#10;&#10;Example: 'The battery life is fantastic and the screen is crisp. Highly recommend!'">{{ review }}</textarea>
        </div>
        <button type="submit" class="btn">🔍 Analyze Sentiment</button>
      </form>
      
      {% if prediction %}
      <div class="result {{ prediction.lower() }}">
        <div class="result-title">Analysis Result</div>
        <span class="sentiment-badge {{ prediction.lower() }}">{{ prediction }}</span>
        
        {% if aspects %}
        <div class="aspects">
          <div class="aspects-title">Product Aspects Mentioned</div>
          <div class="aspect-tags">
            {% for aspect in aspects %}
            <span class="aspect-tag">{{ aspect }}</span>
            {% endfor %}
          </div>
        </div>
        {% else %}
        <div class="aspects">
          <div class="no-aspects">No specific product aspects detected</div>
        </div>
        {% endif %}
      </div>
      {% endif %}
      
      <div class="stats">
        Powered by SVM Machine Learning | Trained on {{ dataset_size }} reviews
      </div>
    </div>
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    review_text = ""
    prediction = None
    aspects_triggered = []

    if request.method == "POST":
        review_text = request.form.get("review", "").strip()
        if review_text:
            features = _vectorizer.transform([review_text])
            prediction = _svm_model.predict(features)[0]
            lowered = review_text.lower()
            aspects_triggered = [
                aspect
                for aspect, keywords in ASPECT_KEYWORDS.items()
                if any(word in lowered for word in keywords)
            ]

    return render_template_string(
        PAGE_TEMPLATE,
        review=review_text,
        prediction=prediction,
        aspects=aspects_triggered,
        dataset_size=len(_dataset.reviews),
    )


if __name__ == "__main__":
    app.run(debug=True)

