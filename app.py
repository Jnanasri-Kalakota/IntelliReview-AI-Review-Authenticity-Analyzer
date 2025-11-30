from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast

# ----------------- FLASK APP -----------------
app = Flask(__name__)

# Load BERT model and tokenizer
print("Loading DistilBERT model...")
bert_model = TFDistilBertForSequenceClassification.from_pretrained("bert_review_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("bert_review_tokenizer")
print("Model loaded successfully!")

# ----------------- PREDICTION FUNCTION -----------------
def predict_bert(review: str):
    inputs = tokenizer(
        review,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=256
    )

    logits = bert_model(inputs).logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    
    fake_prob = probs[0] * 100
    real_prob = probs[1] * 100

    label = "OR (Original)" if real_prob > fake_prob else "CG (Computer Generated)"
    return label, real_prob, fake_prob

# ----------------- ROUTES -----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    real_conf = None
    fake_conf = None
    user_text = None

    if request.method == "POST":
        user_text = request.form.get("review_text", "")

        if user_text.strip() == "":
            prediction = "Please enter a review."
        else:
            prediction, real_conf, fake_conf = predict_bert(user_text)

    return render_template(
        "index.html",
        prediction=prediction,
        real_conf=real_conf,
        fake_conf=fake_conf,
        user_text=user_text
    )

if __name__ == "__main__":
    app.run(debug=True)
