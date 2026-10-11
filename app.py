from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Clean text
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        news = request.form["news"]

        news = clean_text(news)

        vector = vectorizer.transform([news])

        result = model.predict(vector)

        if result[0] == 0:
         prediction = "❌ This news is predicted to be FAKE."

        else:
         prediction = "✅ This news is predicted to be REAL."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
