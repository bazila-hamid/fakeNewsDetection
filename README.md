# Fake News Detection Using Machine Learning

A web application that classifies news article text as likely **Fake** or **Real** using machine learning and natural language processing techniques.

## Overview

This project uses a TF-IDF text representation and a Logistic Regression classifier to analyse submitted news text. A Flask web application provides a simple interface where users can enter an article and view the model's prediction.

## Features

- Accepts news article text through a web interface.
- Cleans text before processing.
- Converts text into numerical features using TF-IDF.
- Classifies the submitted text as Fake or Real.
- Displays the prediction on the results page.

## Technologies Used

- **Language:** Python
- **Web framework:** Flask
- **Machine learning:** scikit-learn, Logistic Regression
- **Text features:** TF-IDF
- **Data handling:** pandas, NumPy
- **Model persistence:** joblib
- **Frontend:** HTML, CSS

## Project Structure

```text
fakeNewsDetection/
├── app.py
├── train.py
├── requirement.txt
├── dataset/
│   ├── Fake.csv
│   └── True.csv
├── model/
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
└── templates/
    └── index.html
```

## Getting Started

### Prerequisites

- Python installed on your computer
- The project files and dataset

### Installation

Clone the repository:

```bash
git clone https://github.com/bazila-hamid/fakeNewsDetection.git
cd fakeNewsDetection
```

Create and activate a virtual environment on Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirement.txt
```

### Run the application

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

### Train the model

If you want to retrain the model and vectorizer using the included dataset, run:

```bash
python train.py
```

Then restart the application.

## Limitations

This project is a text-classification demonstration. Predictions are based on patterns learned from its training dataset. The application does not independently verify claims against trusted news sources, and a prediction may be incorrect. It should not be treated as definitive proof that an article is true or false.

## Author

**Bazila Hamid**

[GitHub Profile](https://github.com/bazila-hamid)
