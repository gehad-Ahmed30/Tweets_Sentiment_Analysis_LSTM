# Twitter Sentiment Analysis using LSTM

This project implements a **Twitter Sentiment Analysis web application** using a **Bidirectional LSTM model**. The system classifies tweets into four categories: **Positive, Negative, Neutral, and Irrelevant**. It provides a simple web interface where users can input a tweet and receive the predicted sentiment along with confidence scores.

## Features

- **Deep Learning Model:** Bidirectional LSTM trained on tweet data for accurate sentiment prediction.
- **Preprocessing:** Text cleaning including removing URLs, mentions, special characters, punctuation, and stopwords.
- **Tokenizer & LabelEncoder:** Tokenizer and label encoder are saved and loaded for consistent predictions.
- **Web Application:** Built with Flask for easy interaction through a user-friendly interface.
- **Deployment Ready:** Can be deployed locally or on cloud platforms with minimal configuration.

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- Numpy & Pandas
- NLTK (Natural Language Processing)
- Flask (Web Framework)
- HTML & CSS (Frontend)

## File Structure
```
├── app.py # Main Flask application
├── templates/
│ └── index.html # HTML template for the web interface
├── static/
│ └── style.css # CSS styling for the interface
├── models/
│ ├── BestLstm_Model.h5 # Trained LSTM model
│ ├── tokenizer_acc.pkl # Tokenizer for preprocessing text
│ └── label_encoder_acc.pkl# LabelEncoder for output classes
└── README.md # Project description
```

## Usage

1. Clone the repository:
```bash
git clone <repo-url>
```
## License

This project is licensed under the MIT License.

## Kaggle:
https://www.kaggle.com/code/gehad830/tweets-sentiment-analysis-lstm


