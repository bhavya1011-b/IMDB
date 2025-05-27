# Project 2: IMDB Sentiment Analysis Using Deep Learning

# 1. Project Content
Build a deep learning model to classify IMDB movie reviews as positive or negative based on textual content.

# 2. Project Code

~~~python
import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

nltk.download('stopwords')
movie_reviews = pd.read_csv("/content/IMDB Dataset.csv")

# Clean reviews
def preprocess(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text().lower()
    tokens = [word for word in text.split() if word not in stopwords.words('english')]
    return ' '.join(tokens)

movie_reviews['cleaned'] = movie_reviews['review'].apply(preprocess)

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(movie_reviews['cleaned'])
sequences = tokenizer.texts_to_sequences(movie_reviews['cleaned'])
X = pad_sequences(sequences, maxlen=200)
y = pd.get_dummies(movie_reviews['sentiment']).values

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# LSTM Model
model = Sequential([
    Embedding(5000, 64, input_length=200),
    LSTM(64),
    Dense(2, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
~~~

# 3. Key Technologies
- Python
- NLTK, BeautifulSoup (Text preprocessing)
- Keras/TensorFlow (Neural Networks: Embedding, LSTM)
- Pandas (Data handling)
- Seaborn/Matplotlib (for EDA - if included)

# 4. Description
This project takes raw movie reviews, cleans them using HTML parsing and stopword removal, and converts them into padded sequences. An LSTM-based neural network is trained to detect sentiment, providing a powerful text classification pipeline. The model learns to associate patterns in text with positive or negative sentiments.

# 5. Output
- Accuracy of the sentiment analysis model.
- Trained model that can classify any new review as positive or negative.
- Can be extended to provide sentiment scores for batches of reviews.

# 6. Further Research
- Use pre-trained word embeddings (like GloVe or Word2Vec).
- Try different RNN variants like GRU or Bidirectional LSTM.
- Deploy the model as an API using FastAPI or Flask.
- Visualize word importance using attention mechanisms.
