# Code_Tech_03
## 🔍 Overview of the CodTechBot Project
This chatbot is built using:

NLTK: For natural language processing (tokenizing, lemmatizing).

TextBlob: (Optional) for correcting misspellings (not used in final version).

scikit-learn: To classify user input using a Naive Bayes model.

JSON: To store patterns (inputs) and responses for various "intents".

Console or GUI (Tkinter): For chatting with the bot.
## 🧠 Step-by-Step Explanation
```python

{
  "tag": "greeting",
  "patterns": ["Hi", "Hello", "Hey", "hi", "hlo"],
  "responses": ["Hello! 😊", "Hi there! 👋"]
}
```

tag: Unique name for the intent category (e.g., greeting).

patterns: Sample user inputs (used to train the model).

## 🛠️ Step 2: Install & Import Required Libraries

```python
pip install nltk textblob scikit-learn

import nltk
import numpy as np
import json
import random
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
```
These libraries help with:

NLP (tokenizing, lemmatizing)

Building ML models

Handling JSON data
## 📦 Step 3: Download NLTK Resources
```python
nltk.download('punkt')
nltk.download('wordnet')
```
These are required for:

punkt: Sentence splitting and tokenizing

wordnet: For lemmatizing words (converting "running" → "run")
## 🧹 Step 4: Preprocessing Patterns
```python
lemmatizer = WordNetLemmatizer()
```
Then, tokenize each sentence from patterns, convert to lowercase, and clean up:

```python

def simple_tokenize(text):
    text = text.lower().replace("?", "").replace("!", "").replace(".", "").replace(",", "")
    return word_tokenize(text)
```
Why?

Tokenizing = splitting sentence into words

Lemmatizing = reducing words to their root form

Cleaning = remove punctuation for consistency

## 🧾 Step 5: Convert Words into Numbers (Bag of Words)
```python
def bag_of_words(tokenized_sentence, all_words):
    ...
```
We convert each pattern into a vector of 0s and 1s indicating the presence of known words.
Example:

Vocabulary: [hello, help, your, name]

Input: "hello name" → [1, 0, 0, 1]

This makes the data numerical, so machine learning models can understand it.

 ##  Step 6: Create Dataset (X, y)
```python
X_train = []
y_train = []
```
For every pattern → tag, we:

Convert pattern to a bag of words

Use the index of the tag as the label

This creates:

X_train: List of bag-of-words vectors

y_train: List of corresponding tag indices

## 🤖 Step 7: Train the Model
```python

model = MultinomialNB()
model.fit(X_train, y_train)
```
We use a Naive Bayes classifier (simple and effective for text).

It learns to classify each sentence based on the words it contains.

## 🔍 Step 8: Predict Tags for User Input
```python

def predict_tag_with_confidence(sentence):
    ...
```
Tokenizes and lemmatizes user input

Converts to bag-of-words vector

Gets predicted tag and its confidence

If confidence is above 50%, we return the tag.
If not, we return "unknown" (to handle unrecognized input).

## 💬 Step 9: Generate a Response
```python
def get_response(tag):
    ...
```
We check the intents.json for matching tag

Randomly select a response from that intent

If no match, return a fallback response like "I didn't understand".

## 🗃️ Step 10: Log the Conversation
```python

def log_chat(user, bot):
    ...
```
Saves each user input and bot response to chat_log.txt

Helps for reviewing conversations or improving the bot

## 🧑‍💻 Step 11: Start the Chat (Console Version)
```python

def chat():
    print("CodTechBot: Hello! Type 'quit' to exit.")
    ...
```
Takes user input in a loop

Predicts the tag

Generates response

Logs the chat









