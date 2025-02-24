import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

from ..s06_clean_data.clean_data import clean_sentence

DATA_FOLDER = 'data'

# Read data
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
input_file = os.path.join(parent_dir, "s06_clean_data", DATA_FOLDER, "data_after_clean.csv")
data = pd.read_csv(input_file)
print('\n---------------------------------------------------------------------')
print('\nSource data:')
print(data.head())

# Encode intents
label_encoder = LabelEncoder()
data['intent_encoded'] = label_encoder.fit_transform(data['intent'])
print('\n---------------------------------------------------------------------')
print('')
for intent, encoded_value in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"Intent: {intent} -> {encoded_value}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data["utterance"], 
    data["intent_encoded"], 
    test_size=0.2, 
    random_state=0)

# Vectorize with TF-IDF + N-Gram + Stop Words
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train
print('\n---------------------------------------------------------------------')
print("\nTraining the model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print('\n---------------------------------------------------------------------')
print("\nEvaluation results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
model_dir = os.path.join(script_dir, "model")
joblib.dump(model, model_dir + "/intent_classification_model.pkl")
joblib.dump(vectorizer, model_dir + "/tfidf_vectorizer.pkl")
joblib.dump(label_encoder, model_dir + "/label_encoder.pkl")
print('\n---------------------------------------------------------------------')
print("\nModel has been saved as 'intent_classification_model.pkl', 'tfidf_vectorizer.pkl', and 'label_encoder.pkl'.")

# Load model
loaded_model = joblib.load(model_dir + "/intent_classification_model.pkl")
loaded_vectorizer = joblib.load(model_dir + "/tfidf_vectorizer.pkl")
loaded_label_encoder = joblib.load(model_dir + "/label_encoder.pkl")

# Predict new data
new_sentence = "how to cancel my order?"
new_sentence_cleaned = clean_sentence(new_sentence)
new_sentence_vectorized = loaded_vectorizer.transform([new_sentence_cleaned])
predicted_intent_encoded = loaded_model.predict(new_sentence_vectorized)
predicted_intent = loaded_label_encoder.inverse_transform(predicted_intent_encoded)
print('\n---------------------------------------------------------------------')
print(f"New data: {new_sentence}")
print(f"Predicted intent: {predicted_intent[0]}")

