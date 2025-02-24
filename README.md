# User Intent Classifier for E-commerce

This repository contains a pipeline for classifying user intents in an e-commerce context. The pipeline includes data collection, data augmentation, data cleaning, and model training using a Naive Bayes classifier. The goal is to accurately predict user intents based on their utterances.

## Overview

### 1. **Data Collection**
   - The initial dataset is collected and stored in the `data/` folder.
   - Files like `data_to_augment.csv` contain raw user utterances and corresponding intents.

### 2. **Data Augmentation**
   - **Synonym Replacement**: Replaces words in utterances with their synonyms to increase dataset diversity.
   - **Random Insertion**: Inserts random words into utterances to simulate variations in user input.
   - **Random Deletion**: Randomly deletes words from utterances to make the model robust to incomplete inputs.
   - **Random Swapping**: Swaps words in utterances to create more training examples.

### 3. **Data Cleaning**
   - The `clean_data.py` script processes the augmented data to remove noise, standardize text, and prepare it for training.
   - Cleaned data is saved as `data_after_clean.csv`.

### 4. **Model Training**
   - The `train_model.py` script trains a **Naive Bayes classifier** using TF-IDF vectorization.
   - The model is trained on cleaned data (`data_after_clean.csv`) and evaluated for accuracy.
   - Trained models and vectorizers are saved in the `model/` folder for future use.

### 5. **Prediction**
   - The trained model can predict user intents for new utterances. For example:
     ```python
     new_sentence = "how to cancel my order?"
     predicted_intent = model.predict(new_sentence)
     ```

## Saved Models
- `intent_classification_model.pkl`: Trained Naive Bayes model.
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer used for text transformation.
- `label_encoder.pkl`: Label encoder for mapping intents to numerical values.

## Evaluation
The model's performance is evaluated using accuracy and a classification report. Example output:
```
Accuracy: 0.9751015730805292
Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.99      0.98      4706
           1       1.00      1.00      1.00       938
           2       0.99      0.99      0.99      4843
           3       1.00      1.00      1.00      4479
           4       0.94      0.98      0.96      5329
           5       0.97      0.95      0.96      6531
           6       0.99      0.90      0.94      1971

    accuracy                           0.98     28797
   macro avg       0.98      0.97      0.98     28797
weighted avg       0.98      0.98      0.98     28797
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

