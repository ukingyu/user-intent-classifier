import os
import pandas as pd
import nltk
from nltk.corpus import wordnet
import random

DATA_FOLDER = 'data'
# intent_to_augment = ['inquiry_order', 'track_refund', 'change_shipping_address']

# E-commerce synonym data
ecommerce_synonyms = {
    "order": ["purchase", "transaction", "request"],
    "status": ["progress", "update", "state"],
    "check": ["verify", "inspect", "look into"],
    "info": ["information", "details", "data"],
    "tell": ["inform", "let me know", "share"],
    "know": ["understand", "be aware", "find out"],
    "want": ["wish", "desire", "need"],
    "need": ["require", "must have", "seek"],
}

def get_synonyms(word):
    synonyms = set()
    
    # # Search WordNet for synonyms
    # for syn in wordnet.synsets(word):
    #     for lemma in syn.lemmas():
    #         synonyms.add(lemma.name().replace("_", " "))  # 替换下划线为空格
            
    # Search ecommerce synonyms
    if word in ecommerce_synonyms:
        synonyms.update(ecommerce_synonyms[word])
    
    # Remove the original word from the synonyms set
    synonyms.discard(word)
    
    return list(synonyms)

def augment_sentences(sentence, num_variations=3):
    words = sentence.split()
    modified_sentences = set()

    for _ in range(num_variations):
        new_sentence = words[:]
        for i, word in enumerate(new_sentence):
            synonyms = get_synonyms(word.lower())
            if synonyms:
                new_sentence[i] = random.choice(synonyms)
        modified_sentences.add(" ".join(new_sentence))

    return list(modified_sentences)

def main():
    # Set path of nltk_data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nltk_data_path = os.path.join(script_dir, "nltk_data")
    nltk.data.path.append(nltk_data_path)

    # Read input csv
    parent_dir = os.path.dirname(script_dir)
    input_file = os.path.join(parent_dir, "s01_generate_data", "data_after_gen.csv")
    df = pd.read_csv(input_file)

    df_to_augment = df
    # df_to_augment = df[df['intent'].isin(intent_to_augment)]

    # Process each sentence
    augmented_data = []
    for sentence, intent in zip(df_to_augment["utterance"], df_to_augment["intent"]):
        augmented_sentences = augment_sentences(sentence)
        for new_sentence in augmented_sentences:
            augmented_data.append({"utterance": new_sentence, "intent": intent})

    # Add to original data
    df_augmented = pd.DataFrame(augmented_data)
    df_combined = pd.concat([df, df_augmented], ignore_index=True)

    # Export
    output_to_augment = os.path.join(script_dir, DATA_FOLDER, "data_to_augment.csv")
    output_augmented = os.path.join(script_dir, DATA_FOLDER, "data_augmented.csv")
    output_combined = os.path.join(script_dir, DATA_FOLDER, "data_after_syns.csv")
    df_to_augment.to_csv(output_to_augment, index=False)
    df_augmented.to_csv(output_augmented, index=False)
    df_combined.to_csv(output_combined, index=False)
    print(f"Augmented data has been saved")

if __name__ == "__main__":
    main()
