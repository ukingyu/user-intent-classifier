import os
import pandas as pd
import random

DATA_FOLDER = 'data'
# intent_to_augment = ['inquiry_order', 'track_refund', 'change_shipping_address']

# Related words for each intent category
intent_keywords = {
    "cancel_order": ["cancel", "terminate", "stop", "discontinue"],
    "change_shipping_address": ["change", "update", "modify", "new address"],
    "check_policy": ["policy", "rules", "guidelines", "terms"],
    "contact_human_agent": ["human", "agent", "real person", "support staff"],
    "inquiry_order": ["status", "details", "information", "progress"],
    "track_order": ["track", "tracking", "shipment", "delivery"],
    "track_refund": ["refund", "reimbursement", "money back", "return"],
    "general": ["please", "kindly", "urgently", "quickly", "recent", "latest"]
}

def get_inserted_words(intent):
    inserted_words = set()
    if intent in intent_keywords:
        inserted_words.update(intent_keywords[intent])
    inserted_words.update(intent_keywords["general"])

    return list(inserted_words)

def augment_sentences(sentence, intent, num_variations=2):
    words = sentence.split()
    modified_sentences = set()
    inserted_words = get_inserted_words(intent)
    if not inserted_words:
        return []

    for _ in range(num_variations):
        new_sentence = words[:]
        inserted_word = random.choice(inserted_words)
        insert_position = random.randint(0, len(words))
        new_sentence.insert(insert_position, inserted_word)
        modified_sentences.add(" ".join(new_sentence))

    return list(modified_sentences)

def main():
    # Read input csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    input_file = os.path.join(parent_dir, "s02_replace_synonym", DATA_FOLDER, "data_after_syns.csv")
    df = pd.read_csv(input_file)

    df_to_augment = df
    # df_to_augment = df[df['intent'].isin(intent_to_augment)]

    # Process each sentence
    augmented_data = []
    for sentence, intent in zip(df_to_augment["utterance"], df_to_augment["intent"]):
        augmented_sentences = augment_sentences(sentence, intent)
        for new_sentence in augmented_sentences:
            augmented_data.append({"utterance": new_sentence, "intent": intent})

    # Add to original data
    df_augmented = pd.DataFrame(augmented_data)
    df_combined = pd.concat([df, df_augmented], ignore_index=True)

    # Export
    output_to_augment = os.path.join(script_dir, DATA_FOLDER, "data_to_augment.csv")
    output_augmented = os.path.join(script_dir, DATA_FOLDER, "data_augmented.csv")
    output_combined = os.path.join(script_dir, DATA_FOLDER, "data_after_insert.csv")
    df_to_augment.to_csv(output_to_augment, index=False)
    df_augmented.to_csv(output_augmented, index=False)
    df_combined.to_csv(output_combined, index=False)
    print(f"Augmented data has been saved")

if __name__ == "__main__":
    main()