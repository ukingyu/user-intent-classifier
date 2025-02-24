import os
import pandas as pd
import random

DATA_FOLDER = 'data'
# intent_to_augment = ['inquiry_order', 'track_refund', 'change_shipping_address']

def augment_sentences(sentence, min_swap=1, max_swap=2, num_variations=2):
    words = sentence.split()
    if len(words) <= 1:
        return []
    
    modified_sentences = set()
    for _ in range(num_variations):
        new_sentence = words[:]
        num_swaps = random.randint(min_swap, max_swap)
        for _ in range(num_swaps):
            idx1, idx2 = random.sample(range(len(words)), 2)
            new_sentence[idx1], new_sentence[idx2] = new_sentence[idx2], new_sentence[idx1]
        modified_sentences.add(" ".join(new_sentence))

    return list(modified_sentences)

def main():
    # Read input csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    input_file = os.path.join(parent_dir, "s04_random_delete", DATA_FOLDER, "data_after_delete.csv")
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
    output_combined = os.path.join(script_dir, DATA_FOLDER, "data_after_swap.csv")
    df_to_augment.to_csv(output_to_augment, index=False)
    df_augmented.to_csv(output_augmented, index=False)
    df_combined.to_csv(output_combined, index=False)
    print(f"Augmented data has been saved")

if __name__ == "__main__":
    main()
