import json
import random
import spacy
from nltk.corpus import wordnet
from datasets import load_dataset
import nltk

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Make sure to download WordNet if it's not already downloaded
nltk.download("wordnet")

# Load the SNLI dataset from Hugging Face
snli_data = load_dataset("snli", split="train")

# Define helper function to get antonyms using NLTK WordNet
def get_antonym(word):
    antonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    return antonyms[0] if antonyms else None

# Function to generate contrast set using spaCy and NLTK
def generate_contrast_set(example):
    contrast_example = example.copy()
    doc = nlp(example["hypothesis"])

    # Apply transformations based on label
    if contrast_example["label"] == 0:  # Entailment
        # Convert entailment to contradiction
        new_hypothesis = []
        for token in doc:
            # Introduce negation where possible
            if token.dep_ == "ROOT":
                antonym = get_antonym(token.lemma_)
                new_hypothesis.append(antonym if antonym else "not " + token.text)
            else:
                new_hypothesis.append(token.text)
        contrast_example["hypothesis"] = " ".join(new_hypothesis)
        contrast_example["label"] = 2  # Contradiction

    elif contrast_example["label"] == 2:  # Contradiction
        # Convert contradiction to entailment by removing negations
        new_hypothesis = []
        for token in doc:
            # Remove 'not' or 'n't' to change meaning to entailment
            if token.text.lower() in ["not", "n't"]:
                continue
            new_hypothesis.append(token.text)
        contrast_example["hypothesis"] = " ".join(new_hypothesis)
        contrast_example["label"] = 0  # Entailment

    elif contrast_example["label"] == 1:  # Neutral
        # Change neutral by slightly altering context (e.g., changing quantifiers)
        new_hypothesis = []
        for token in doc:
            if token.lemma_ == "some":
                new_hypothesis.append("all")  # Example of quantifier change
            else:
                new_hypothesis.append(token.text)
        contrast_example["hypothesis"] = " ".join(new_hypothesis)
        contrast_example["label"] = 2  # Change to contradiction for variety

    return contrast_example

# Generate contrast sets for each example in the dataset
contrast_sets = []
for x in range(300000):
    example = snli_data[x]
    print("example check...", x)
    # Only generate contrast sets if the label is valid (0, 1, 2)
    if example["label"] in [0, 1, 2]:
        contrast_example = generate_contrast_set(example)
        contrast_sets.append(contrast_example)

# Combine original and contrast examples
output_data = contrast_sets

# Save the output as a JSON file
output_file = 'snli_contrast_sets_antonyms.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Contrast sets have been saved to '{output_file}'")