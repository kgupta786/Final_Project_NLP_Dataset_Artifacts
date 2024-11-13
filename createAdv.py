# import nltk
# import spacy
# import json
# import random
# from datasets import load_dataset

# # Download necessary NLTK resources
# nltk.download('wordnet')
# nltk.download('punkt')

# from nltk.corpus import wordnet as wn
# from nltk.tokenize import word_tokenize

# # Load the English model for spaCy
# nlp = spacy.load("en_core_web_sm")

# # Load the SNLI dataset using Hugging Face datasets library
# snli_data = load_dataset("snli", split="train")

# # Define a function to get antonyms
# def get_antonym(word):
#     antonyms = []
#     for syn in wn.synsets(word):
#         for lemma in syn.lemmas():
#             if lemma.antonyms():
#                 antonyms.append(lemma.antonyms()[0].name())
#     return random.choice(antonyms) if antonyms else None

# # Define a function to get a synonym
# def get_synonym(word):
#     synonyms = [lemma.name() for syn in wn.synsets(word) for lemma in syn.lemmas() if lemma.name() != word]
#     return random.choice(synonyms) if synonyms else None

# # Function to create adversarial example by adding negation or swapping words
# def create_adversarial_hypothesis(hypothesis):
#     tokens = word_tokenize(hypothesis)
    
#     # Try to add a negation or swap with an antonym
#     adversarial_hypothesis = []
#     for token in tokens:
#         antonym = get_antonym(token)
#         synonym = get_synonym(token)
        
#         # 50% chance to use an antonym or negate
#         if antonym and random.random() < 0.5:
#             adversarial_hypothesis.append(antonym)
#         elif token.lower() not in ['no', 'not'] and random.random() < 0.3:
#             adversarial_hypothesis.append("not")
#             adversarial_hypothesis.append(token)
#         elif synonym and random.random() < 0.2:
#             adversarial_hypothesis.append(synonym)
#         else:
#             adversarial_hypothesis.append(token)
    
#     return ' '.join(adversarial_hypothesis)

# # Create adversarial dataset
# adversarial_data = []
# for x in range(300000):
#     print("Example: ", x)
#     row = snli_data[x]
#     if(row['premise'] == "A few people in a restaurant setting, one of them is drinking orange juice."):
#         print(row['hypothesis'])
#         break
#     adversarial_example = {
#         "premise": row['premise'],
#         "label": row['label'],
#         "hypothesis": create_adversarial_hypothesis(row['hypothesis'])
#     }
#     adversarial_data.append(adversarial_example)

# # Save the adversarial dataset to a JSON file
# with open('adversarial_snli.json', 'w') as f:
#     json.dump(adversarial_data, f, indent=4)

import json
from datasets import load_dataset

# Load the ANLI dataset (you can specify the split as 'train', 'dev', or 'test')
anli_dataset = load_dataset("anli", split="train_r2")
print("Length: ", len(anli_dataset))
# Process the data to store only the 'premise', 'hypothesis', and 'label'
anli_examples = []

# Iterate through the dataset and prepare the data
for example in anli_dataset:
    anli_example = {
        "premise": example['premise'],
        "hypothesis": example['hypothesis'],
        "label": example['label']
    }
    anli_examples.append(anli_example)

# Save the processed examples to a JSON file
with open("anli_dataset.json", "w") as f:
    json.dump(anli_examples, f, indent=4)

print(f"ANLI dataset saved to 'anli_dataset.json'.")


# [[7073, 3895, 3480], [2509, 12337, 6113], [2480, 2239, 5334]]