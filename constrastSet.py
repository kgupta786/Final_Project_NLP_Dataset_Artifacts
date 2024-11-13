import json
from datasets import load_dataset

# Define a function to save the dataset in JSON format
def save_contrast_dataset(dataset, dataset_name, output_file):
    # Process dataset and extract premise, hypothesis, and label
    output_data = []
    
    for entry in dataset:
        premise = entry.get("premise") or entry.get("sentence1")
        hypothesis = entry.get("hypothesis") or entry.get("sentence2")
        label = entry.get("label")
        
        # Skip examples without necessary fields
        if premise is None or hypothesis is None or label is None:
            continue
        
        # Append entry in desired format
        output_data.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label
        })
    
    # Save to a JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"{dataset_name} saved to {output_file}.")
# HANS Dataset
hans = load_dataset("hans", split="train", trust_remote_code=True)
save_contrast_dataset(hans, "HANS", "hans.json")

# [[14697, 0, 303], [13684, 0, 1316], [0, 0, 0]]
# {'eval_loss': 2.393868923187256, 'eval_model_preparation_time': 0.0011, 'eval_accuracy': 0.48989999294281006, 'eval_runtime': 94.9075, 'eval_samples_per_second': 316.097, 'eval_steps_per_second': 39.512}