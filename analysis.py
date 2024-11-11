import json

# Path to your .jsonl file
file_path = 'eval_output/eval_predictions.jsonl'

# Open the file and read each line as a separate JSON object
analysed_output_error = []
ct=0
with open(file_path, 'r') as f:
    for line in f:
        # Parse the line as JSON
        json_object = json.loads(line.strip())
        # Process or print the JSON object
        # print(json_object)
        if(json_object['label'] != json_object['predicted_label']):
            ct+=1
            analysed_output_error.append(json_object)
        # break
output_file = 'ant_analysed_output.json'
print("Count: ", ct)
with open(output_file, 'w') as f:
    json.dump(analysed_output_error, f, indent=4)