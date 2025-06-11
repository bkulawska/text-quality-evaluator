import csv
import json
from tool import TextQualityEvaluator

evaluator = TextQualityEvaluator()

def process_csv(input_file, output_file):
    results = []
    
    with open(input_file, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        for row in csvreader:
            if len(row) == 2:
                group, text = row
                result = evaluator.evaluate_text(text, group, "text suitable for: " + group)
                results.append(result)
    
    with open(output_file, mode='w', encoding='utf-8') as jsonfile:
        json.dump(results, jsonfile, indent=4)

input_csv_file = 'dataset_S.csv'
output_json_file = 'output_S1.json'

process_csv(input_csv_file, output_json_file)