import csv
import json
from tool import TextQualityEvaluator

evaluator = TextQualityEvaluator()

def process_csv(input_file, output_file):
    results = []
    temps = []
    
    with open(input_file, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        for label, text in csvreader:
            result = evaluator.evaluate_text(text, "adults with basic or secondary education", label)
            results.append(result)

    for result in results:
        temp = {}
        temp['label'] = result['label']
        temp['metrics'] = result['metrics']
        temp['quality_score'] = result['quality_score']
        temp['quality_issues'] = result['quality_issues']
        temp['quality_assessment'] = result['quality_assessment']
        temps.append(temp)
    
    with open(output_file, mode='w', encoding='utf-8') as jsonfile:
        json.dump(temps, jsonfile, indent=4)

input_csv_file = 'dataset_Q.csv'
output_json_file = 'output_Q.json'

process_csv(input_csv_file, output_json_file)