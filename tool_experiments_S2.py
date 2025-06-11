import csv
import json
from tool import TextQualityEvaluator

evaluator = TextQualityEvaluator()

groups = ["children", "teenagers", "adults with basic or secondary education", "adults with higher education", "adults learning language", "seniors"]

def process_csv(input_file, output_file):
    results = []
    temps = []
    
    with open(input_file, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        for row in csvreader:
            if len(row) == 2:
                exclude_group, text = row
                for group in groups:
                    if exclude_group != group:
                        result = evaluator.evaluate_text(text, group, "text suitable for: " + exclude_group)
                        results.append(result)
    

    for result in results:
        temp = {}
        temp['label'] = result['label']
        temp['target_audience'] = result['target_audience']
        temp['suitability_score'] = result['suitability_score']
        temp['suitability_issues'] = result['suitability_issues']
        temp['suitability_assessment'] = result['suitability_assessment']
        temps.append(temp)
    
    with open(output_file, mode='w', encoding='utf-8') as jsonfile:
        json.dump(temps, jsonfile, indent=4)

input_csv_file = 'dataset_S.csv'
output_json_file = 'output_S2.json'

process_csv(input_csv_file, output_json_file)