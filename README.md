# Text Quality Evaluator

## Installation

```bash
pip install -r requirements.txt
```

After installing the requirements, you'll also need to download the spaCy English language model:

```bash
python -m spacy download en_core_web_sm
```

## Usage

```python
from tool import TextQualityEvaluator

evaluator = TextQualityEvaluator()

# Evaluate text for a specific audience
text = "Your text here..."
result = evaluator.evaluate_text(text, "adults with basic or secondary education")
print(result)
```

## Examples

Evaluates a simple informative text about electric cars for adults with basic education:
```bash
python tool_experiments_Q.py
```
Analyzes a complex literary text (from Shakespeare) for adults with higher education:

```bash
python tool_experiments_Q.py
```

Processes a CSV file with labeled texts, evaluating quality for a specific audience:
```bash
python tool_experiments_Q.py
```
Evaluates texts for their intended target audiences:

```bash
python tool_experiments_S1.py
```
Cross-evaluates texts against multiple audiences (excluding their intended audience):

```bash
python tool_experiments_S2.py
```

## Testing

Running unit tests:

```bash
python tool_unit_tests.py
```

## Technical Details

The evaluator uses several NLP libraries and models:
- spaCy for linguistic analysis
- NLTK for tokenization and text processing
- TextBlob for sentiment analysis
- GPT-2 for perplexity calculation
- BERT for semantic coherence analysis
- Language Tool for grammatical error detection