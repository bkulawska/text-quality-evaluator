from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import word_tokenize
from bert_score import score as bert_score
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import language_tool_python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import textstat
import numpy as np
import pandas as pd
import csv
import gruen
import string
import re
from transformers import BertTokenizer, BertModel
import bert_score
import spacy
from nltk.util import ngrams
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
import math
from textblob import TextBlob

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

nlp = spacy.load("en_core_web_sm")

stopwords_eng = stopwords.words('english')

language_tool = language_tool_python.LanguageTool('en-US')

perplexity_model = GPT2LMHeadModel.from_pretrained("gpt2")
perplexity_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

lcs_model = BertModel.from_pretrained("bert-base-uncased")
lcs_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Grammatical Error Rate (scores from 0 to infinity)
def calculate_ger(text):
    errors = language_tool.check(text)
    words = len(text.split())
    error_rate = len(errors) / words if words > 0 else 0
    return error_rate

# Perplexity (scores from 1 to infinity)
def calculate_perplexity(text):
    encodings = perplexity_tokenizer(text, return_tensors="pt")
    max_length = perplexity_model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = perplexity_model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

# Flesch-Kincaid Readability Score (scores from 0 to 100)
def calculate_flesch_kincaid(text):
    return textstat.flesch_reading_ease(text)

# Gunning Fog Index (scores from 0 to infinity)
def calculate_fog_index(text):
    return textstat.gunning_fog(text)

# Average Dependency Distance (scores from 0 to infinity)
def calculate_add(text):
    doc = nlp(text)
    total_distance = 0
    total_deps = 0
    
    for sent in doc.sents:
        sent_deps = 0
        sent_dist = 0
        for token in sent:
            if token.head != token and not token.is_punct:
                sent_deps += 1
                sent_dist += abs(token.i - token.head.i)
        if sent_deps > 0:
            total_deps += sent_deps
            total_distance += sent_dist
    
    return total_distance / total_deps if total_deps > 0 else 0

# Parse Tree Depth (scores from 0 to infinity)
def calculate_ptd(text):
    doc = nlp(text)
    max_depths = []
    
    for sent in doc.sents:
        depths = {}
        for token in sent:
            if token.head == token:
                depths[token.i] = 0
            else:
                depths[token.i] = -1

        changed = True
        while changed:
            changed = False
            for token in sent:
                if token.head == token:
                    continue
                if depths[token.head.i] >= 0 and depths[token.i] == -1:
                    depths[token.i] = depths[token.head.i] + 1
                    changed = True

        if depths:
            max_depth = max(depths.values())
            max_depths.append(max_depth)

    return sum(max_depths) / len(max_depths) if max_depths else 0

# Type-Token Ratio (scores from 0 to 1)
def calculate_ttr(text):
    words = word_tokenize(text.lower())
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0

# Lexical Density (scores from 0 to 1)
def calculate_lexical_density(text):
    words = word_tokenize(text.lower())
    meaningful_words = [word for word in words if word not in stopwords_eng and word not in string.punctuation]
    return len(meaningful_words) / len(words) if len(words) > 0 else 0

# Local Coherence Score (scores from 0 to 1)
def calculate_local_coherence(text):
    def get_syntactic_role(token):
        """Determine syntactic role of a token"""
        if token.dep_ in ("nsubj", "nsubjpass"):
            return "S"  # Subject
        elif token.dep_ in ("dobj", "pobj", "iobj"):
            return "O"  # Object
        elif token.pos_ in ("NOUN", "PROPN", "PRON"):
            return "X"  # Other nominal mention
        return None

    def extract_entities(doc):
        """Extract entities and their syntactic roles from a document"""
        entities = []
        for token in doc:
            role = get_syntactic_role(token)
            if role:
                # Use lemma to normalize entity mentions
                entities.append((token.lemma_.lower(), role))
        return entities

    def build_entity_grid(parsed_sentences):
        """Build entity grid from parsed sentences"""
        # Extract entities and roles for each sentence
        sentence_entities = [extract_entities(doc) for doc in parsed_sentences]
        
        # Build a set of all unique entities
        all_entities = set()
        for sent_ents in sentence_entities:
            all_entities.update([ent[0] for ent in sent_ents])
        
        # Create the entity grid
        entity_grid = {}
        for entity in all_entities:
            entity_grid[entity] = []
            for sent_ents in sentence_entities:
                # Find this entity in the current sentence
                roles = [role for ent, role in sent_ents if ent == entity]
                # If multiple mentions, take the most prominent (S > O > X)
                if "S" in roles:
                    entity_grid[entity].append("S")
                elif "O" in roles:
                    entity_grid[entity].append("O")
                elif "X" in roles:
                    entity_grid[entity].append("X")
                else:
                    entity_grid[entity].append("-")  # Entity not present
        
        return entity_grid

    def extract_transitions(entity_grid):
        """Extract transitions from the entity grid"""
        transitions = []
        for entity, roles in entity_grid.items():
            for i in range(len(roles) - 1):
                transitions.append((roles[i], roles[i+1]))
        return transitions
    
    """
    Calculate Local Coherence Score using entity grid model with pre-trained transition probabilities.
    
    Args:
        text (str): The input text to evaluate
        
    Returns:
        float: Local coherence score between 0 and 1
    """
    # Handle edge cases
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return 1.0  # Perfect coherence for single sentence
    
    
    # Pre-trained transition probabilities from Barzilay & Lapata (2008)
    transition_probs = {
        ('S', 'S'): 0.37, ('S', 'O'): 0.17, ('S', 'X'): 0.08, ('S', '-'): 0.38,
        ('O', 'S'): 0.19, ('O', 'O'): 0.16, ('O', 'X'): 0.11, ('O', '-'): 0.54,
        ('X', 'S'): 0.13, ('X', 'O'): 0.11, ('X', 'X'): 0.09, ('X', '-'): 0.67,
        ('-', 'S'): 0.05, ('-', 'O'): 0.04, ('-', 'X'): 0.03, ('-', '-'): 0.88
    }
    
    # Parse each sentence
    parsed_sentences = [nlp(sent) for sent in sentences]
    
    # Build entity grid
    entity_grid = build_entity_grid(parsed_sentences)
    
    # Extract transitions
    transitions = extract_transitions(entity_grid)
    
    if not transitions:
        return 0.5  # Default score when no transitions found
    
    # Calculate coherence score as product of transition probabilities
    # Using log probabilities to avoid underflow
    log_score = sum(np.log(transition_probs.get(t, 0.01)) for t in transitions)
    
    # Normalize by number of transitions
    log_score /= len(transitions)
    
    # Convert back from log space and scale to [0,1]
    # We use a sigmoid-like transformation to map scores to [0,1]
    score = 1 / (1 + np.exp(-log_score - 2))  # Offset for better scaling
    
    return score

# GRUEN (scores from 0 to 1)
def calculate_gruen(text):
    return gruen.get_gruen(text)

# Sentiment Score (scores from -1 to 1)
def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

####################################################################

# Distinct-n (scores from 0 to 1)
def calculate_distinct_n(text, n=2):
    words = word_tokenize(text.lower())
    n_grams = list(ngrams(words, n))
    unique_ngrams = set(n_grams)
    return len(unique_ngrams) / len(n_grams) if len(n_grams) > 0 else 0

# Jaccard Similarity (scores from 0 to 1)
def calculate_jaccard_similarity(text):
    def _preprocess_sentence(sentence):
        sentence = sentence.lower()
        sentence = re.sub(f'[{string.punctuation}]', ' ', sentence)
        words = [word.strip() for word in sentence.split() 
                if word.strip() and word.strip() not in set(stopwords_eng)]
        return words

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return 1.0
    similarities = []
    for i in range(len(sentences) - 1):
        words_a = set(_preprocess_sentence(sentences[i]))
        words_b = set(_preprocess_sentence(sentences[i + 1]))

        intersection = words_a.intersection(words_b)
        union = words_a.union(words_b)

        if not union:
            similarity = 0.0
        else:
            similarity = len(intersection) / len(union)
        
        similarities.append(similarity)

    return sum(similarities) / len(similarities)

# Local Coherence Score (scores from 0 to 1)
def calculate_local_coherence_other(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return 1.0

    embeddings = []
    for sentence in sentences:
        inputs = lcs_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        outputs = lcs_model(**inputs)
        sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1)
        embeddings.append(sentence_embedding)
    
    coherence_scores = [torch.cosine_similarity(embeddings[i], embeddings[i+1]).item() for i in range(len(embeddings)-1)]
    return sum(coherence_scores) / len(coherence_scores)

# BLEU (scores from 0 to 1)
def calculate_bleu(generated, reference):
    return corpus_bleu([generated], [[reference]]).score / 100

# ROUGE (scores from 0 to 1)
def calculate_rouge(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(generated, reference)['rougeL'].fmeasure

# METEOR (scores from 0 to 1)
def calculate_meteor(generated, reference):
    generated_tokens = word_tokenize(generated)
    reference_tokens = word_tokenize(reference)
    return nltk.translate.meteor_score.meteor_score([reference_tokens], generated_tokens)

# BERTScore (scores from 0 to 1)
def calculate_bertscore(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return 0
    
    references = sentences[:-1]
    candidates = sentences[1:]
    P, R, F1 = bert_score.score(candidates, references, lang="en", model_type="microsoft/deberta-xlarge-mnli")
    return torch.mean(F1).item()

# Self-BLEU (scores from 0 to 1)
def calculate_self_bleu(text, n=4):
    sentences = nltk.sent_tokenize(text)
    scores = []
    for i, sent in enumerate(sentences):
        other_sentences = sentences[:i] + sentences[i+1:]
        reference = [nltk.word_tokenize(s) for s in other_sentences]
        candidate = nltk.word_tokenize(sent)
        score = sentence_bleu(reference, candidate, weights=[1/n]*n)
        scores.append(score)
    return np.mean(scores) if scores else 0

# Entropy (scores from 0 to infinity)
def calculate_entropy(text):
    words = nltk.word_tokenize(text)
    word_counts = Counter(words)
    total_words = len(words)
    entropy = -sum((count/total_words) * math.log2(count/total_words) for count in word_counts.values())
    return entropy

# FactCC (scores from 0 to 1)
def calculate_factcc(generated):
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    premise = generated
    hypothesis = "This text contains factual information."
    input_text = f"{premise}</s>{hypothesis}"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        factuality_score = predictions[:, 2].item()
    
    return factuality_score

# FEQA (scores from 0 to 1)
def calculate_feqa(generated):
    qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

    sentences = sent_tokenize(generated)
    questions = []
    for sent in sentences:
        words = sent.split()
        if len(words) > 3:
            question = f"What {' '.join(words[1:])}"
            questions.append(question)

    scores = []
    for question in questions:
        result = qa_pipeline(question=question, context=generated)
        scores.append(result['score'])

    return sum(scores) / len(scores) if scores else 0

####################################################################

file_path = 'dataset.csv'

generated = []
references = []

with open(file_path, mode='r', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter=';')

    for row in reader:
        if len(row) == 2:
            text1, text2 = row
            generated.append(text1)
            references.append(text2)

####################################################################

size = len(generated)

results = []

for i in range(0, size):
    gen = generated[i]
    ref = references[i]
    result = {
        "ID": i,
        
        #"BLEU": calculate_bleu(gen, ref),
        #"ROUGE": calculate_rouge(gen, ref),
        #"METEOR": calculate_meteor(gen, ref),

        #"FactCC": calculate_factcc(gen),
        #"FEQA": calculate_feqa(gen),

        "GER": calculate_ger(gen),

        "Perplexity": calculate_perplexity(gen),

        "Flesch-Kincaid": calculate_flesch_kincaid(gen),
        "Gunning Fog Index": calculate_fog_index(gen),
        "ADD": calculate_add(gen),
        "PTD": calculate_ptd(gen),

        "TTR": calculate_ttr(gen),
        "Lexical Density": calculate_lexical_density(gen),
        #"Distinct-n": calculate_distinct_n(gen),
        #"Self-BLEU": calculate_self_bleu(gen),
        #"Entropy": calculate_entropy(gen),

        "Local Coherence Score": calculate_local_coherence(gen),
        #"Jaccard Similarity": calculate_jaccard_similarity(gen),
        #"BERTScore": calculate_bertscore(gen),

        "GRUEN": calculate_gruen(gen),
        "Sentiment": calculate_sentiment(gen),
    }

    results.append(result)

####################################################################

df = pd.DataFrame(results)
df.to_csv("metrics_results.csv", index=False)

####################################################################
