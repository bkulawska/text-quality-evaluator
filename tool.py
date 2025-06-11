import string
import math
import numpy as np
import spacy
import textstat
from textblob import TextBlob
import language_tool_python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class TextQualityEvaluator:

    def __init__(self):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)

        self.nlp = spacy.load("en_core_web_sm")

        self.stopwords_eng = stopwords.words('english')

        self.language_tool = language_tool_python.LanguageTool('en-US')

        self.perplexity_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.perplexity_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        self.lcs_model = BertModel.from_pretrained("bert-base-uncased")
        self.lcs_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.metrics_sig = {
            'GER': 3,
            'Perplexity': 3,
            'Cosine Similarity': 4,
            'Flesch-Kincaid': 1,
            'Gunning Fog Index': 2,
            'ADD': 3,
            'PTD': 2,
            'TTR': 3,
            'Lexical Density': 3,
            'Sentiment': 3,
            'quality_score': 2,
            'suitability_score': 2
        }

        self.audience_profiles = {
            "children": {
                "Flesch-Kincaid": {"ideal": (80, 100), "weight": 0.2},    # Very easy to read
                "Gunning Fog Index": {"ideal": (1, 6), "weight": 0.2},    # Elementary school level
                "PTD": {"ideal": (2, 4), "weight": 0.1},                  # Simple sentence structures
                "ADD": {"ideal": (1, 2), "weight": 0.1},                  # Short dependency distances
                "Lexical Density": {"ideal": (0.2, 0.5), "weight": 0.2},  # Lower lexical density
                "TTR": {"ideal": (0.2, 0.5), "weight": 0.2},              # Limited vocabulary diversity
                "Sentiment": {"ideal": (0.2, 1), "weight": 0.15},         # Positive sentiment
            },
            "teenagers": {
                "Flesch-Kincaid": {"ideal": (60, 80), "weight": 0.2},     # Fairly easy to read
                "Gunning Fog Index": {"ideal": (7, 12), "weight": 0.2},   # Middle/high school level
                "PTD": {"ideal": (3, 5), "weight": 0.1},                  # Moderate sentence complexity
                "ADD": {"ideal": (1.5, 2.5), "weight": 0.1},              # Moderate dependency distances
                "Lexical Density": {"ideal": (0.3, 0.6), "weight": 0.2},  # Moderate lexical density
                "TTR": {"ideal": (0.3, 0.6), "weight": 0.2},              # Moderate vocabulary diversity
                "Sentiment": {"ideal": (0, 1), "weight": 0.15},           # Neutral to positive sentiment
            },
            "adults with basic or secondary education": {
                "Flesch-Kincaid": {"ideal": (50, 70), "weight": 0.2},     # Standard reading level
                "Gunning Fog Index": {"ideal": (9, 13), "weight": 0.2},   # High school level
                "PTD": {"ideal": (4, 6), "weight": 0.1},                  # More complex sentences
                "ADD": {"ideal": (1.5, 3), "weight": 0.1},                # Longer dependency distances
                "Lexical Density": {"ideal": (0.3, 0.7), "weight": 0.2},  # Higher lexical density
                "TTR": {"ideal": (0.3, 0.7), "weight": 0.2},              # More rich vocabulary
                "Sentiment": {"ideal": (-1, 1), "weight": 0.15},          # Negative to positive sentiment
            },
            "adults with higher education": {
                "Flesch-Kincaid": {"ideal": (0, 50), "weight": 0.2},      # Difficult reading level
                "Gunning Fog Index": {"ideal": (13, 22), "weight": 0.2},  # College level and higher
                "PTD": {"ideal": (5, 10), "weight": 0.1},                 # Very complex sentence structures
                "ADD": {"ideal": (2.5, 5), "weight": 0.1},                # Very long dependency distances
                "Lexical Density": {"ideal": (0.5, 1), "weight": 0.2},    # Very high lexical density
                "TTR": {"ideal": (0.5, 1), "weight": 0.2},                # Very rich vocabulary
                "Sentiment": {"ideal": (-1, 1), "weight": 0.15},          # Negative to positive sentiment
            },
            "adults learning language": {
                "Flesch-Kincaid": {"ideal": (70, 90), "weight": 0.2},     # Easy to read
                "Gunning Fog Index": {"ideal": (4, 9), "weight": 0.2},    # Simplified level
                "PTD": {"ideal": (2, 5), "weight": 0.1},                  # Moderate sentence complexity
                "ADD": {"ideal": (1.5, 2.5), "weight": 0.1},              # Moderate dependency distances
                "Lexical Density": {"ideal": (0.2, 0.6), "weight": 0.2},  # Moderate lexical density
                "TTR": {"ideal": (0.3, 0.6), "weight": 0.2},              # Moderate vocabulary diversity
                "Sentiment": {"ideal": (0, 1), "weight": 0.15},           # Neutral to positive sentiment
            },
            "seniors": {
                "Flesch-Kincaid": {"ideal": (50, 70), "weight": 0.2},     # Standard reading level
                "Gunning Fog Index": {"ideal": (10, 13), "weight": 0.2},  # High school level
                "PTD": {"ideal": (3, 6), "weight": 0.1},                  # Moderate sentence complexity
                "ADD": {"ideal": (2, 3), "weight": 0.1},                  # Moderate dependency distances
                "Lexical Density": {"ideal": (0.3, 0.7), "weight": 0.2},  # Higher lexical density
                "TTR": {"ideal": (0.4, 0.7), "weight": 0.2},              # Rich vocabulary
                "Sentiment": {"ideal": (0, 1), "weight": 0.15},           # Neutral to positive sentiment
            }
        }

        self.quality_metrics = {
            "GER": {"ideal": (0, 0.05), "weight": 0.25},                  # Few grammatical errors
            "Perplexity": {"ideal": (1, 40), "weight": 0.25},             # Natural language flow
            "Cosine Similarity": {"ideal": (0.7, 1.0), "weight": 0.5},    # High coherence
        }

    # Grammatical Error Rate (scores from 0 to 1)
    def calculate_ger(self, text):
        errors = self.language_tool.check(text)
        words = len(text.split())
        error_rate = len(errors) / words if words > 0 else 0
        return error_rate

    # Perplexity (scores from 1 to infinity)
    def calculate_perplexity(self, text):
        encodings = self.perplexity_tokenizer(text, return_tensors="pt")
        max_length = self.perplexity_model.config.n_positions
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
                outputs = self.perplexity_model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
                
            nlls.append(neg_log_likelihood)
        
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()

    # Cosine Similarity (scores from 0 to 1)
    def calculate_cosine_similarity(self, text):
        def get_sentence_embedding(sentence, tokenizer, model):
            inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            sentence_embedding = outputs.last_hidden_state.mean(dim=1)
            return sentence_embedding

        sentences = nltk.sent_tokenize(text)

        embeddings = [get_sentence_embedding(sentence, self.lcs_tokenizer, self.lcs_model) for sentence in sentences]

        coherence_scores = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity(embeddings[i], embeddings[i + 1])
            coherence_scores.append(similarity[0][0])

        return float(np.mean(coherence_scores))

    # Flesch-Kincaid Readability Score (scores from 0 to 100)
    def calculate_flesch_kincaid(self, text):
        return textstat.flesch_reading_ease(text)

    # Gunning Fog Index (scores from 0 to infinity)
    def calculate_fog_index(self, text):
        return textstat.gunning_fog(text)

    # Average Dependency Distance (scores from 0 to infinity)
    def calculate_add(self, text):
        doc = self.nlp(text)
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
    def calculate_ptd(self, text):
        doc = self.nlp(text)
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
    def calculate_ttr(self, text):
        words = word_tokenize(text.lower())
        unique_words = set(words)
        return len(unique_words) / len(words) if words else 0

    # Lexical Density (scores from 0 to 1)
    def calculate_lexical_density(self, text):
        words = word_tokenize(text.lower())
        meaningful_words = [word for word in words if word not in self.stopwords_eng and word not in string.punctuation]
        return len(meaningful_words) / len(words) if len(words) > 0 else 0

    # Sentiment Score (scores from -1 to 1)
    def calculate_sentiment(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def format_metric(self, x, name):
        if x == 0:
            return 0
        sig = self.metrics_sig.get(name, 3)
        return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

    def calculate_metrics(self, text):
        result = {
            "GER": self.format_metric(self.calculate_ger(text), "GER"),
            "Perplexity": self.format_metric(self.calculate_perplexity(text), "Perplexity"),
            "Cosine Similarity": self.format_metric(self.calculate_cosine_similarity(text), "Cosine Similarity"),
            "Flesch-Kincaid": self.format_metric(self.calculate_flesch_kincaid(text), "Flesch-Kincaid"),
            "Gunning Fog Index": self.format_metric(self.calculate_fog_index(text), "Gunning Fog Index"),
            "ADD": self.format_metric(self.calculate_add(text), "ADD"),
            "PTD": self.format_metric(self.calculate_ptd(text), "PTD"),
            "TTR": self.format_metric(self.calculate_ttr(text), "TTR"),
            "Lexical Density": self.format_metric(self.calculate_lexical_density(text), "Lexical Density"),
            "Sentiment": self.format_metric(self.calculate_sentiment(text), "Sentiment"),
        }
        return result

    def assess_overall_quality(self, metrics):
        quality_score = 0
        quality_weight = 0
        quality_issues = []
        
        for metric, config in self.quality_metrics.items():
            if metric in metrics:
                min_val, max_val = config["ideal"]
                weight = config["weight"]
                value = metrics[metric]

                # Calculate how well the metric fits within the ideal range
                
                # For GER and PP, lower is better
                if metric == "GER" or metric == "Perplexity":
                    if value > max_val:
                        fit = max(0, 1 - (value - max_val) / max_val)
                    else:
                        fit = 1

                # For Cosine Similarity, higher is better
                else:
                    if value < min_val:
                        fit = max(0, 1 - (min_val - value) / (max_val - min_val))
                    else:
                        fit = 1
                
                quality_score += fit * weight
                quality_weight += weight

                if fit < 1:
                    if metric == "GER" or metric == "Perplexity":
                        quality_issues.append(f"{metric} is too high (value: {value}, ideal: below {max_val})")
                    else:
                        quality_issues.append(f"{metric} is too low (value: {value}, ideal: above {min_val})")
        
        quality_score = self.format_metric(quality_score / quality_weight, "quality_score")

        if quality_score < 0.7:
            quality_assessment = "The text has significant quality issues that should be addressed."
        elif quality_score < 0.9:
            quality_assessment = "The text has moderate quality issues that could be improved."
        else:
            quality_assessment = "The text has good overall quality."

        result = {
            "quality_score": quality_score,
            "quality_issues": quality_issues,
            "quality_assessment": quality_assessment
        }

        return result

    def assess_target_audience_suitability(self, metrics, target_audience): 
        suitability_score = 0
        suitability_weight = 0
        suitability_issues = []
        
        for metric, config in self.audience_profiles[target_audience].items():
            if metric in metrics:
                min_val, max_val = config["ideal"]
                weight = config["weight"]
                value = metrics[metric]
                
                # Calculate how well the metric fits within the ideal range
                if value < min_val:
                    fit = max(0, 1 - (min_val - value) / (max_val - min_val))
                elif value > max_val:
                    fit = max(0, 1 - (value - max_val) / (max_val - min_val))
                else:
                    fit = 1
                
                suitability_score += fit * weight
                suitability_weight += weight

                if fit < 1:
                    if value < min_val:
                        suitability_issues.append(f"{metric} is too low for {target_audience} (value: {value}, ideal: {min_val}-{max_val})")
                    else:
                        suitability_issues.append(f"{metric} is too high for {target_audience} (value: {value}, ideal: {min_val}-{max_val})")

        suitability_score = self.format_metric(suitability_score / suitability_weight, "suitability_score")

        if suitability_score < 0.7:
            suitability_assessment = f"The text is not well-suited for the {target_audience} audience."
        elif suitability_score < 0.9:
            suitability_assessment = f"The text is moderately suited for the {target_audience} audience, some adjustments could be made."
        else:
            suitability_assessment = f"The text is well-suited for the {target_audience} audience."

        result = {
            "suitability_score": suitability_score,
            "suitability_issues": suitability_issues,
            "suitability_assessment": suitability_assessment
        }
        
        return result

    def evaluate_text(self, text, target_audience, label=None):
        metrics = self.calculate_metrics(text)
        quality_result = self.assess_overall_quality(metrics)
        suitability_result = self.assess_target_audience_suitability(metrics, target_audience)

        result = {
            "label": label if label else "",
            "target_audience": target_audience,
            "metrics": metrics,
            "quality_score": quality_result['quality_score'],
            "quality_issues": quality_result['quality_issues'],
            "quality_assessment": quality_result['quality_assessment'],
            "suitability_score": suitability_result['suitability_score'],
            "suitability_issues": suitability_result['suitability_issues'],
            "suitability_assessment": suitability_result['suitability_assessment']
        }

        return result
