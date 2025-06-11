import unittest
from tool import TextQualityEvaluator
import numpy as np

class TestTextMetrics(unittest.TestCase):

    def setUp(self):
        self.metrics = TextQualityEvaluator()

    def test_calculate_ger(self):
        text_with_errors = "This are a test sentence with error. It not make sense. The cat chase mouse. They was happy. She go school every day. He have a car. We is friends."
        text_without_errors = "This is a test sentence without errors. It makes sense. The cat chased the mouse. They were happy. She goes to school every day. He has a car. We are friends."
        
        ger_with_errors = self.metrics.calculate_ger(text_with_errors)
        ger_without_errors = self.metrics.calculate_ger(text_without_errors)
        
        self.assertGreater(ger_with_errors, ger_without_errors)

    def test_calculate_perplexity(self):
        simple_text = "The cat sat on the mat. It was a sunny day. The dog barked at the mailman. Birds were singing in the trees. Children played in the park. The sky was clear and blue. Everyone was happy."
        complex_text = "Incomprehensibilities are often found in convoluted texts. The juxtaposition of disparate elements creates a cacophony of confusion. As the narrative unfolds, the labyrinthine structure becomes apparent. Readers must navigate through a plethora of esoteric references. The syntax is dense, requiring careful analysis. Metaphors abound, adding layers of meaning. Ultimately, the text challenges conventional understanding."
        
        perplexity_simple = self.metrics.calculate_perplexity(simple_text)
        perplexity_complex = self.metrics.calculate_perplexity(complex_text)
        
        self.assertLess(perplexity_simple, perplexity_complex)

    def test_calculate_cosine_similarity(self):
        coherent_text = "The cat chased the mouse. It caught it and played with it, the mouse was scared but unharmed. The cat eventually let it go and the mouse ran back to its hole. The cat watched the hole for a while. Then it lost interest and walked away."
        incoherent_text = "The cat jumped over the fence. A bright star twinkled in the night sky. She found a rare coin on the sidewalk. The concert tickets sold out in minutes. A gentle breeze rustled the autumn leaves. The book was published in three different languages. He painted the room a vibrant shade of blue. The train arrived at the station five minutes early. A new bakery opened on the corner street. The dog barked at the passing car."
        
        coherence_coherent = self.metrics.calculate_cosine_similarity(coherent_text)
        coherence_incoherent = self.metrics.calculate_cosine_similarity(incoherent_text)
        
        self.assertGreater(coherence_coherent, coherence_incoherent)

    def test_calculate_flesch_kincaid(self):
        easy_text = "The cat sat on the mat. It was a sunny day. The dog barked at the mailman. Birds were singing in the trees. Children played in the park. The sky was clear and blue. Everyone was happy."
        difficult_text = "Incomprehensibilities are often found in convoluted texts. The juxtaposition of disparate elements creates a cacophony of confusion. As the narrative unfolds, the labyrinthine structure becomes apparent. Readers must navigate through a plethora of esoteric references. The syntax is dense, requiring careful analysis. Metaphors abound, adding layers of meaning. Ultimately, the text challenges conventional understanding."
        
        fk_easy = self.metrics.calculate_flesch_kincaid(easy_text)
        fk_difficult = self.metrics.calculate_flesch_kincaid(difficult_text)
        
        self.assertGreater(fk_easy, fk_difficult)

    def test_calculate_fog_index(self):
        easy_text = "The cat sat on the mat. It was a sunny day. The dog barked at the mailman. Birds were singing in the trees. Children played in the park. The sky was clear and blue. Everyone was happy."
        difficult_text = "Incomprehensibilities are often found in convoluted texts. The juxtaposition of disparate elements creates a cacophony of confusion. As the narrative unfolds, the labyrinthine structure becomes apparent. Readers must navigate through a plethora of esoteric references. The syntax is dense, requiring careful analysis. Metaphors abound, adding layers of meaning. Ultimately, the text challenges conventional understanding."
        
        fog_easy = self.metrics.calculate_fog_index(easy_text)
        fog_difficult = self.metrics.calculate_fog_index(difficult_text)
        
        self.assertLess(fog_easy, fog_difficult)

    def test_calculate_add(self):
        simple_text = "The cat sat on the mat. It was a sunny day. The dog barked at the mailman. Birds were singing in the trees. Children played in the park. The sky was clear and blue. Everyone was happy."
        complex_text = "Incomprehensibilities are often found in convoluted texts. The juxtaposition of disparate elements creates a cacophony of confusion. As the narrative unfolds, the labyrinthine structure becomes apparent. Readers must navigate through a plethora of esoteric references. The syntax is dense, requiring careful analysis. Metaphors abound, adding layers of meaning. Ultimately, the text challenges conventional understanding."
        
        add_simple = self.metrics.calculate_add(simple_text)
        add_complex = self.metrics.calculate_add(complex_text)
        
        self.assertLess(add_simple, add_complex)

    def test_calculate_ptd(self):
        simple_text = "The cat sat on the mat. It was a sunny day. The dog barked at the mailman. Birds were singing in the trees. Children played in the park. The sky was clear and blue. Everyone was happy."
        complex_text = "Incomprehensibilities are often found in convoluted texts. The juxtaposition of disparate elements creates a cacophony of confusion. As the narrative unfolds, the labyrinthine structure becomes apparent. Readers must navigate through a plethora of esoteric references. The syntax is dense, requiring careful analysis. Metaphors abound, adding layers of meaning. Ultimately, the text challenges conventional understanding."
        
        ptd_simple = self.metrics.calculate_ptd(simple_text)
        ptd_complex = self.metrics.calculate_ptd(complex_text)
        
        self.assertLess(ptd_simple, ptd_complex)

    def test_calculate_ttr(self):
        diverse_text = "Cat dog mouse bird fish. The cat chased the dog. The mouse ran from the bird. Fish swim in the pond. Birds fly in the sky. Dogs bark at strangers. Cats purr when happy."
        repetitive_text = "Cat cat cat cat cat. Dog dog dog dog dog. Mouse mouse mouse mouse mouse. Bird bird bird bird bird. Fish fish fish fish fish. Sky sky sky sky sky. Happy happy happy happy happy."
        
        ttr_diverse = self.metrics.calculate_ttr(diverse_text)
        ttr_repetitive = self.metrics.calculate_ttr(repetitive_text)
        
        self.assertGreater(ttr_diverse, ttr_repetitive)

    def test_calculate_lexical_density(self):
        high_density_text = "Cat dog mouse bird fish. Tree desk monitor glass. Pencil paper book. Computer internet software. Knowledge learning education. Science technology innovation. Creativity imagination inspiration."
        low_density_text = "The cat and the dog. They are friends. The mouse is small. The bird is flying. Fish are in the water. The sky is blue. Everyone is happy."
        
        ld_high = self.metrics.calculate_lexical_density(high_density_text)
        ld_low = self.metrics.calculate_lexical_density(low_density_text)
        
        self.assertGreater(ld_high, ld_low)

    def test_calculate_sentiment(self):
        positive_text = "I love sunny days. The weather is beautiful. Everyone is smiling. The flowers are blooming. Birds are singing. Life is wonderful. I feel so happy."
        negative_text = "I hate rainy days. The weather is terrible. Everyone is frowning. The flowers are wilting. Birds are silent. Life is miserable. I feel so sad."
        
        sentiment_positive = self.metrics.calculate_sentiment(positive_text)
        sentiment_negative = self.metrics.calculate_sentiment(negative_text)
        
        self.assertGreater(sentiment_positive, sentiment_negative)

if __name__ == '__main__':
    unittest.main()
