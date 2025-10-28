import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
import spacy
from .base import ABSAAnalyzer, AspectSentiment


class TransformerABSA(ABSAAnalyzer):
    def __init__(self, model_name: str = "yangheng/deberta-v3-base-absa-v1.1"):
        try:
            print(f"Loading transformer model: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            print(f"Model '{model_name}' loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # spaCy for aspect extraction
        self.nlp = spacy.load("en_core_web_sm")
        self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def analyze(self, text: str) -> List[AspectSentiment]:
        #Analyze text and extract aspect-sentiment pairs.
        aspects = self._extract_aspects(text)
        results = []

        for aspect in aspects:
            sentiment, confidence = self._predict_sentiment(text, aspect)
            results.append(AspectSentiment(
                aspect=aspect,
                sentiment=sentiment,
                confidence=confidence,
                text_span=None
            ))

        return results

    def _extract_aspects(self, text: str) -> List[str]:
        #Extract nouns as aspect candidates using spaCy.
        doc = self.nlp(text)
        aspects = []

        # Get all nouns from the text
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                noun = token.text.lower()
                if len(noun) > 2 and noun not in aspects:
                    if noun not in ['thing', 'place', 'way', 'time']:
                        aspects.append(noun)

        # If no aspects found, use 'overall' as fallback   #Recommended by Claude
        return aspects[:5] if aspects else ['overall']

    def _predict_sentiment(self, text: str, aspect: str) -> Tuple[str, float]:
        #Predict sentiment for text-aspect pair.
        input_text = f"{text} [SEP] {aspect}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}   #This was recommended by Claude to have GPU support

        with torch.no_grad():    #Recommended by Claude to save memory and speed up computation
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            predicted_class = probs.argmax().item()
            confidence = probs[predicted_class].item()

        return self.label_map[predicted_class], confidence