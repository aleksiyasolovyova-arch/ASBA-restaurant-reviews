import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
import re
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
        #Extract nouns as aspect candidates.
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stop_words = {'the', 'was', 'were', 'and', 'but', 'very', 'really', 'quite'}
        aspects = [w for w in words if w not in stop_words]
        return list(dict.fromkeys(aspects))[:5] or ['item']

    def _predict_sentiment(self, text: str, aspect: str) -> Tuple[str, float]:
        #Predict sentiment for text-aspect pair.
        input_text = f"{text} [SEP] {aspect}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()} #This was recommended by Claude to have GPU support

        with torch.no_grad():  #Recommended by Claude to save memory and speed up computation
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            predicted_class = probs.argmax().item()
            confidence = probs[predicted_class].item()

        return self.label_map[predicted_class], confidence