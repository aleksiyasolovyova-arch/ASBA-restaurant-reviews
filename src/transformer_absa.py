
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
import re

try:
    from .base import ABSAAnalyzer, AspectSentiment
except ImportError:
    from base import ABSAAnalyzer, AspectSentiment


class TransformerABSA(ABSAAnalyzer):

    # Initialize the transformer-based analyzer.
    def __init__(self, model_name: str = "yangheng/deberta-v3-base-absa-v1.1"):

        print(f"Loading transformer model: {model_name}")
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Label mapping for the model
        self.label_map = {
            0: 'negative',
            1: 'neutral',
            2: 'positive'
        }

        # Common aspects to extract from restaurant reviews
        self.common_aspects = [
            'food', 'service', 'ambiance', 'price', 'location',
            'quality', 'staff', 'atmosphere', 'value', 'menu',
            'decor', 'drinks', 'wait time', 'experience'
        ]

    # Analyze text and extract aspect-sentiment pairs.
    def analyze(self, text: str) -> List[AspectSentiment]:

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        # First, extract candidate aspects from the text
        candidate_aspects = self._extract_aspects(text)

        # For each aspect, predict sentiment
        results = []
        for aspect in candidate_aspects:
            sentiment, confidence = self._predict_sentiment(text, aspect)

            # Find aspect position in text
            text_span = self._find_aspect_span(text, aspect)

            results.append(AspectSentiment(
                aspect=aspect,
                sentiment=sentiment,
                confidence=confidence,
                text_span=text_span
            ))

        return results

    # Extract aspect terms from the text.
    def _extract_aspects(self, text: str) -> List[str]:

        text_lower = text.lower()
        detected_aspects = []

        for aspect in self.common_aspects:
            if aspect in text_lower:
                detected_aspects.append(aspect)

        # If no common aspects found, extract nouns
        if not detected_aspects:
            words = re.findall(r'\b[a-z]{3,}\b', text_lower)
            detected_aspects = list(set(words[:3]))

        return detected_aspects if detected_aspects else ['restaurant']

    # Predict sentiment for a given text-aspect pair.
    def _predict_sentiment(self, text: str, aspect: str) -> Tuple[str, float]:

        # Format input as expected by the model: "[CLS] text [SEP] aspect [SEP]"
        input_text = f"{text} [SEP] {aspect}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()

        sentiment = self.label_map.get(predicted_class, 'neutral')

        return sentiment, confidence

    # Find the position of aspect in the original text.
    def _find_aspect_span(self, text: str, aspect: str) -> Tuple[int, int]:

        text_lower = text.lower()
        aspect_lower = aspect.lower()

        start = text_lower.find(aspect_lower)
        if start != -1:
            return (start, start + len(aspect))

        return None

    # Analyze sentiment for specific aspects provided by the user.
    def analyze_with_given_aspects(self, text: str, aspects: List[str]) -> List[AspectSentiment]:

        results = []

        for aspect in aspects:
            sentiment, confidence = self._predict_sentiment(text, aspect)
            text_span = self._find_aspect_span(text, aspect)

            results.append(AspectSentiment(
                aspect=aspect,
                sentiment=sentiment,
                confidence=confidence,
                text_span=text_span
            ))

        return results


def main():
    #Test TransformerABSA analyzer

    print("=" * 60)
    print("TransformerABSA Test")
    print("=" * 60)

    # Initialize analyzer
    analyzer = TransformerABSA()

    # Test cases
    test_reviews = [
        "The food was absolutely delicious but the service was terrible and slow.",
        "Great ambiance and friendly staff, but the prices are too high.",
        "Amazing experience! The menu has great variety and the quality is outstanding."
    ]

    print("\n--- Test 1: Automatic Aspect Extraction ---")
    for i, review in enumerate(test_reviews, 1):
        print(f"\nReview {i}: {review}")
        results = analyzer.analyze(review)

        for result in results:
            print(
                f"  Aspect: {result.aspect:15} | Sentiment: {result.sentiment:8} | Confidence: {result.confidence:.3f}")

    print("\n\n--- Test 2: Custom Aspects ---")
    custom_review = "The pizza was excellent, wine selection is impressive, but waiting time was awful."
    custom_aspects = ['pizza', 'wine', 'wait time']

    print(f"\nReview: {custom_review}")
    print(f"Custom aspects: {custom_aspects}")

    results = analyzer.analyze_with_given_aspects(custom_review, custom_aspects)

    for result in results:
        print(f"  Aspect: {result.aspect:15} | Sentiment: {result.sentiment:8} | Confidence: {result.confidence:.3f}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
