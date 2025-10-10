
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class AspectSentiment:

    aspect: str
    sentiment: str
    confidence: float
    text_span: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        #Validate sentiment and confidence values.
        valid_sentiments = {'positive', 'negative', 'neutral'}
        if self.sentiment.lower() not in valid_sentiments:
            raise ValueError(f"Sentiment must be one of {valid_sentiments}, got '{self.sentiment}'")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        self.sentiment = self.sentiment.lower()

    def to_dict(self) -> dict:
        #Convert to dictionary format.
        return {
            'aspect': self.aspect,
            'sentiment': self.sentiment,
            'confidence': round(self.confidence, 3),
            'text_span': self.text_span
        }

    def __str__(self) -> str:
        return f"Aspect('{self.aspect}') → Sentiment: {self.sentiment.upper()} (conf: {self.confidence:.2f})"

#Abstract base class for all ABSA implementations.
class ABSAAnalyzer(ABC):

    @abstractmethod
    def analyze(self, text: str) -> List[AspectSentiment]:

        raise NotImplementedError("Subclasses must implement analyze() method")

    # Analyze multiple texts in batch.
    def batch_analyze(self, texts: List[str]) -> List[List[AspectSentiment]]:

        results = []
        for text in texts:
            try:
                results.append(self.analyze(text))
            except Exception as e:
                print(f"Error analyzing text: {e}")
                results.append([])
        return results

    def get_name(self) -> str:
        return self.__class__.__name__