import spacy
import statistics
import nltk
from typing import List
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from src.base import ABSAAnalyzer, AspectSentiment

nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)


class LexiconABSAAnalyzer(ABSAAnalyzer):

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load('en_core_web_sm')

    def _return_polarity_scores(self, word: str, negated: bool = False) -> float:
        score = self.sia.polarity_scores(word)['compound']
        if abs(score) < 0.2:
            score *= 2.5
        if negated:
            score = -score * 1.2
        return max(-1.0, min(1.0, score))

    def _check_negation(self, token):
        NEGATION_WORDS = {'not', 'no', 'never', "n't", 'none', 'neither'}
        return any(child.dep_ == 'neg' or child.text.lower() in NEGATION_WORDS for child in token.children)

    def _linear_norm(self, x: float) -> float:
        return (x + 1) / 2

    def _polarity_to_sentiment(self, score: float) -> str:
        if score >= 0.6:
            return 'positive'
        elif score <= 0.4:
            return 'negative'
        else:
            return 'neutral'

    def _extract_aspect_opinion_pairs(self, doc):
        aspects = {}
        for chunk in doc.noun_chunks:
            aspect = None
            for token in chunk:
                if token.pos_ == "NOUN" and token.lemma_.lower() not in self.stop_words:
                    aspect = token.lemma_.lower()
                    break
            if not aspect:
                continue

            modifiers = []
            for token in chunk:
                if token.pos_ in ["ADJ", "ADV"]:
                    neg = self._check_negation(token)
                    pol = self._return_polarity_scores(token.lemma_.lower(), neg)
                    if abs(pol) >= 0.15:
                        modifiers.append(pol)

            if modifiers:
                aspects[aspect] = statistics.mean(modifiers)

        return aspects

    def analyze(self, text: str) -> List[AspectSentiment]:
        doc = self.nlp(text)
        aspect_opinions = self._extract_aspect_opinion_pairs(doc)
        results = []

        for aspect, mean_polarity in aspect_opinions.items():
            normalized = self._linear_norm(mean_polarity)
            sentiment = self._polarity_to_sentiment(normalized)
            confidence = abs(mean_polarity)

            results.append(
                AspectSentiment(
                    aspect=aspect,
                    sentiment=sentiment,
                    confidence=confidence
                )
            )

        return results