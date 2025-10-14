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
        self.nlp = spacy.load("en_core_web_sm")

    def _get_sentiment(self, word: str, negated: bool = False) -> float:
        score = self.sia.polarity_scores(word)["compound"]
        if negated:
            score *= -1
        return score

    def _is_negated(self, token):
        NEG_WORDS = {"not", "no", "never", "n't", "none", "neither"}
        if any(c.dep_ == "neg" or c.text.lower() in NEG_WORDS for c in token.children):
            return True
        if token.head.text.lower() in NEG_WORDS or token.head.dep_ == "neg":
            return True
        return False

    def _find_opinion_modifiers(self, aspect_token):
        modifiers = []

        for child in aspect_token.children:
            if child.pos_ == "ADJ":
                neg = self._is_negated(child)
                modifiers.append(self._get_sentiment(child.text, neg))

        for ancestor in aspect_token.ancestors:
            if ancestor.pos_ == "VERB":
                for child in ancestor.children:
                    if child.pos_ == "ADJ":
                        neg = self._is_negated(child)
                        modifiers.append(self._get_sentiment(child.text, neg))
        return modifiers

    def analyze(self, text: str) -> List[AspectSentiment]:
        doc = self.nlp(text)
        aspects = {}

        for token in doc:
            if token.pos_ == "NOUN" and token.lemma_.lower() not in self.stop_words:
                mods = self._find_opinion_modifiers(token)
                if mods:
                    aspects[token.lemma_.lower()] = statistics.mean(mods)

        results = []
        for aspect, score in aspects.items():
            sentiment = (
                "positive" if score > 0.2 else
                "negative" if score < -0.2 else
                "neutral"
            )
            confidence = min(1.0, abs(score))
            results.append(AspectSentiment(aspect, sentiment, confidence))

        return results
