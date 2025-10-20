import spacy
import statistics
import nltk
from typing import List, Dict, Set
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from src.base import ABSAAnalyzer, AspectSentiment

nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)


class LexiconABSA(ABSAAnalyzer):

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load("en_core_web_sm")
        self.aspect_categories = {
            'food': {'food', 'dish', 'meal', 'pizza', 'burger', 'chicken', 'beef',
                     'salad', 'soup', 'appetizer', 'entree', 'dessert', 'curry',
                     'naan', 'pasta', 'gnocchi', 'eggplant', 'cheese', 'waffle'},
            'service': {'service', 'server', 'waiter', 'waitress', 'waitstaff',
                        'staff', 'bartender', 'host', 'hostess'},
            'ambiance': {'atmosphere', 'ambiance', 'decor', 'interior', 'patio',
                         'seating', 'music', 'noise', 'outside'},
            'price': {'price', 'cost', 'expensive', 'cheap', 'value', 'money'},
            'time': {'wait', 'waiting', 'speed', 'slow', 'fast', 'quick'}
        }

    def _return_polarity(self, word: str, negated: bool = False) -> float:
        score = self.sia.polarity_scores(word)["compound"]
        if abs(score) < 0.2:
            score *= 2.5
        if negated:
            score = -score * 1.2
        return max(-1.0, min(1.0, score))

    def _check_negation(self, token) -> bool:
        NEGATION_WORDS = {'not', 'no', 'never', "n't", 'none', 'neither',
                          'nobody', 'nothing', 'nowhere', 'hardly', 'scarcely',
                          'barely', "won't", "wouldn't", "didn't"}

        if any(child.dep_ == "neg" or child.text.lower() in NEGATION_WORDS
               for child in token.children):
            return True

        head = token.head
        if head and head.text.lower() in NEGATION_WORDS:
            return True

        prev_tokens = list(token.doc[max(0, token.i - 3):token.i])
        for t in reversed(prev_tokens):
            if t.text.lower() in NEGATION_WORDS:
                return True
            if t.pos_ in ["PUNCT", "CCONJ"]:
                break
        return False

    def _extract_compound_aspects(self, doc) -> Set[str]:
        compounds = set()

        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                parts = []

                for child in token.children:
                    if child.dep_ == "compound":
                        parts.append(child.lemma_.lower())

                parts.append(token.lemma_.lower())

                if len(parts) > 1:
                    compounds.add(" ".join(parts))

        return compounds

    def _extract_aspects_from_chunks(self, doc) -> Set[str]:
        aspects = set()
        GENERIC_WORDS = {'time', 'thing', 'way', 'year', 'day', 'lot', 'bit',
                         'kind', 'sort', 'type', 'minute', 'hour', 'week',
                         'month', 'person', 'people', 'someone', 'anyone',
                         'case', 'review', 'star', 'place'}

        for chunk in doc.noun_chunks:
            for token in chunk:
                if token.pos_ in ["NOUN", "PROPN"]:
                    lemma = token.lemma_.lower()
                    if (lemma not in self.stop_words and
                            len(lemma) > 2 and
                            lemma not in GENERIC_WORDS and
                            token.ent_type_ not in ["PERSON", "DATE", "TIME",
                                                    "CARDINAL", "ORDINAL"]):
                        aspects.add(lemma)

        return aspects

    def _get_opinion_words(self, doc) -> List[tuple]:
        opinions = []

        for token in doc:
            if token.pos_ in ["ADJ", "ADV", "VERB"]:
                score = self.sia.polarity_scores(token.lemma_)["compound"]

                if abs(score) >= 0.15:
                    negated = self._check_negation(token)
                    final_score = self._return_polarity(token.lemma_, negated)
                    opinions.append((token, final_score))

        return opinions

    def _find_aspect_opinion_pairs(self, doc, aspects: Set[str],
                                   opinions: List[tuple]) -> Dict[str, List[float]]:
        aspect_scores = {asp: [] for asp in aspects}

        aspect_tokens = {}
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in aspects:
                aspect_tokens[token.i] = lemma

            for compound in aspects:
                if " " in compound:
                    words = compound.split()
                    if lemma == words[-1]:
                        aspect_tokens[token.i] = compound

        for opinion_token, score in opinions:
            opinion_idx = opinion_token.i

            # Strategy 1: Dependency-based connection
            head = opinion_token.head
            if head.i in aspect_tokens:
                aspect_scores[aspect_tokens[head.i]].append(score)
                continue

            # Check if aspect is a child of the opinion
            for child in opinion_token.children:
                if child.i in aspect_tokens:
                    aspect_scores[aspect_tokens[child.i]].append(score)
                    break
            else:
                # Strategy 2: Window-based approach
                min_dist = float('inf')
                nearest_aspect = None

                for aspect_idx, aspect in aspect_tokens.items():
                    dist = abs(opinion_idx - aspect_idx)
                    if dist < min_dist and dist <= 5:
                        min_dist = dist
                        nearest_aspect = aspect

                if nearest_aspect:
                    aspect_scores[nearest_aspect].append(score)

        for aspect in aspects:
            if not aspect_scores[aspect]:
                # Find sentence containing the aspect
                for sent in doc.sents:
                    if any(token.lemma_.lower() == aspect for token in sent):
                        sent_score = self.sia.polarity_scores(sent.text)["compound"]
                        if abs(sent_score) >= 0.1:
                            aspect_scores[aspect].append(sent_score)
                        break

        return aspect_scores

    def analyze(self, text: str) -> List[AspectSentiment]:
        doc = self.nlp(text)

        compound_aspects = self._extract_compound_aspects(doc)
        single_aspects = self._extract_aspects_from_chunks(doc)
        all_aspects = compound_aspects | single_aspects

        opinions = self._get_opinion_words(doc)

        aspect_scores = self._find_aspect_opinion_pairs(doc, all_aspects, opinions)

        results = []
        for aspect, scores in aspect_scores.items():
            if scores:
                avg_score = statistics.mean(scores)

                sentiment = (
                    "positive" if avg_score > 0.15 else
                    "negative" if avg_score < -0.15 else
                    "neutral"
                )

                confidence = min(1.0, abs(avg_score))
                results.append(AspectSentiment(aspect, sentiment, confidence))

        return results
