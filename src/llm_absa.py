import json
import re
import time
from typing import List, Dict, Any, Optional
import ollama
from src.base import ABSAAnalyzer, AspectSentiment


class LLMABSA(ABSAAnalyzer):

    def __init__(
            self,
            model_name: str = "orca2",
            temperature: float = 0.1,
            max_retries: int = 1,
            cache_responses: bool = True,
            timeout: int = 30
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_responses = cache_responses
        self.timeout = timeout
        self._response_cache = {}
        self._model_verified = False
        self._verify_model()

    def _verify_model(self):
        if self._model_verified:
            return
        try:
            ollama.show(self.model_name)
            print(f"Model '{self.model_name}' ready")
            self._model_verified = True
        except Exception as e:
            print(f"Model '{self.model_name}' not available: {e}")

    def _build_prompt(self, text: str) -> str:
        return f"""Analyze this review and expertly extract aspects with their sentiments. Return ONLY valid JSON.

REVIEW: "{text}"

Return ONLY JSON with this exact structure:
{{
  "aspects": [
    {{"aspect": "aspect_name", "sentiment": "positive|negative|neutral", "confidence": 0.9}}
  ]
}}

Guidelines:
- Be concise and extract specific terms like "service", "pasta", "price", "staff"
- If sentiment unclear, set it to "neutral"
- Confidence between 0.5 and 1.0
- Return ONLY JSON, nothing else.
"""

    def _call_llm(self, prompt: str) -> str:
        cache_key = hash(prompt)
        if self.cache_responses and cache_key in self._response_cache:
            return self._response_cache[cache_key]

        for attempt in range(self.max_retries):
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": self.temperature,
                        "num_predict": 100,
                        "top_k": 20,
                        "top_p": 0.9,
                    }
                )
                content = response.get("message", {}).get("content", "").strip()
                if content:
                    if self.cache_responses:
                        self._response_cache[cache_key] = content
                    return content
            except Exception as e:
                print(f"LLM call failed (attempt {attempt + 1}): {e}")
                time.sleep(0.3)

        return '{"aspects": []}'

    def _extract_json(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"aspects": []}

        text = text.strip()
        text = re.sub(r'```json\s*|\s*```', '', text)
        start_idx = text.find('{')
        end_idx = text.rfind('}')

        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return {"aspects": []}

        json_str = text[start_idx:end_idx + 1]
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed
            else:
                return {"aspects": []}
        except json.JSONDecodeError as e:
            try:
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

            print(f"JSON parse failed: {e}")
            return {"aspects": []}

    def _quick_sentiment_analysis(self, text: str, aspect: str) -> tuple[str, float]:
        """Local heuristic-based sentiment check."""
        text_lower = text.lower()
        aspect_lower = aspect.lower()

        positive_words = {'amazing', 'delicious', 'yummy', 'great', 'excellent',
                          'perfect', 'love', 'loved', 'favorite', 'best', 'good',
                          'friendly', 'fast', 'quick', 'pleasant'}
        negative_words = {'terrible', 'awful', 'bad', 'slow', 'rude', 'cold',
                          'dirty', 'expensive', 'overpriced', 'worst', 'wait'}

        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count > neg_count:
            return "positive", 0.8
        elif neg_count > pos_count:
            return "negative", 0.8
        return "neutral", 0.6

    def _validate_aspect(self, text: str, aspect_data: Any) -> Optional[AspectSentiment]:
        if not isinstance(aspect_data, dict):
            return None

        aspect = str(aspect_data.get("aspect", "")).strip().lower()
        if not aspect:
            return None

        raw_sentiment = str(aspect_data.get("sentiment", "neutral")).lower()
        sentiment_map = {
            "pos": "positive", "positive": "positive", "good": "positive",
            "neg": "negative", "negative": "negative", "bad": "negative",
            "neu": "neutral", "neutral": "neutral"
        }
        sentiment = sentiment_map.get(raw_sentiment, "neutral")

        try:
            confidence = float(aspect_data.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.7

        if sentiment not in {"positive", "negative", "neutral"} or confidence < 0.6:
            sentiment, confidence = self._quick_sentiment_analysis(text, aspect)

        return AspectSentiment(aspect=aspect, sentiment=sentiment, confidence=confidence)

    def _rule_based_fallback(self, text: str) -> List[AspectSentiment]:
        aspects = []
        text_lower = text.lower()

        common_aspects = {
            'food', 'service', 'price', 'atmosphere', 'staff',
            'location', 'menu', 'quality', 'portion', 'wait', 'cleanliness'
        }

        for aspect in common_aspects:
            if aspect in text_lower:
                sentiment, confidence = self._quick_sentiment_analysis(text, aspect)
                aspects.append(AspectSentiment(aspect, sentiment, confidence))
        return aspects[:5]

    def analyze(self, text: str) -> List[AspectSentiment]:
        if not text.strip():
            return []

        try:
            prompt = self._build_prompt(text)
            response = self._call_llm(prompt)
            parsed = self._extract_json(response)
            aspects = []

            seen = set()
            for aspect_data in parsed.get("aspects", []):
                aspect_obj = self._validate_aspect(text, aspect_data)
                if aspect_obj and aspect_obj.aspect not in seen:
                    aspects.append(aspect_obj)
                    seen.add(aspect_obj.aspect)

            if not aspects:
                aspects = self._rule_based_fallback(text)

            return aspects

        except Exception as e:
            print(f"Analysis failed for text '{text[:40]}...': {e}")
            return self._rule_based_fallback(text)

    def analyze_batch(self, texts: List[str]) -> List[List[AspectSentiment]]:
        if not texts:
            return []

        print(f"Processing {len(texts)} reviews with optimized LLM...")
        start_time = time.time()
        results = []

        for i, text in enumerate(texts):
            results.append(self.analyze(text))
            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                avg = elapsed / (i + 1)
                remaining = avg * (len(texts) - (i + 1))
                print(f"📊 {i + 1}/{len(texts)} processed | avg={avg:.2f}s | ETA={remaining:.1f}s")

        total_time = time.time() - start_time
        print(f"Completed {len(texts)} reviews in {total_time:.1f}s (avg={total_time / len(texts):.2f}s each)")
        return results