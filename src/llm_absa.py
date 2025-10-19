import json
import re
import time
from typing import List, Dict, Any, Optional
import ollama
from src.base import ABSAAnalyzer, AspectSentiment


class LLMABSA(ABSAAnalyzer):

    def __init__(
            self,
            model_name: str = "llama3.2:1b",
            temperature: float = 0.0,
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

# Used Claude for this function, this is the exact prompt I gave it:
# "Create a Python function that generates a prompt for ASBA. It should take text/string as input and return a formatted prompt string.
# Include clear instructions and examples, specify JSON output with aspect, sentiment and confidence fields.Format the final text input at the end of the prompt"
    def _build_prompt(self, text: str) -> str:
        return f"""Analyze this restaurant review and extract aspects with their sentiments.

    Review: "{text}"

    Instructions:
    - Extract ONLY aspects explicitly mentioned in the review
    - For each aspect, determine sentiment: "positive", "negative", or "neutral"
    - Assign confidence 0.6-1.0 based on how clear the sentiment is
    - Return ONLY valid JSON, no other text

    Output format:
    {{
      "aspects": [
        {{"aspect": "aspect_name", "sentiment": "sentiment", "confidence": 0.0}}
      ]
    }}

    Examples of different scenarios:
    - "The food was amazing but service was slow" → food:positive, service:negative
    - "Everything was perfect" → food:positive, service:positive, ambiance:positive
    - "It was okay, nothing special" → food:neutral, service:neutral

    Now analyze this review:"""

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
                        "num_predict": 150,
                        "top_k": 10,
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

        # Fix common JSON errors in order

        # 1. Fix missing commas between objects: }{  ->  },{
        json_str = re.sub(r'\}\s*\{', '},{', json_str)

        # 2. Fix missing commas between array items: ][  ->  ],[
        json_str = re.sub(r'\]\s*\[', '],[', json_str)

        # 3. Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # 4. Fix common quote issues
        json_str = re.sub(r':\s*"([^"]*)"([^,}\]]*)', r': "\1\2"', json_str)

        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed
            return {"aspects": []}
        except json.JSONDecodeError as e:
            try:
                json_str = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)', r'\1"\2"\3', json_str)
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

            try:
                aspects = []
                pattern = r'"aspect"\s*:\s*"([^"]+)".*?"sentiment"\s*:\s*"([^"]+)".*?"confidence"\s*:\s*([0-9.]+)'
                matches = re.finditer(pattern, json_str, re.DOTALL)

                for match in matches:
                    aspects.append({
                        "aspect": match.group(1),
                        "sentiment": match.group(2),
                        "confidence": float(match.group(3))
                    })

                if aspects:
                    return {"aspects": aspects}
            except Exception:
                pass

            return {"aspects": []}

    def _quick_sentiment_analysis(self, text: str, aspect: str) -> tuple[str, float]:
        text_lower = text.lower()
        aspect_lower = aspect.lower()

        strong_positive = {'amazing', 'excellent', 'outstanding', 'perfect', 'love', 'loved', 'best', 'fantastic',
                           'wonderful', 'delicious', 'awesome'}
        positive = {'good', 'great', 'nice', 'pleasant', 'enjoyable', 'decent', 'fine', 'tasty'}
        strong_negative = {'terrible', 'awful', 'horrible', 'disgusting', 'worst', 'hate', 'hated'}
        negative = {'bad', 'poor', 'slow', 'rude', 'cold', 'overpriced', 'expensive', 'disappointing', 'mediocre'}
        neutral = {'okay', 'decent', 'fine', 'average', 'ordinary', 'nothing special', 'standard'}

        if aspect_lower not in text_lower:
            return "neutral", 0.6

        aspect_pos = text_lower.find(aspect_lower)
        context_start = max(0, aspect_pos - 100)
        context_end = min(len(text_lower), aspect_pos + len(aspect) + 100)
        context = text_lower[context_start:context_end]

        strong_pos_count = sum(1 for w in strong_positive if w in context)
        pos_count = sum(1 for w in positive if w in context)
        strong_neg_count = sum(1 for w in strong_negative if w in context)
        neg_count = sum(1 for w in negative if w in context)
        neutral_count = sum(1 for w in neutral if w in context)

        total_score = (strong_pos_count * 2 + pos_count * 1 +
                       neutral_count * 0 +
                       neg_count * -1 + strong_neg_count * -2)

        if total_score > 1:
            confidence = min(0.95, 0.7 + (total_score - 1) * 0.1)
            return "positive", confidence
        elif total_score < -1:
            confidence = min(0.95, 0.7 + (abs(total_score) - 1) * 0.1)
            return "negative", confidence
        else:
            if neutral_count > 0 or ("okay" in context and "but" not in context):
                return "neutral", 0.7
            return "neutral", 0.6

    def _validate_aspect(self, text: str, aspect_data: Any) -> Optional[AspectSentiment]:
        if not isinstance(aspect_data, dict):
            return None

        aspect = str(aspect_data.get("aspect", "")).strip().lower()
        if not aspect or len(aspect) < 2:
            return None

        raw_sentiment = str(aspect_data.get("sentiment", "neutral")).lower().strip()
        sentiment_map = {
            "pos": "positive", "positive": "positive", "good": "positive",
            "neg": "negative", "negative": "negative", "bad": "negative",
            "neu": "neutral", "neutral": "neutral", "mixed": "neutral"
        }
        sentiment = sentiment_map.get(raw_sentiment, "neutral")

        try:
            confidence = float(aspect_data.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.7

        if confidence < 0.5:
            sentiment, confidence = self._quick_sentiment_analysis(text, aspect)

        return AspectSentiment(aspect=aspect, sentiment=sentiment, confidence=confidence)

    def _rule_based_fallback(self, text: str) -> List[AspectSentiment]:
        aspects = []
        text_lower = text.lower()

        aspect_keywords = {
            'food': ['food', 'dish', 'meal', 'plate', 'menu'],
            'service': ['service', 'server', 'waiter', 'waitress', 'staff'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money'],
            'ambiance': ['atmosphere', 'ambiance', 'decor', 'music', 'noise'],
            'quality': ['quality', 'fresh', 'stale', 'hot', 'cold']
        }

        for aspect, keywords in aspect_keywords.items():
            if any(kw in text_lower for kw in keywords):
                sentiment, confidence = self._quick_sentiment_analysis(text, aspect)
                aspects.append(AspectSentiment(aspect, sentiment, confidence))

        return aspects[:4]

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
            print(f" Analysis error: {e}")
            return self._rule_based_fallback(text)

    def analyze_batch(self, texts: List[str]) -> List[List[AspectSentiment]]:
        if not texts:
            return []

        print(f"Processing {len(texts)} reviews with {self.model_name}...")
        start_time = time.time()
        results = []

        for i, text in enumerate(texts):
            results.append(self.analyze(text))

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                avg = elapsed / (i + 1)
                remaining = avg * (len(texts) - (i + 1))
                print(f"   {i + 1}/{len(texts)} | avg={avg:.2f}s | ETA={remaining:.0f}s")

        total_time = time.time() - start_time
        print(f"  ✓ Completed {len(texts)} reviews in {total_time:.1f}s "
              f"(avg={total_time / len(texts):.2f}s each)")
        return results
