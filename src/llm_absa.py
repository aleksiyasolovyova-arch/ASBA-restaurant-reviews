"""
LLM-Based Aspect-Based Sentiment Analysis using Ollama
"""
import json
import re
from typing import List, Dict, Any, Optional
import ollama
from src.base import ABSAAnalyzer, AspectSentiment


class LLMABSA(ABSAAnalyzer):

    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.1, max_retries: int = 3):
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self._verify_model()

    def _verify_model(self):
        try:
            print(f"Using model '{self.model_name}'...")

            try:
                ollama.show(self.model_name)
                print(f"✓ Model '{self.model_name}' is available.")
            except:
                print(f"Model '{self.model_name}' not found. Attempting to pull...")
                ollama.pull(self.model_name)
                print(f"✓ Model '{self.model_name}' pulled successfully.")

        except Exception as e:
            print(f"Warning: Could not verify model: {e}")
            print(f"Will attempt to use model '{self.model_name}' anyway.")
            print("If this fails, run: ollama pull " + self.model_name)

#Used Claude for this function, this is the exact prompt I gave it:
#"Create a Python function that generates a prompt for ASBA. It should take text/string as input and return a formatted prompt string.
# Include clear instructions and examples, specify JSON output with aspect, sentiment and confidence fields.Format the final text input at the end of the prompt"
    def _create_prompt(self, text: str) -> str:
        """
        Create a well-structured prompt with few-shot examples.
        """
        prompt = """You are an expert at aspect-based sentiment analysis. Your task is to identify specific aspects (features, attributes, or entities) mentioned in text and determine the sentiment expressed toward each aspect.

    Instructions:
    1. Identify all aspects mentioned in the text (e.g., food, service, price, quality, etc.)
    2. For each aspect, determine if the sentiment is positive, negative, or neutral
    3. Provide a confidence score between 0.0 and 1.0
    4. Return ONLY valid JSON in the exact format shown below

    Output format (JSON only, no other text):
    {{
      "aspects": [
        {{
          "aspect": "aspect name",
          "sentiment": "positive/negative/neutral",
          "confidence": 0.0-1.0
        }}
      ]
    }}

    Examples:

    Input: "The pizza was delicious but the service was terrible."
    Output:
    {{
      "aspects": [
        {{"aspect": "pizza", "sentiment": "positive", "confidence": 0.95}},
        {{"aspect": "service", "sentiment": "negative", "confidence": 0.95}}
      ]
    }}

    Input: "Great food and amazing atmosphere, though a bit pricey."
    Output:
    {{
      "aspects": [
        {{"aspect": "food", "sentiment": "positive", "confidence": 0.90}},
        {{"aspect": "atmosphere", "sentiment": "positive", "confidence": 0.95}},
        {{"aspect": "price", "sentiment": "negative", "confidence": 0.75}}
      ]
    }}

    Input: "The battery life is excellent and the screen is bright, but it's quite heavy."
    Output:
    {{
      "aspects": [
        {{"aspect": "battery life", "sentiment": "positive", "confidence": 0.95}},
        {{"aspect": "screen", "sentiment": "positive", "confidence": 0.90}},
        {{"aspect": "weight", "sentiment": "negative", "confidence": 0.85}}
      ]
    }}

    Now analyze this text:
    Input: "{text}"
    Output:"""

        return prompt.format(text=text)

    def _call_llama(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': self.temperature,
                        'num_predict': 500,  # Maximum tokens to generate
                    }
                )
                return response['response']
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to generate response after {self.max_retries} attempts: {e}")
                print(f"Attempt {attempt + 1} failed, retrying... Error: {e}")

        return ""

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'\{[^{}]*\{.*?\}[^{}]*\}|\{.*?\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        brace_count = 0
        start_idx = None
        for i, char in enumerate(response):
            if char == '{':
                if start_idx is None:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    try:
                        json_str = response[start_idx:i + 1]
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
                    start_idx = None

        aspects_match = re.search(r'"aspects"\s*:\s*\[(.+?)\]', response, re.DOTALL)
        if aspects_match:
            try:
                return json.loads('{"aspects": [' + aspects_match.group(1) + ']}')
            except json.JSONDecodeError:
                pass

        # This is Claude generated
        cleaned = re.sub(r'^```json\s*', '', response.strip())
        cleaned = re.sub(r'\s*```$', '', cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print("Warning: Failed to parse JSON from response.")
            print(f"Response preview: {response[:500]}...")
            return {"aspects": []}

    def _validate_and_convert(self, parsed_data: Dict[str, Any]) -> List[AspectSentiment]:
        results = []
        aspects = parsed_data.get('aspects', [])
        if not isinstance(aspects, list):
            print("Warning: 'aspects' is not a list")
            return results

        for item in aspects:
            try:
                aspect = item.get('aspect', '').strip()
                sentiment = item.get('sentiment', 'neutral').lower().strip()
                confidence = float(item.get('confidence', 0.5))

                if sentiment not in ['positive', 'negative', 'neutral']:
                    print(f"Warning: Invalid sentiment '{sentiment}', defaulting to 'neutral'")
                    sentiment = 'neutral'

                confidence = max(0.0, min(1.0, confidence))

                if aspect:
                    results.append(AspectSentiment(
                        aspect=aspect,
                        sentiment=sentiment,
                        confidence=confidence,
                        text_span=None
                    ))
            except (ValueError, TypeError) as e:
                print(f"Warning: Skipping invalid aspect entry: {item}. Error: {e}")
                continue

        return results

    def analyze(self, text: str, debug: bool = False) -> List[AspectSentiment]:
        if not text or not text.strip():
            return []

        prompt = self._create_prompt(text)
        response = self._call_llama(prompt)

        if debug:
            print("\n=== DEBUG: LLM Response ===")
            print(response)
            print("=" * 50)

        parsed_data = self._parse_json_response(response)

        if debug:
            print("\n=== DEBUG: Parsed Data ===")
            print(parsed_data)
            print("=" * 50)

        return self._validate_and_convert(parsed_data)

    def analyze_batch(self, texts: List[str]) -> List[List[AspectSentiment]]:
        return [self.analyze(text) for text in texts]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        """
        try:
            model_info = ollama.show(self.model_name)
            return {
                'model_name': self.model_name,
                'temperature': self.temperature,
                'max_retries': self.max_retries,
                'model_details': model_info
            }
        except Exception as e:
            return {
                'model_name': self.model_name,
                'temperature': self.temperature,
                'max_retries': self.max_retries,
                'error': str(e)
            }

