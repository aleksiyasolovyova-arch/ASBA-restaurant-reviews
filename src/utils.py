
import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import time

from .base import AspectSentiment

#Load Yelp reviews from JSON file.
def load_yelp_reviews(filepath: str, sample_size: int = None) -> List[Dict[str, Any]]:

    reviews = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
                break
            try:
                review = json.loads(line)
                reviews.append(review)
            except json.JSONDecodeError:
                continue

    return reviews

#Save analysis results to JSON file.
def save_results(results: List[Dict[str, Any]], filepath: str):

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

#Convert AspectSentiment results to pandas DataFrame.
def results_to_dataframe(results: List[List[AspectSentiment]]) -> pd.DataFrame:

    data = []
    for result_list in results:
        for aspect_sent in result_list:
            data.append(aspect_sent.to_dict())

    return pd.DataFrame(data)

#Compare multiple analyzers on the same text.
def compare_analyzers(text: str, analyzers: Dict[str, Any]) -> pd.DataFrame:

    comparison_data = []

    for name, analyzer in analyzers.items():
        start_time = time.time()
        try:
            results = analyzer.analyze(text)
            elapsed = time.time() - start_time

            for result in results:
                comparison_data.append({
                    'analyzer': name,
                    'aspect': result.aspect,
                    'sentiment': result.sentiment,
                    'confidence': result.confidence,
                    'time_ms': elapsed * 1000
                })
        except Exception as e:
            print(f"Error with {name}: {e}")

    return pd.DataFrame(comparison_data)
