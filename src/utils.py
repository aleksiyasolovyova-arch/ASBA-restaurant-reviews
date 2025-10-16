import json
import pandas as pd
from typing import List, Dict, Any

def load_test_samples(filepath: str = "data/test_samples.json") -> List[Dict[str, Any]]:
    #Load test samples from JSON file.
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get('reviews', [])
    return data


def load_evaluation_data(filepath: str = "data/evaluation_data.json") -> List[Dict[str, Any]]:
    #Load evaluation data from JSON file.
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get('labeled_reviews', [])
    return data


def compare_analyzers(text: str, analyzers: Dict[str, Any]) -> pd.DataFrame:
    #Compare multiple analyzers on the same text.
    comparison_data = []

    for name, analyzer in analyzers.items():
        try:
            results = analyzer.analyze(text)

            for result in results:
                comparison_data.append({
                    'analyzer': name,
                    'aspect': result.aspect,
                    'sentiment': result.sentiment,
                    'confidence': result.confidence
                })
        except Exception as e:
            print(f"Error with {name}: {e}")

    return pd.DataFrame(comparison_data)
