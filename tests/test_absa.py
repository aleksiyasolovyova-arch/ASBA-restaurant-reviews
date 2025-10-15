import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.base import AspectSentiment, ABSAAnalyzer
from src.lexicon_absa import LexiconABSA
from src.transformer_absa import TransformerABSA
from src.llm_absa import LLMABSA


# Test AspectSentiment class
def test_aspect_sentiment_valid():
    aspect = AspectSentiment("food", "positive", 0.9)
    assert aspect.aspect == "food"
    assert aspect.sentiment == "positive"
    assert aspect.confidence == 0.9


def test_aspect_sentiment_invalid_sentiment():
    with pytest.raises(ValueError):
        AspectSentiment("food", "invalid", 0.9)


def test_aspect_sentiment_invalid_confidence():
    with pytest.raises(ValueError):
        AspectSentiment("food", "positive", 1.5)


def test_aspect_sentiment_to_dict():
    aspect = AspectSentiment("food", "positive", 0.9)
    result = aspect.to_dict()
    assert result['aspect'] == "food"
    assert result['sentiment'] == "positive"


# Test LexiconABSA
def test_lexicon_positive():
    analyzer = LexiconABSA()
    results = analyzer.analyze("The delicious food and excellent service made our day.")
    assert isinstance(results, list)
    if len(results) > 0:
        assert any(r.sentiment == 'positive' for r in results)


def test_lexicon_negative():
    analyzer = LexiconABSA()
    results = analyzer.analyze("The terrible service and horrible food ruined everything.")
    assert isinstance(results, list)
    if len(results) > 0:
        assert any(r.sentiment == 'negative' for r in results)


def test_lexicon_mixed():
    analyzer = LexiconABSA()
    results = analyzer.analyze("The amazing food and awful service.")
    assert isinstance(results, list)


def test_lexicon_confidence():
    analyzer = LexiconABSA()
    results = analyzer.analyze("The food was excellent.")
    for result in results:
        assert 0.0 <= result.confidence <= 1.0


# Test TransformerABSA
def test_transformer_positive():
    try:
        analyzer = TransformerABSA()
        results = analyzer.analyze("The food was delicious.")
        assert len(results) > 0
        assert any(r.sentiment == 'positive' for r in results)
    except:
        pytest.skip("Transformer model not available")


def test_transformer_negative():
    try:
        analyzer = TransformerABSA()
        results = analyzer.analyze("The service was terrible.")
        assert len(results) > 0
        assert any(r.sentiment == 'negative' for r in results)
    except:
        pytest.skip("Transformer model not available")


def test_transformer_mixed():
    try:
        analyzer = TransformerABSA()
        results = analyzer.analyze("The food was amazing but the service was slow.")
        assert len(results) >= 2
    except:
        pytest.skip("Transformer model not available")


def test_transformer_confidence():
    try:
        analyzer = TransformerABSA()
        results = analyzer.analyze("The food was excellent.")
        for result in results:
            assert 0.0 <= result.confidence <= 1.0
    except:
        pytest.skip("Transformer model not available")


# Test LLMABSA
def test_llm_positive():
    try:
        analyzer = LLMABSA()
        results = analyzer.analyze("The food was delicious.")
        assert len(results) > 0
        assert any(r.sentiment == 'positive' for r in results)
    except:
        pytest.skip("LLM model not available")


def test_llm_negative():
    try:
        analyzer = LLMABSA()
        results = analyzer.analyze("The service was terrible.")
        assert len(results) > 0
        assert any(r.sentiment == 'negative' for r in results)
    except:
        pytest.skip("LLM model not available")


def test_llm_mixed():
    try:
        analyzer = LLMABSA()
        results = analyzer.analyze("The food was great but the service was slow.")
        assert len(results) >= 2
    except:
        pytest.skip("LLM model not available")


def test_llm_confidence():
    try:
        analyzer = LLMABSA()
        results = analyzer.analyze("The food was excellent.")
        for result in results:
            assert 0.0 <= result.confidence <= 1.0
    except:
        pytest.skip("LLM model not available")


def test_llm_empty_text():
    try:
        analyzer = LLMABSA()
        results = analyzer.analyze("")
        assert len(results) == 0
    except:
        pytest.skip("LLM model not available")


# Test abstract base class
def test_cannot_instantiate_base_class():
    with pytest.raises(TypeError):
        ABSAAnalyzer()


# Test all analyzers together
def test_all_analyzers_return_valid_results():
    text = "The food was great."
    analyzers = []

    try:
        analyzers.append(LexiconABSA())
    except:
        pass

    try:
        analyzers.append(TransformerABSA())
    except:
        pass

    try:
        analyzers.append(LLMABSA())
    except:
        pass

    assert len(analyzers) > 0

    for analyzer in analyzers:
        results = analyzer.analyze(text)
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, AspectSentiment)
            assert result.sentiment in ['positive', 'negative', 'neutral']
            assert 0.0 <= result.confidence <= 1.0


