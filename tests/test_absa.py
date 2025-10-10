
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.base import AspectSentiment, ABSAAnalyzer
from src.transformer_absa import TransformerABSA

class TestAspectSentiment:
    #Test cases for AspectSentiment data class.

    def test_valid_creation(self):
        #Test creating a valid AspectSentiment object.
        aspect_sent = AspectSentiment(
            aspect="food",
            sentiment="positive",
            confidence=0.9,
            text_span=(0, 4)
        )

        assert aspect_sent.aspect == "food"
        assert aspect_sent.sentiment == "positive"
        assert aspect_sent.confidence == 0.9
        assert aspect_sent.text_span == (0, 4)

    def test_invalid_sentiment(self):
        #Test that invalid sentiment raises ValueError.
        with pytest.raises(ValueError):
            AspectSentiment(
                aspect="food",
                sentiment="invalid",
                confidence=0.9
            )

    def test_invalid_confidence(self):
        #Test that confidence outside [0,1] raises ValueError.
        with pytest.raises(ValueError):
            AspectSentiment(
                aspect="food",
                sentiment="positive",
                confidence=1.5
            )

        with pytest.raises(ValueError):
            AspectSentiment(
                aspect="food",
                sentiment="positive",
                confidence=-0.1
            )

    def test_sentiment_normalization(self):
        #Test that sentiment is normalized to lowercase.
        aspect_sent = AspectSentiment(
            aspect="food",
            sentiment="POSITIVE",
            confidence=0.9
        )

        assert aspect_sent.sentiment == "positive"

    def test_to_dict(self):
        #Test conversion to dictionary.
        aspect_sent = AspectSentiment(
            aspect="food",
            sentiment="positive",
            confidence=0.9,
            text_span=(0, 4)
        )

        result = aspect_sent.to_dict()

        assert result['aspect'] == "food"
        assert result['sentiment'] == "positive"
        assert result['confidence'] == 0.9
        assert result['text_span'] == (0, 4)

    def test_str_representation(self):
        #Test string representation.
        aspect_sent = AspectSentiment(
            aspect="food",
            sentiment="positive",
            confidence=0.9
        )

        result = str(aspect_sent)
        assert "food" in result
        assert "POSITIVE" in result
        assert "0.90" in result

class TestTransformerABSA:
    #Test cases for TransformerABSA implementation.

    @pytest.fixture
    def analyzer(self):
        #Create a TransformerABSA instance for testing.
        try:
            return TransformerABSA()
        except Exception as e:
            pytest.skip(f"Transformer model not available: {e}")

    def test_simple_positive(self, analyzer):
        #Test simple positive sentiment detection.
        text = "The food was delicious."
        results = analyzer.analyze(text)

        assert len(results) > 0

        # Check for positive sentiment
        positive_results = [r for r in results if r.sentiment == 'positive']
        assert len(positive_results) > 0

    def test_simple_negative(self, analyzer):
        #Test simple negative sentiment detection.
        text = "The service was terrible."
        results = analyzer.analyze(text)

        assert len(results) > 0

        # Check for negative sentiment
        negative_results = [r for r in results if r.sentiment == 'negative']
        assert len(negative_results) > 0

    def test_mixed_sentiments(self, analyzer):
        #Test detection of mixed sentiments.
        text = "The food was amazing but the service was slow."
        results = analyzer.analyze(text)

        assert len(results) >= 2

        # There should be different sentiments
        sentiments = {r.sentiment for r in results}
        assert len(sentiments) > 1

    def test_confidence_range(self, analyzer):
        #Test that confidence scores are in valid range.
        text = "The food was excellent."
        results = analyzer.analyze(text)

        for result in results:
            assert 0.0 <= result.confidence <= 1.0

    def test_analyze_with_given_aspects(self, analyzer):
        #Test analyzing with user-provided aspects.
        text = "Great experience overall."
        aspects = ["food", "service"]

        results = analyzer.analyze_with_given_aspects(text, aspects)

        assert len(results) == 2
        detected_aspects = {r.aspect for r in results}
        assert detected_aspects == {"food", "service"}

class TestABSAAnalyzerInterface:
    #Test the abstract base class interface.

    def test_cannot_instantiate_abstract_class(self):
        #Test that ABSAAnalyzer cannot be instantiated directly.
        with pytest.raises(TypeError):
            ABSAAnalyzer()

    def test_get_name(self):
        #Test get_name method.
        analyzer = TransformerABSA()
        assert analyzer.get_name() == "TransformerABSA"


