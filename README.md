# Aspect-Based Sentiment Analysis (ABSA)

This repository implements multiple approaches to Aspect-Based Sentiment Analysis for short reviews (e.g., restaurant reviews). Given a piece of text, each implementation extracts aspect terms (e.g., food, service, ambiance) and predicts a sentiment label for each aspect with a confidence score.

Implementations provided under src/:
- LexiconABSA: rule/lexicon-based approach using spaCy + NLTK VADER (no external ML models).
- TransformerABSA: transformer-based classifier using a pretrained Hugging Face model.
- LLMABSA: large language model approach via a local Ollama model with JSON parsing and fallbacks.


## Project overview
- Goal: Extract (aspect, sentiment, confidence) triplets from input text.
- Core abstraction: ABSAAnalyzer (base class) defines the analyze(text) method returning a list of AspectSentiment.
- Data classes:
  - AspectSentiment(aspect: str, sentiment: str, confidence: float, text_span: Optional[Tuple[int, int]] = None)
- Typical usage: choose an analyzer, call analyze on a review, and use results downstream or aggregate to overall sentiment.

Repository layout (ASCII tree):
```
project-root
├── README.md
├── requirements.txt
├── data
│   ├── test_samples.json
│   └── evaluation_data.json
├── notebooks
│   └── comparison.ipynb
├── src
│   ├── base.py
│   ├── lexicon_absa.py
│   ├── transformer_absa.py
│   └── llm_absa.py
└── tests
    └── test_llm.py
```

## Setup instructions

Prerequisites
- Python 3.9+ recommended
- Internet access for downloading models (first run)
- Optional GPU: CUDA-enabled PyTorch for the TransformerABSA model

1) Create and activate a virtual environment (recommended)
- macOS/Linux:
```
python -m venv .venv
source .venv/bin/activate
```
- Windows (PowerShell):
```
python -m venv .venv
.venv\\Scripts\\Activate.ps1
```

2) Install Python dependencies (includes spaCy and the en_core_web_sm model)
```
pip install -r requirements.txt
```
Additionally, you have to install the english model for spacy
```
python -m spacy download en_core_web_sm
```

3) Install Ollama and pull a local model (for LLMABSA)
- macOS (Homebrew):
```
brew install ollama
ollama serve  # start the daemon (keep running)
```
- Linux:
```
curl -fsSL https://ollama.com/install.sh | sh
ollama serve  # start the daemon (keep running)
```
- Windows (PowerShell):
```
winget install Ollama.Ollama
ollama serve  # start the service in a terminal
```
Pull a small default model used by this repo:
```
ollama pull llama3.2:1b
```
NLTK data
- LexiconABSA downloads 'vader_lexicon' and 'stopwords' automatically on first use.
- If running in a restricted environment, pre-download with:
```
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"
```


## Dependencies and installation guide
Main dependencies (see requirements.txt for full list):
- spacy, nltk, vaderSentiment
- transformers, torch, sentencepiece
- ollama (Python client for local LLM runtime)
- pandas, numpy, tqdm, pydantic (utils and notebooks)
- pytest, pytest-cov (testing)
- jupyter, notebook, matplotlib, seaborn (exploration)

Install all with:
````
  pip install -r requirements.txt
````


## Usage examples
Below are basic Python examples for each implementation. All analyzers return List[AspectSentiment].

Common utilities \
  from src.base import AspectSentiment \
  text = "The food was amazing but the service was slow, though the patio was lovely."

***LexiconABSA (no external ML downloads beyond spaCy/NLTK)*** \
````
  from src.lexicon_absa import LexiconABSA
  analyzer = LexiconABSA()
  aspects = analyzer.analyze(text)
  for a in aspects:
      print(a.aspect, a.sentiment, a.confidence)
````
***TransformerABSA (downloads model on first run; GPU used if available)***\
````
  from src.transformer_absa import TransformerABSA
  analyzer = TransformerABSA(model_name="yangheng/deberta-v3-base-absa-v1.1")
  aspects = analyzer.analyze(text)
  for a in aspects:
      print(a.aspect, a.sentiment, a.confidence)
````
***LLMABSA (requires Ollama running locally and the specified model pulled)***\
````
  from src.llm_absa import LLMABSA
  analyzer = LLMABSA(model_name="llama3.2:1b", temperature=0.0)
  aspects = analyzer.analyze(text)
  for a in aspects:
      print(a.aspect, a.sentiment, a.confidence)
````
***Batch usage with LLMABSA***
````
  from src.llm_absa import LLMABSA
  analyzer = LLMABSA()
  reviews = [
      "Loved the pasta but the service was slow.",
      "Great ambiance, okay food."
  ]
  batch_results = analyzer.analyze_batch(reviews)
  for i, aspects in enumerate(batch_results):
      print(f"Review {i}:")
      for a in aspects:
          print("  ", a.aspect, a.sentiment, a.confidence)
````
Notebook
- See notebooks/comparison.ipynb for qualitative output and simple evaluation/aggregation helpers.


## API documentation
Base types
- AspectSentiment
  - Fields:
    - aspect: str — extracted aspect term
    - sentiment: str — one of {"positive", "neutral", "negative"}
    - confidence: float — 0.0 to 1.0
    - text_span: Optional[Tuple[int, int]] — start/end indices if available (may be None)
  - Methods:
    - to_dict() -> dict: serialize rounded confidence and fields
    - __str__(): human-readable one-line summary

- ABSAAnalyzer (abstract)
  - analyze(text: str) -> List[AspectSentiment]

Implementations
- LexiconABSA (src/lexicon_absa.py)
  - __init__(): loads spaCy en_core_web_sm, NLTK stopwords, and VADER
  - analyze(text: str) -> List[AspectSentiment]:
    - Extracts candidate aspects with noun chunking and compounds
    - Associates nearby opinion words; applies VADER polarity, negation handling
    - Averages scores and thresholds to produce sentiment + confidence

- TransformerABSA (src/transformer_absa.py)
  - __init__(model_name: str = "yangheng/deberta-v3-base-absa-v1.1"):
    - Loads tokenizer and model via transformers; moves to CUDA if available
    - Also loads spaCy en_core_web_sm for aspect extraction
  - analyze(text: str) -> List[AspectSentiment]:
    - Extracts noun aspects with spaCy and predicts sentiment for each using the transformer classifier

- LLMABSA (src/llm_absa.py)
  - __init__(model_name: str = "llama3.2:1b", temperature: float = 0.0, max_retries: int = 1, cache_responses: bool = True, timeout: int = 30)
  - analyze(text: str) -> List[AspectSentiment]
    - Builds a constrained JSON prompt, calls Ollama chat, parses JSON with robust fallbacks
  - analyze_batch(texts: List[str]) -> List[List[AspectSentiment]]
  - Notes:
    - Requires the specified Ollama model to be available locally (e.g., run: ollama pull llama3.2:1b)

Exceptions and validation
- AspectSentiment ensures sentiment in {positive, neutral, negative} and confidence in [0.0, 1.0]; ValueError otherwise.
- TransformerABSA may raise RuntimeError if model download fails.


## Running tests
  pytest -q


## Troubleshooting
- spaCy model not found: run python -m spacy download en_core_web_sm
- Transformer model download issues: ensure internet access and sufficient disk; try setting HF_HOME
- CUDA not available: the transformer falls back to CPU automatically (slower)
- Ollama errors: ensure the daemon is running (ollama serve) and the model has been pulled
- NLTK downloads blocked: pre-download required resources as shown above


## License
This repository includes third-party models subject to their respective licenses (Hugging Face models, spaCy models, Ollama model weights).The code in this repository is provided as-is without warranty.
