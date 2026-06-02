# Machine Translation and Relation Extraction with Attention

Interactive demo for Machine Translation and Relation Extraction with Attention, inspired by Bahdanau et al. (2015) and implemented using Hugging Face Transformers.

##

> Python 3.10
> Node 24

##

## Stack
- **Frontend**: React (Vite)
- **Backend**: FastAPI
- **NLP**: Hugging Face Transformers (`Helsinki-NLP/opus-mt-en-de`, `Davlan/bert-base-multilingual-cased-ner-hrl`)

## Features
- English -> German translation
- Side-by-side:
  - fixed-vector baseline model (**without attention**)
  - translation **with attention**
- Attention heatmap:
  - rows: generated target tokens
  - columns: source tokens
  - cell intensity: attention weight
- Preset buttons:
  - **Short Example**
  - **Long Example**
  - **Paper Example**
- Relation extraction on translated text:
  - named entities
  - simple relation triples
  - lightweight graph view

## Connection to Bahdanau et al. (2015)

This demo is built around the core ideas from *"Neural Machine Translation by
Jointly Learning to Align and Translate"* (ICLR 2015 oral):

| Demo feature | Paper concept |
|---|---|
| **Translation with attention** | The proposed encoder-decoder with a soft-alignment mechanism that searches the source sentence at each decoding step |
| **Baseline (fixed-vector)** | The RNN Encoder-Decoder baseline (Cho et al., 2014) that compresses the whole source into one vector |
| **Attention heatmap** | The alignment matrix *α* (Section 3.1, Figure 3 in the paper) showing which source tokens each target token focuses on |
| **Peak alignment / Entropy** | Quantitative measures of attention sharpness — analogous to the qualitative "alignment agrees with intuition" analysis in the paper |
| **Long vs short examples** | The paper's key claim: fixed-vector bottleneck degrades *longer* sentences more (Section 2, last paragraph; Section 4, Table 1) |
| **Relation extraction** | Downstream task showing that better translation (via attention) preserves named entities and relationships, enabling IE |

Layman's terms for the metrics shown in the UI:
- **Peak alignment**: how strongly attention locks onto a single source word at each step on average. Higher means more "laser-focused" attention.
- **Entropy**: how spread out attention is on average. Lower means more focused; higher means more diffuse.

### Implementation notes

- The **attention model** uses `Helsinki-NLP/opus-mt-en-de` (MarianMT), a
  Transformer-based model. We extract its cross-attention from the last
  decoder layer, averaged over attention heads.
- The **baseline** runs the full encoder on the complete sentence (no text
  modification), then subsamples the encoder's output hidden states — keeping
  the first 4 and last 3 tokens, taking every 3rd from the middle — and feeds
  the compressed hidden states directly to the decoder via `encoder_outputs`.
  This simulates the information-loss bottleneck at the representation level
  rather than by degrading the input text.
- Relation extraction uses `Davlan/bert-base-multilingual-cased-ner-hrl` for
  NER, then derives adjacency-based relation triples between consecutive
  entities — a simplified approach suitable for a live demo.

## Run

### 1. Backend
> Create a virtual environment and install dependencies from `backend/requirements.txt` before running the server.
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Frontend
```bash
cd frontend
yarn install
yarn dev
```

Open: `http://localhost:5173`

## API
- `GET /api/examples`
- `POST /api/demo`
  - payload:
    ```json
    {
      "text": "The agreement on the European Economic Area was signed in August 1992.",
      "show_attention": true,
      "show_relations": true
    }
    ```
