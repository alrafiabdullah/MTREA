import os
import re
import torch
import numpy as np

from typing import Any
from fastapi import FastAPI
from functools import lru_cache
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from transformers import MarianMTModel, MarianTokenizer, pipeline

APP_TITLE = "Machine Translation and Relation Extraction with Attention"
TRANSLATION_MODEL_NAME = os.getenv("TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-en-de")
NER_MODEL_NAME = os.getenv("NER_MODEL", "Davlan/bert-base-multilingual-cased-ner-hrl")

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DemoRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1500)
    show_attention: bool = True
    show_relations: bool = True


class DemoResponse(BaseModel):
    source_text: str
    baseline_translation: str
    attention_translation: str
    source_tokens: list[str]
    target_tokens: list[str]
    attention_matrix: list[list[float]]
    metrics: dict[str, float]
    entities: list[dict[str, Any]]
    relations: list[dict[str, str]]


def _clean_tokens(tokenizer: MarianTokenizer, ids: list[int]) -> list[str]:
    """
    Convert token IDs to clean strings, removing special tokens and handling subword markers.
    """
    tokens: list[str] = []
    for tok in tokenizer.convert_ids_to_tokens(ids):
        if tok in tokenizer.all_special_tokens:
            continue
        tokens.append(tok.replace("▁", "").strip() or "_")
    return tokens


def _attention_metrics(matrix: np.ndarray) -> dict[str, float]:
    """
    Compute simple metrics from the attention matrix:
    """
    eps = 1e-12
    if matrix.size == 0:
        return {"peak_alignment": 0.0, "entropy": 0.0}

    peak = float(np.mean(np.max(matrix, axis=1)))
    probs = matrix / (matrix.sum(axis=1, keepdims=True) + eps)
    entropy = float(-np.mean(np.sum(probs * np.log(probs + eps), axis=1)))
    return {"peak_alignment": peak, "entropy": entropy}


@lru_cache(maxsize=1)
def get_translation_stack() -> tuple[MarianTokenizer, MarianMTModel, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
    model = MarianMTModel.from_pretrained(
        TRANSLATION_MODEL_NAME, attn_implementation="eager"
    ).to(device)
    model.eval()
    return tokenizer, model, device


@lru_cache(maxsize=1)
def get_ner_pipeline():
    return pipeline(
        "ner",
        model=NER_MODEL_NAME,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
    )


def translate_with_attention(text: str) -> dict[str, Any]:
    tokenizer, model, device = get_translation_stack()
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            num_beams=4,
            max_new_tokens=512,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    pred_ids = generated.sequences[0].detach().cpu()
    translation = tokenizer.decode(pred_ids, skip_special_tokens=True)

    rows: list[torch.Tensor] = []
    for step in generated.cross_attentions or ():
        if not step:
            continue
        layer = step[-1][0].squeeze(1)
        if layer.ndim != 2:
            continue
        rows.append(layer.mean(dim=0).detach().cpu())

    matrix = torch.stack(rows, dim=0).numpy() if rows else np.zeros((1, 1), dtype=float)

    source_tokens = _clean_tokens(
        tokenizer, inputs["input_ids"][0].detach().cpu().tolist()
    )
    target_tokens = _clean_tokens(tokenizer, pred_ids.tolist())
    tgt_len = min(len(target_tokens), matrix.shape[0])
    src_len = min(len(source_tokens), matrix.shape[1])
    matrix = matrix[:tgt_len, :src_len]
    source_tokens = source_tokens[:src_len]
    target_tokens = target_tokens[:tgt_len]

    return {
        "translation": translation,
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "attention_matrix": matrix,
    }


def baseline_without_attention_simulation(text: str) -> str:
    """
    Simulate a fixed-context-vector baseline by compressing encoder hidden
    states rather than source text.

    The encoder runs on the FULL sentence (nothing artificial), then we
    subsample its output hidden states: keep the first few and last few
    tokens, compress the middle. This removes fine-grained detail that
    the decoder would normally attend to, simulating the information-loss
    bottleneck of a fixed-context encoder — without modifying the input text.

    For short sentences (< 12 tokens after encoding), the full sequence
    passes through so there is no visible degradation on simple examples.
    """
    tokenizer, model, device = get_translation_stack()
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        encoder_outputs = model.model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )

    hidden = encoder_outputs.last_hidden_state  # (1, src_len, hidden_dim)
    src_len = hidden.shape[1]
    orig_mask = inputs["attention_mask"]  # (1, src_len)

    if src_len >= 18:
        keep_front = 4
        keep_back = 3
        front = hidden[:, :keep_front, :]
        back = hidden[:, -keep_back:, :]
        middle = hidden[:, keep_front:-keep_back, :]
        middle_compressed = middle[:, ::3, :]
        compressed_hidden = torch.cat([front, middle_compressed, back], dim=1)

        new_src_len = compressed_hidden.shape[1]
        compressed_mask = torch.ones(
            (1, new_src_len), dtype=orig_mask.dtype, device=device
        )

        from transformers.modeling_outputs import BaseModelOutput

        compressed_encoder = BaseModelOutput(last_hidden_state=compressed_hidden)
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs["input_ids"],
                encoder_outputs=compressed_encoder,
                attention_mask=compressed_mask,
                max_new_tokens=512,
            )
    else:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512)

    return tokenizer.decode(out[0], skip_special_tokens=True)


def extract_relations(text: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    ner = get_ner_pipeline()
    entities_raw = ner(text)
    entities: list[dict[str, Any]] = []
    for ent in entities_raw:
        label = ent.get("entity_group", ent.get("entity", "ENTITY"))
        entities.append(
            {
                "text": ent["word"],
                "label": label,
                "start": int(ent.get("start", 0)),
                "end": int(ent.get("end", 0)),
                "score": float(ent.get("score", 0.0)),
            }
        )

    entities.sort(key=lambda x: x["start"])
    relations: list[dict[str, str]] = []
    for left, right in zip(entities, entities[1:]):
        between = text[left["end"] : right["start"]].strip()
        between = re.sub(r"\s+", " ", between)
        relation = between if between else "related_to"
        relation = relation[:40]
        relations.append(
            {"head": left["text"], "relation": relation, "tail": right["text"]}
        )

    return entities, relations


@app.get("/api/examples")
def get_examples() -> dict[str, str]:
    return {
        "short": "The cat is sleeping on the sofa.",
        "long": "Although the weather was terrible, the team decided to continue the experiment because the data was very important for the final report.",
        "paper": "The agreement on the European Economic Area was signed in August 1992.",
        "relation": "Barack Obama met Angela Merkel in Berlin and discussed NATO at the White House.",
    }


@app.post("/api/demo", response_model=DemoResponse)
def run_demo(payload: DemoRequest) -> DemoResponse:
    text = payload.text.strip()
    translation_data = translate_with_attention(text)
    baseline = baseline_without_attention_simulation(text)

    matrix = translation_data["attention_matrix"]
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, str]] = []
    if payload.show_relations:
        entities, relations = extract_relations(translation_data["translation"])

    attention_matrix = matrix.tolist() if payload.show_attention else []
    metrics = (
        _attention_metrics(matrix)
        if payload.show_attention
        else {
            "peak_alignment": 0.0,
            "entropy": 0.0,
        }
    )

    return DemoResponse(
        source_text=text,
        baseline_translation=baseline,
        attention_translation=translation_data["translation"],
        source_tokens=translation_data["source_tokens"],
        target_tokens=translation_data["target_tokens"],
        attention_matrix=attention_matrix,
        metrics=metrics,
        entities=entities,
        relations=relations,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
