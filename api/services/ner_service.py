import json
import re

import torch

from pipeline.config import ROOT
from api.services.translation.transliterate_util import clean_hindi_artifacts


class NERService:
    def __init__(self):
        self.model = None
        self.model_type = None
        self._load_best_model()

        from api.services.lexicon_service import LexiconService
        self.lexicon = LexiconService()

    def _load_best_model(self):
        info_path = ROOT / "best_model_info.json"
        if not info_path.exists():
            print("[ner_service] no best_model_info.json, falling back to crf")
            self.model_type = "crf"
        else:
            with open(info_path) as f:
                info = json.load(f)
            self.model_type = info["model"]

        print(f"[ner_service] loading {self.model_type}")
        loaders = {
            "crf": self._load_crf,
            "bilstm_crf": self._load_bilstm_crf,
            "bert_ner": self._load_bert_ner,
            "distilbert_ner": self._load_distilbert_ner,
        }
        loaders[self.model_type]()

    def _load_crf(self):
        from pipeline.models.crf_model import CRFModel
        model_path = ROOT / "saved_models" / "crf" / "model.crfsuite"
        self.model = CRFModel()
        self.model.load(model_path)

    def _load_bilstm_crf(self):
        from pipeline.models.bilstm_crf import BiLSTMCRF
        processed = ROOT / "data" / "processed"

        with open(processed / "word2idx.json") as f:
            self.word2idx = json.load(f)
        with open(processed / "tag2idx.json") as f:
            self.tag2idx = json.load(f)
        with open(processed / "idx2tag.json") as f:
            self.idx2tag = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiLSTMCRF(
            vocab_size=len(self.word2idx),
            tagset_size=len(self.tag2idx),
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(ROOT / "saved_models" / "bilstm_crf" / "model.pt", weights_only=True, map_location=self.device)
        )
        self.model.eval()

    def _load_bert_ner(self):
        from pipeline.models.bert_ner import BertNER, BertNERTokenizer
        processed = ROOT / "data" / "processed"

        with open(processed / "tag2idx.json") as f:
            self.tag2idx = json.load(f)
        with open(processed / "idx2tag.json") as f:
            self.idx2tag = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = ROOT / "saved_models" / "bert_ner"

        self.bert_tokenizer = BertNERTokenizer.__new__(BertNERTokenizer)
        from transformers import BertTokenizerFast
        self.bert_tokenizer.tokenizer = BertTokenizerFast.from_pretrained(str(model_dir / "tokenizer"))
        self.bert_tokenizer.max_len = 128

        self.model = BertNER("bert-base-uncased", len(self.tag2idx)).to(self.device)
        self.model.load_state_dict(
            torch.load(model_dir / "model.pt", weights_only=True, map_location=self.device)
        )
        self.model.eval()

    def _load_distilbert_ner(self):
        from pipeline.models.distilbert_ner import DistilBertNER, DistilBertNERTokenizer
        processed = ROOT / "data" / "processed"

        with open(processed / "tag2idx.json") as f:
            self.tag2idx = json.load(f)
        with open(processed / "idx2tag.json") as f:
            self.idx2tag = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = ROOT / "saved_models" / "distilbert_ner"

        self.bert_tokenizer = DistilBertNERTokenizer.__new__(DistilBertNERTokenizer)
        from transformers import DistilBertTokenizerFast
        self.bert_tokenizer.tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir / "tokenizer"))
        self.bert_tokenizer.max_len = 128

        self.model = DistilBertNER("distilbert-base-uncased", len(self.tag2idx)).to(self.device)
        self.model.load_state_dict(
            torch.load(model_dir / "model.pt", weights_only=True, map_location=self.device)
        )
        self.model.eval()

    def predict(self, text: str) -> tuple[list[str], list[str], list[float], str]:
        text = clean_hindi_artifacts(text)
        tokens = _tokenize(text)
        if not tokens:
            return [], [], [], text

        confidences = [1.0] * len(tokens)
        if self.model_type == "crf":
            tags, confidences = self.model.predict_tokens_with_confidence(tokens)
        elif self.model_type == "bilstm_crf":
            tags = self._predict_bilstm(tokens)
        elif self.model_type in ("bert_ner", "distilbert_ner"):
            tags = self._predict_bert(tokens)

        return tokens, tags, confidences, text

    def _predict_bilstm(self, tokens):
        ids = [self.word2idx.get(t.lower(), 1) for t in tokens]
        length = len(ids)
        pad_len = 128 - length
        ids += [0] * pad_len
        mask = [True] * length + [False] * pad_len

        x = torch.tensor([ids], dtype=torch.long).to(self.device)
        m = torch.tensor([mask], dtype=torch.bool).to(self.device)

        with torch.no_grad():
            preds = self.model.predict(x, m)[0].cpu().tolist()

        return [self.idx2tag.get(str(p), "O") for p in preds[:length]]

    def _predict_bert(self, tokens):
        encoding, word_ids = self.bert_tokenizer.encode_for_inference(tokens)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)[0].cpu().tolist()

        tags = []
        prev_wid = None
        for pid, wid in zip(preds, word_ids):
            if wid is None or wid == prev_wid:
                continue
            tags.append(self.idx2tag.get(str(pid), "O"))
            prev_wid = wid

        return tags[:len(tokens)]

    def extract_entities(self, tokens, tags, text=None, confidences=None, manual_entities=None):
        entities = []
        current_entity = None
        last_found_pos = 0

        for i, (token, tag) in enumerate(zip(tokens, tags)):
            # Robustly find the token position in the text starting from the last found position
            # This handles varying whitespace, punctuation, etc.
            if text:
                start = text.find(token, last_found_pos)
                if start == -1:
                    # Fallback if somehow not found (shouldn't happen with correct tokens)
                    start = last_found_pos
                end = start + len(token)
                last_found_pos = end
            else:
                # Fallback to naive logic if text is not provided
                start = last_found_pos
                end = start + len(token)
                last_found_pos = end + 1

            conf = confidences[i] if confidences else 1.0

            if tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "label": tag[2:],
                    "start": start,
                    "end": end,
                    "confidence": conf
                }
            elif tag.startswith("I-") and current_entity:
                # Append the gap between tokens if we have text
                if text:
                    gap = text[current_entity["end"]:start]
                    current_entity["text"] += gap + token
                else:
                    current_entity["text"] += " " + token
                current_entity["end"] = end
                current_entity["confidence"] = min(current_entity.get("confidence", 1.0), conf)
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        # Reinforce with Lexicon (Gazetteer) matches
        if text:
            lexicon_matches = self.lexicon.find_matches(text)
            for lex in lexicon_matches:
                # Only add if it doesn't overlap with model predictions
                if not any(_overlap(lex, existing) for existing in entities):
                    entities.append(lex)
            # Re-sort after adding lexicon matches
            entities = sorted(entities, key=lambda e: e["start"])

        if manual_entities:
            final_entities = list(manual_entities)
            for pred in entities:
                if not any(_overlap(pred, man) for man in manual_entities):
                    final_entities.append(pred)
            entities = sorted(final_entities, key=lambda e: e["start"])

        return entities


def _overlap(e1, e2):
    return not (e1["end"] <= e2["start"] or e2["end"] <= e1["start"])


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text)
