import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast


class BertNER(nn.Module):
    def __init__(self, model_name, num_tags, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(outputs.last_hidden_state))
        return logits


class BertNERTokenizer:
    def __init__(self, model_name, max_len=128):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.max_len = max_len

    def encode_sentence(self, words, tags, tag2idx):
        encoding = self.tokenizer(
            words, is_split_into_words=True,
            max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        word_ids = encoding.word_ids()

        label_ids = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_word_id:
                label_ids.append(tag2idx.get(tags[wid], 0))
            else:
                label_ids.append(-100)
            prev_word_id = wid

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }

    def encode_for_inference(self, words):
        encoding = self.tokenizer(
            words, is_split_into_words=True,
            max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        return encoding, encoding.word_ids()
