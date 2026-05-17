import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, sentences, word2idx, tag2idx, max_len=128):
        self.sentences = sentences
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        words = [w for w, _, _ in sent]
        tags = [t for _, _, t in sent]

        word_ids = [self.word2idx.get(w.lower(), 1) for w in words]
        tag_ids = [self.tag2idx.get(t, 0) for t in tags]

        word_ids = word_ids[:self.max_len]
        tag_ids = tag_ids[:self.max_len]
        length = len(word_ids)

        pad_len = self.max_len - length
        word_ids += [0] * pad_len
        tag_ids += [0] * pad_len
        mask = [True] * length + [False] * pad_len

        return {
            "input_ids": torch.tensor(word_ids, dtype=torch.long),
            "tags": torch.tensor(tag_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "length": length,
        }


class BertNERDataset(Dataset):
    def __init__(self, sentences, tokenizer_wrapper, tag2idx):
        self.sentences = sentences
        self.tokenizer = tokenizer_wrapper
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        words = [w for w, _, _ in sent]
        tags = [t for _, _, t in sent]
        return self.tokenizer.encode_sentence(words, tags, self.tag2idx)
