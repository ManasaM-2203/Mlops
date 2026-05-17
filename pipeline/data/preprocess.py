import json
import pickle
import re
from pathlib import Path

from sklearn.model_selection import train_test_split

from pipeline.config import ROOT, load_params
from pipeline.data.loader import load_raw, group_sentences


def clean_hindi_artifacts(text):
    """
    Strips English alphabet characters from words containing Devanagari 
    characters to fix transliteration artifacts.
    """
    text_str = str(text)
    if re.search(r'[\u0900-\u097F]', text_str):
        cleaned = re.sub(r'[A-Za-z]', '', text_str)
        return cleaned if cleaned else text_str
    return text_str


def build_vocab(sentences):
    words, tags = set(), set()
    for sent in sentences:
        for word, _, tag in sent:
            words.add(word.lower())
            tags.add(tag)

    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for i, w in enumerate(sorted(words), start=2):
        word2idx[w] = i

    tag2idx = {"<PAD>": 0}
    for i, t in enumerate(sorted(tags), start=1):
        tag2idx[t] = i

    return word2idx, tag2idx


def run():
    params = load_params()
    data_params = params["data"]

    print("[preprocess] loading raw data")
    df = load_raw(data_params["raw_path"])
    
    # Clean transliteration noise from words
    if "word" in df.columns:
        df["word"] = df["word"].apply(clean_hindi_artifacts)
        
    sentences = group_sentences(df)
    print(f"[preprocess] {len(sentences)} sentences loaded")

    word2idx, tag2idx = build_vocab(sentences)
    idx2tag = {v: k for k, v in tag2idx.items()}

    train_val, test = train_test_split(
        sentences, test_size=data_params["test_size"], random_state=42
    )
    adjusted_val = data_params["val_size"] / (1 - data_params["test_size"])
    train, val = train_test_split(
        train_val, test_size=adjusted_val, random_state=42
    )

    print(f"[preprocess] train={len(train)} val={len(val)} test={len(test)}")

    out = Path(ROOT / data_params["processed_path"])
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "train.pkl", "wb") as f:
        pickle.dump(train, f)
    with open(out / "val.pkl", "wb") as f:
        pickle.dump(val, f)
    with open(out / "test.pkl", "wb") as f:
        pickle.dump(test, f)
    with open(out / "word2idx.json", "w") as f:
        json.dump(word2idx, f)
    with open(out / "tag2idx.json", "w") as f:
        json.dump(tag2idx, f)
    with open(out / "idx2tag.json", "w") as f:
        json.dump(idx2tag, f)

    print("[preprocess] done")


if __name__ == "__main__":
    run()
