import pandas as pd
from pipeline.config import ROOT


def load_raw(path: str) -> pd.DataFrame:
    filepath = ROOT / path
    df = pd.read_csv(filepath, encoding="latin1")
    df.columns = ["sentence_id", "word", "pos", "tag"]
    df["sentence_id"] = df["sentence_id"].ffill()
    df = df.dropna(subset=["word", "tag"])
    return df


def group_sentences(df: pd.DataFrame) -> list[list[tuple[str, str, str]]]:
    grouped = df.groupby("sentence_id")
    sentences = []
    for _, group in grouped:
        tokens = list(zip(group["word"], group["pos"], group["tag"]))
        sentences.append(tokens)
    return sentences
