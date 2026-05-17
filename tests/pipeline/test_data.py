import pandas as pd
import pytest

from pipeline.data.loader import group_sentences
from pipeline.data.preprocess import build_vocab


@pytest.fixture
def sample_df():
    data = {
        "sentence_id": ["Sentence: 1", None, None, "Sentence: 2", None],
        "word": ["John", "lives", "in", "New", "York"],
        "pos": ["NNP", "VBZ", "IN", "NNP", "NNP"],
        "tag": ["B-per", "O", "O", "B-geo", "I-geo"],
    }
    df = pd.DataFrame(data)
    df.columns = ["sentence_id", "word", "pos", "tag"]
    df["sentence_id"] = df["sentence_id"].ffill()
    return df


def test_group_sentences(sample_df):
    sentences = group_sentences(sample_df)
    assert len(sentences) == 2
    assert sentences[0][0] == ("John", "NNP", "B-per")
    assert len(sentences[0]) == 3
    assert len(sentences[1]) == 2


def test_sentence_structure(sample_df):
    sentences = group_sentences(sample_df)
    for sent in sentences:
        for token in sent:
            assert len(token) == 3
            word, _, tag = token
            assert isinstance(word, str)
            assert isinstance(tag, str)


def test_build_vocab(sample_df):
    sentences = group_sentences(sample_df)
    word2idx, tag2idx = build_vocab(sentences)

    assert "<PAD>" in word2idx
    assert "<UNK>" in word2idx
    assert word2idx["<PAD>"] == 0
    assert word2idx["<UNK>"] == 1
    assert "john" in word2idx

    assert "<PAD>" in tag2idx
    assert "B-per" in tag2idx
    assert "B-geo" in tag2idx
    assert "I-geo" in tag2idx
    assert "O" in tag2idx


def test_vocab_no_duplicates(sample_df):
    sentences = group_sentences(sample_df)
    word2idx, tag2idx = build_vocab(sentences)

    values = list(word2idx.values())
    assert len(values) == len(set(values))

    tag_values = list(tag2idx.values())
    assert len(tag_values) == len(set(tag_values))
