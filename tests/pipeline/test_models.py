import torch

from pipeline.models.bilstm_crf import BiLSTMCRF
from pipeline.models.crf_model import sent_features, _guess_pos


def test_bilstm_crf_shapes():
    vocab_size, tagset_size = 100, 10
    model = BiLSTMCRF(vocab_size, tagset_size, embedding_dim=32, hidden_dim=64)

    batch_size, seq_len = 4, 20
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    tags = torch.randint(0, tagset_size, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    loss = model.loss(x, tags, mask)
    assert loss.shape == ()
    assert loss.item() > 0

    preds = model.predict(x, mask)
    assert preds.shape == (batch_size, seq_len)


def test_bilstm_crf_variable_lengths():
    model = BiLSTMCRF(50, 5, embedding_dim=16, hidden_dim=32)
    x = torch.randint(0, 50, (2, 10))
    mask = torch.tensor([
        [True] * 8 + [False] * 2,
        [True] * 5 + [False] * 5,
    ])
    preds = model.predict(x, mask)
    assert preds.shape == (2, 10)


def test_bilstm_crf_loss_decreases():
    model = BiLSTMCRF(30, 5, embedding_dim=16, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.randint(1, 30, (2, 8))
    tags = torch.randint(1, 5, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.bool)

    losses = []
    for _ in range(10):
        loss = model.loss(x, tags, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_crf_feature_extraction():
    sent = [("John", "NNP", "B-per"), ("lives", "VBZ", "O"), ("in", "IN", "O")]
    features = sent_features(sent)
    assert len(features) == 3

    f0 = features[0]
    assert f0["word.lower"] == "john"
    assert f0["word.istitle"] is True
    assert f0["BOS"] is True
    assert "+1:word.lower" in f0

    f2 = features[2]
    assert f2["EOS"] is True
    assert "-1:word.lower" in f2


def test_crf_feature_window():
    sent = [
        ("The", "DT", "O"),
        ("United", "NNP", "B-org"),
        ("Nations", "NNP", "I-org"),
        ("met", "VBD", "O"),
        ("today", "NN", "O"),
    ]
    features = sent_features(sent)
    f2 = features[2]
    assert "-2:word.lower" in f2
    assert "+2:word.lower" in f2


def test_guess_pos():
    assert _guess_pos("London") == "NNP"
    assert _guess_pos("42") == "CD"
    assert _guess_pos("the") == "DT"
    assert _guess_pos("in") == "IN"
    assert _guess_pos("my") == "PRP$"
    assert _guess_pos("he") == "PRP"
    assert _guess_pos("running") == "NN"
