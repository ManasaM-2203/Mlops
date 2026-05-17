import torch
import torch.nn as nn


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.tagset_size = tagset_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2,
            num_layers=1, bidirectional=True, batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))
        self.start_transitions = nn.Parameter(torch.randn(tagset_size))
        self.end_transitions = nn.Parameter(torch.randn(tagset_size))

    def _get_emissions(self, x):
        embeds = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(self.dropout(lstm_out))
        return emissions

    def _forward_alg(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)

        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _score_sentence(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            trans = self.transitions[tags[:, i - 1], tags[:, i]]
            emit = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            step = trans + emit
            score += step * mask[:, i].float()

        last_idx = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        return score

    def loss(self, x, tags, mask):
        emissions = self._get_emissions(x)
        forward_score = self._forward_alg(emissions, mask)
        gold_score = self._score_sentence(emissions, tags, mask)
        return (forward_score - gold_score).mean()

    def predict(self, x, mask):
        emissions = self._get_emissions(x)
        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
            history.append(indices)

        score += self.end_transitions
        _, best_last = score.max(dim=1)

        best_paths = [best_last]
        for hist in reversed(history):
            best_last = hist.gather(1, best_last.unsqueeze(1)).squeeze(1)
            best_paths.append(best_last)

        best_paths.reverse()
        return torch.stack(best_paths, dim=1)
