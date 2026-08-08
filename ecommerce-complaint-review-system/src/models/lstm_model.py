"""
Bidirectional LSTM classifier for complaint text.
Captures longer-range dependencies and negation/contrast patterns
(e.g. "was fine but then completely broke") better than the plain RNN.
"""
import torch
import torch.nn as nn


class TextLSTM(nn.Module):
    def __init__(self, vocab_size, num_categories, num_severities,
                 embed_dim=100, hidden_dim=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        fused_dim = hidden_dim * 2
        self.category_head = nn.Linear(fused_dim, num_categories)
        self.severity_head = nn.Linear(fused_dim, num_severities)

    def forward(self, x):
        emb = self.embedding(x)                     # (batch, seq_len, embed_dim)
        output, (hidden, cell) = self.lstm(emb)
        h_fwd = hidden[-2]
        h_bwd = hidden[-1]
        features = torch.cat([h_fwd, h_bwd], dim=1)
        features = self.dropout(features)
        category_logits = self.category_head(features)
        severity_logits = self.severity_head(features)
        return category_logits, severity_logits
