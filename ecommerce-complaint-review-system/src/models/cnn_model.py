"""
TextCNN: Convolutional model for complaint text classification.
Uses multiple parallel kernel sizes (n-gram feature detectors) followed by
max-over-time pooling, then two output heads: complaint category and severity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_categories, num_severities,
                 embed_dim=100, num_filters=64, kernel_sizes=(2, 3, 4), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        fused_dim = num_filters * len(kernel_sizes)
        self.category_head = nn.Linear(fused_dim, num_categories)
        self.severity_head = nn.Linear(fused_dim, num_severities)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(emb))                 # (batch, num_filters, seq_len')
            pooled = F.max_pool1d(c, c.size(2)).squeeze(2)  # (batch, num_filters)
            conv_outs.append(pooled)
        features = torch.cat(conv_outs, dim=1)
        features = self.dropout(features)
        category_logits = self.category_head(features)
        severity_logits = self.severity_head(features)
        return category_logits, severity_logits
