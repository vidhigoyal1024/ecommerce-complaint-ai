"""
Model Fusion: combines CNN, RNN, and LSTM predictions into a single
category + severity decision with a confidence score.

Strategy: weighted average of softmax probability distributions
("soft voting"). Weights default to equal (1/3 each) but can be tuned
based on each model's validation accuracy.
"""
import torch
import torch.nn.functional as F


def fuse_predictions(cnn_out, rnn_out, lstm_out, weights=(1 / 3, 1 / 3, 1 / 3)):
    """
    Each *_out is a tuple (category_logits, severity_logits) with shape (batch, num_classes).
    Returns dict with fused probabilities, predicted indices, and confidence.
    """
    w_cnn, w_rnn, w_lstm = weights

    cnn_cat_p = F.softmax(cnn_out[0], dim=-1)
    rnn_cat_p = F.softmax(rnn_out[0], dim=-1)
    lstm_cat_p = F.softmax(lstm_out[0], dim=-1)
    fused_cat_p = w_cnn * cnn_cat_p + w_rnn * rnn_cat_p + w_lstm * lstm_cat_p

    cnn_sev_p = F.softmax(cnn_out[1], dim=-1)
    rnn_sev_p = F.softmax(rnn_out[1], dim=-1)
    lstm_sev_p = F.softmax(lstm_out[1], dim=-1)
    fused_sev_p = w_cnn * cnn_sev_p + w_rnn * rnn_sev_p + w_lstm * lstm_sev_p

    cat_conf, cat_idx = torch.max(fused_cat_p, dim=-1)
    sev_conf, sev_idx = torch.max(fused_sev_p, dim=-1)

    return {
        "category_idx": cat_idx,
        "category_confidence": cat_conf,
        "category_probs": fused_cat_p,
        "severity_idx": sev_idx,
        "severity_confidence": sev_conf,
        "severity_probs": fused_sev_p,
        "per_model": {
            "cnn": {"category_probs": cnn_cat_p, "severity_probs": cnn_sev_p},
            "rnn": {"category_probs": rnn_cat_p, "severity_probs": rnn_sev_p},
            "lstm": {"category_probs": lstm_cat_p, "severity_probs": lstm_sev_p},
        },
    }
