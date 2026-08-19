"""RNN encoder-decoder building blocks."""
import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim=None):
        super().__init__()
        attn_dim = attn_dim or hidden_dim
        self.W1 = nn.Linear(hidden_dim, attn_dim)
        self.W2 = nn.Linear(2 * hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_pad_mask):
        # decoder_hidden: [B, H]
        # encoder_outputs: [B, S, 2H]
        # src_pad_mask: [B, S] bool (True = pad)
        query = self.W1(decoder_hidden).unsqueeze(1)
        keys = self.W2(encoder_outputs)
        scores = self.v(torch.tanh(query + keys)).squeeze(-1)

        scores = scores.masked_fill(src_pad_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, alpha
