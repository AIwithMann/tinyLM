import torch
import torch.nn as nn 
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 128):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.register_buffer('positions', torch.arange(max_len))

    def forward(self, x):
        #x: (batch, seq_len, d_model)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), x.size(1))
        return x + self.pe(positions)


class ChatBot(nn.Module):
    def __init__(self, hidden_dim:int, heads:int, dropout:float, vocab_size:int, max_seq_len:int, layers:int, activation:str = 'gelu'):
        super().__init__()
        self.pe = PositionalEncoding(hidden_dim, max_len=max_seq_len)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decoderLayers = nn.ModuleList([
            nn.TransformerEncoderLayer( #Using encoder layers ecause decoder layers had some errors
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim*4,
                dropout=dropout,
                activation=activation,
                batch_first=True  # easier batch-first shape
            )
            for _ in range(layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
    def generate_square_subsequent_mask(self, sz: int):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    def forward(self,  x):
        out = self.token_embedding(x)
        out = self.pe(out)
        mask = self.generate_square_subsequent_mask(out.size(1)).to(out.device)
        for layer in self.decoderLayers:
            out = layer(out,  src_mask = mask)
        logits = self.output_layer(out)
        return logits