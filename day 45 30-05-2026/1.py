"""Bai 1 - Tu code kien truc Transformer Decoder nhieu lop.

Yeu cau:
- Code lai Transformer Decoder nhung KHONG duoc dung nn.TransformerDecoderLayer.
- Su dung cac tang co ban nhu Linear, LayerNorm, Dropout de tao thanh 1 Decoder Layer.
- Xep chong nhieu Decoder Layer thanh 1 Transformer Decoder.

Comment tieng Viet khong dau theo phong cach quen thuoc.
"""

import math
import torch
import torch.nn as nn

# ----------------------------- CAC LOP CO BAN -----------------------------

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention dung chung cho ca Self-Attention va Cross-Attention."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim phai chia het cho num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # 1. Tuyen tinh Q, K, V roi cat thanh num_heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Tinh attention score (Dot product)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # mask co shape (batch, 1, seq_len, seq_len) hoac tuong tu de broadcast
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. Softmax de ra trong so
        attn_weights = torch.softmax(scores, dim=-1)

        # 4. Nhan voi Value
        context = torch.matmul(attn_weights, V) # (batch, num_heads, seq_len, head_dim)

        # 5. Gop cac head lai
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        return self.W_o(context)


class PositionwiseFeedForward(nn.Module):
    """Mang Feed Forward phi tuyen tinh cho tung vi tri."""
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


# ----------------------------- DECODER LAYER -----------------------------

class DecoderLayer(nn.Module):
    """Mot khoi Decoder (Decoder Layer). Giao hop cua:
    1. Masked Multi-Head Self Attention
    2. Multi-Head Cross Attention (nhin sang Encoder)
    3. Position-wise Feed Forward
    Kem theo cac khoi Add & Norm (Residual + LayerNorm).
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, trg_mask, src_mask):
        # 1. Masked Self-Attention: chi nhin cac tu da sinh ra (khong nhin tuong lai)
        attn_out = self.self_attn(x, x, x, mask=trg_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Cross-Attention: Decoder (Q) chu y vao Encoder (K, V)
        # mask o day la src_mask de bo qua cac padding cua cau goc
        attn_out = self.cross_attn(x, enc_out, enc_out, mask=src_mask)
        x = self.norm2(x + self.dropout(attn_out))

        # 3. Feed Forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x


# ----------------------------- TRANSFORMER DECODER -----------------------------

class TransformerDecoder(nn.Module):
    """Ket hop nhieu lop DecoderLayer de tao ra Transformer Decoder."""
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, dropout=0.1, max_len=1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional Encoding don gian bang cach hoc Embedding hoac dung Sinusoidal
        # O day the hien Positional Embedding co the hoc duoc cho gon
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, trg, enc_out, trg_mask, src_mask):
        # trg: (batch_size, trg_len)
        batch_size, trg_len = trg.shape
        
        # Tao vi tri (0, 1, 2, ..., trg_len-1)
        positions = torch.arange(0, trg_len).unsqueeze(0).expand(batch_size, trg_len).to(trg.device)
        
        # Embedding = Token Embedding + Position Embedding
        x = self.dropout(
            (self.token_embedding(trg) * math.sqrt(self.embed_dim)) + self.pos_embedding(positions)
        )

        # Di qua tung lop Decoder Layer
        for layer in self.layers:
            x = layer(x, enc_out, trg_mask, src_mask)
            
        # Ra lopp phan loai tu vung
        out = self.fc_out(x)
        return out


# ----------------------------- MAIN DE KIEM TRA -----------------------------
def main():
    print("Kiem tra kien truc Transformer Decoder tu code...")
    
    # Tham so dummy
    batch_size = 2
    trg_len = 5
    src_len = 7
    embed_dim = 256
    num_heads = 8
    vocab_size = 10000
    
    # Khoi tao model
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=3,
        num_heads=num_heads,
        ff_dim=512
    )
    
    # Tao tensor gia
    trg = torch.randint(0, vocab_size, (batch_size, trg_len))
    enc_out = torch.randn(batch_size, src_len, embed_dim)
    
    # Mask gia (Khong quan tam den viec mask that su dung hay chua trong test nay)
    trg_mask = torch.ones(batch_size, 1, trg_len, trg_len)
    src_mask = torch.ones(batch_size, 1, 1, src_len)
    
    out = decoder(trg, enc_out, trg_mask, src_mask)
    
    print(f"Dau vao trg shape      : {trg.shape}")
    print(f"Dau ra enc_out shape   : {enc_out.shape}")
    print(f"Dau ra decoder (logits): {out.shape}")
    print("Thanh cong! Decoder chay on ma khong can nn.TransformerDecoderLayer.")

if __name__ == "__main__":
    main()
