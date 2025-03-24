
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionWithMaskBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout= 0):
        super(MultiHeadAttentionWithMaskBlock, self).__init__()

        assert dim%num_heads==0, "dim should be divisible by num_heads"

        self.dim= dim
        self.heads = num_heads
        self.per_head = dim // num_heads

        self.query_layer = nn.Linear(dim, dim, bias= False)
        self.keys_layer = nn.Linear(dim, dim, bias= False)
        self.values_layer = nn.Linear(dim, dim, bias= False)
        self.linear_layer = nn.Linear(dim, dim, bias= False)
        self.dropout = nn.Dropout(dropout)

    def split_head(self, tensor: Tensor):
        batch_size, num_tokens, dim = tensor.size()
        return tensor.view(batch_size, num_tokens, self.heads, self.per_head).transpose(1, 2)

    def forward(self, query, keys, values, mask= None):

        query= self.split_head(self.query_layer(query))
        keys= self.split_head(self.keys_layer(keys))
        values= self.split_head(self.values_layer(values))

        weights = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.per_head)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = F.softmax(weights, dim= -1)
        weights = self.dropout(weights)

        weighted_values = torch.matmul(weights, values)
        weighted_values = weighted_values.transpose(1,2).contiguous().reshape(query.shape[0], -1, self.dim)
        attention = self.linear_layer(weighted_values)

        return attention


class FeedForwardNetworkBlock(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super(FeedForwardNetworkBlock, self).__init__()

        self.ff1 = nn.Linear(dim, inter_dim, bias=True)
        self.ff2 = nn.Linear(inter_dim, dim, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, attention):
        return self.ff2(self.relu(self.ff1(attention)))


class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size, dim):
        super(TransformerEmbeddings, self).__init__()

        self.embedding = nn.Embedding(vocab_size, dim)
        self.dim = dim

    def forward(self, tokens):
        return self.embedding(tokens) * math.sqrt(self.dim)


class PositionalEncodingBlock(nn.Module):
    def __init__(self, max_token_length, dim):
        super(PositionalEncodingBlock, self).__init__()

        pe= torch.zeros(max_token_length, dim)
        position = torch.arange(0, max_token_length, dtype=torch.float).unsqueeze(1)
        div_alt = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        # div = 1 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        pe[:, 0::2] = torch.sin(position * div_alt)
        pe[:, 1::2] = torch.cos(position * div_alt)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, tokens):
        return tokens + self.pe[:, :tokens.size(1), :].requires_grad_(False)
    

class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, dim: int, inter_dim: int, dropout: float = 0):
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttentionWithMaskBlock(dim, num_heads, dropout)
        self.ff = FeedForwardNetworkBlock(dim, inter_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embeddings, input_mask):
        mha_res = self.mha(token_embeddings, token_embeddings, token_embeddings, input_mask)
        add_norm1 = self.norm1(token_embeddings + self.dropout(mha_res))
        ff_res = self.ff(add_norm1)
        add_norm2 = self.norm2(add_norm1 + self.dropout(ff_res))
        return add_norm2


class DecoderBlock(nn.Module):
    def __init__(self, num_heads: int, dim: int, inter_dim: int, dropout: float = 0):
        super(DecoderBlock, self).__init__()

        self.self_attention = MultiHeadAttentionWithMaskBlock(dim, num_heads, dropout)
        self.cross_attention = MultiHeadAttentionWithMaskBlock(dim, num_heads, dropout)
        self.ff = FeedForwardNetworkBlock(dim, inter_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, output_embeddings, encoder_output, source_mask, target_mask):
        self_att_res = self.self_attention(output_embeddings, output_embeddings, output_embeddings, target_mask)
        add_norm1 = self.norm1(output_embeddings + self.dropout(self_att_res))
        cross_att_res = self.cross_attention(add_norm1, encoder_output, encoder_output, source_mask)
        add_norm2 = self.norm2(add_norm1 + self.dropout(cross_att_res))
        ff_res = self.ff(add_norm2)
        add_norm3 = self.norm3(add_norm2 + self.dropout(ff_res))
        return add_norm3


class TransformerModule(nn.Module):
    def __init__(self, dim, vocab_size, max_token_length, num_heads, num_layers, dropout = 0):
        super(TransformerModule, self).__init__()

        self.embedding = TransformerEmbeddings(vocab_size, dim)
        self.positional_encoder = PositionalEncodingBlock(max_token_length, dim)

        self.encoder_layers = nn.ModuleList([EncoderBlock(num_heads, dim, 4* dim, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(num_heads, dim, 4* dim, dropout) for _ in range(num_layers)])
        
        self.linear = nn.Linear(dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, input, target):
        input_mask = (input != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (target != 0).unsqueeze(1).unsqueeze(2)
        batch_size, sequence_length= target.size()
        tri = torch.tril(torch.ones(batch_size, 1, sequence_length, sequence_length))
        target_mask = torch.logical_and(target_mask, tri).long()

        return input_mask, target_mask

    def forward(self, input, target):
        input_mask, target_mask = self.generate_mask(input, target)

        input_embedding = self.dropout(self.positional_encoder(self.embedding(input)))
        output_embedding = self.dropout(self.positional_encoder(self.embedding(target)))

        encoder_output = input_embedding
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, input_mask)

        decoder_output = output_embedding
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, input_mask, target_mask)

        linear_output = self.linear(decoder_output)
        return linear_output
