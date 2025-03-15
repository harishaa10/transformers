
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionWithMaskBlock(nn.Module):
    def __init__(self, dim= 1024, num_heads= 4):
        super(MultiHeadAttentionWithMaskBlock, self).__init__()

        assert dim%num_heads==0, "dim should be divisible by num_heads"

        self.dim= dim
        self.heads = num_heads
        self.per_head = dim // num_heads

        self.query_layer = nn.Linear(dim, dim)
        self.keys_layer = nn.Linear(dim, dim)
        self.values_layer = nn.Linear(dim, dim)
        self.linear_layer = nn.Linear(dim, dim)

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
        weights = F.softmax(weights, dim= -2)

        weighted_values = torch.matmul(weights, values)
        weighted_values = weighted_values.transpose(1,2).contiguous().reshape(query.shape[0], -1, self.dim)
        attention = self.linear_layer(weighted_values)

        return attention


class FeedForwardNetworkBlock(nn.Module):
    def __init__(self, dim= 1024, inter_dim= 512):
        super(FeedForwardNetworkBlock, self).__init__()

        self.ff1 = nn.Linear(dim, inter_dim)
        self.ff2 = nn.Linear(inter_dim, dim)
        self.relu = nn.ReLU()
    
    def forward(self, attention):
        return self.ff2(self.relu(self.ff1(attention)))


class PositionalEncodingBlock(nn.Module):
    def __init__(self, max_token_length, dim):
        super(PositionalEncodingBlock, self).__init__()

        pe= torch.zeros(max_token_length, dim)
        position = torch.arange(0, max_token_length, dtype=torch.float).unsqueeze(1)
        # div_alt = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        div = 1 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, tokens):
        return tokens + self.pe[:, :tokens.size(1), :]
    

class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, dim: int, inter_dim: int, dropout: float = 0):
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttentionWithMaskBlock(dim, num_heads)
        self.ff = FeedForwardNetworkBlock(dim, inter_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embeddings, input_mask):
        mha_res = self.mha(token_embeddings, token_embeddings, token_embeddings, input_mask)
        mha_res = self.dropout(mha_res)
        add_norm1= self.norm1(torch.add(token_embeddings, mha_res))
        ff_res = self.ff(add_norm1)
        ff_res = self.dropout(ff_res)
        add_norm2 = self.norm2(torch.add(add_norm1, ff_res))
        return add_norm2


class DecoderBlock(nn.Module):
    def __init__(self, num_heads: int, dim: int, inter_dim: int, dropout: float = 0):
        super(DecoderBlock, self).__init__()

        self.self_attention = MultiHeadAttentionWithMaskBlock(dim, num_heads)
        self.cross_attention = MultiHeadAttentionWithMaskBlock(dim, num_heads)
        self.ff = FeedForwardNetworkBlock(dim, inter_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, output_embeddings, encoder_output, source_mask, target_mask):
        mmha_res = self.self_attention(output_embeddings, output_embeddings, output_embeddings, target_mask)
        mmha_res = self.dropout(mmha_res)
        add_norm1= self.norm1(torch.add(output_embeddings, mmha_res))
        mha_res = self.cross_attention(add_norm1, encoder_output, encoder_output, source_mask)
        mha_res = self.dropout(mha_res)
        add_norm2 = self.norm2(torch.add(add_norm1, mha_res))
        ff_res = self.ff(add_norm2)
        ff_res = self.dropout(ff_res)
        add_norm3 = self.norm3(torch.add(add_norm2, ff_res))
        return add_norm3


class TransformerModule(nn.Module):
    def __init__(self, dim, encoder_vocab_size, decoder_vocab_size, max_token_length, num_heads, num_layers, dropout = 0):
        super(TransformerModule, self).__init__()

        self.encoder_embedding = nn.Embedding(encoder_vocab_size, dim)
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, dim)
        self.positional_encoder = PositionalEncodingBlock(max_token_length, dim)

        self.encoder_layers = nn.ModuleList([EncoderBlock(num_heads, dim, 4* dim, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(num_heads, dim, 4* dim, dropout) for _ in range(num_layers)])
        
        self.linear = nn.Linear(dim, decoder_vocab_size)
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

        input_embedding = self.dropout(self.positional_encoder(self.encoder_embedding(input)))
        output_embedding = self.dropout(self.positional_encoder(self.decoder_embedding(target)))

        encoder_output = input_embedding
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, input_mask)

        decoder_output = output_embedding
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, input_mask, target_mask)

        linear_output = self.linear(decoder_output)
        return linear_output
