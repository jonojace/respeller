import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
import math
from modules.gumbel_vector_quantizer import GumbelVectorQuantizer

# class EncoderDecoderRespeller(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder =
#         self.decoder =

def init_embedding_weights(source_tensor, target_tensor):
    """copy weights inplace from source tensor to target tensor"""
    target_tensor.requires_grad = False
    target_tensor.copy_(source_tensor.clone().detach())
    target_tensor.requires_grad = True

class EncoderRespeller(nn.Module):
    def __init__(
            self,
            n_symbols,
            pretrained_tts,
            d_model=512,
            nhead=4,
            num_layers=4,
            pretrained_embedding_table=None, # optional - initialise from pretrained grapheme embedding table from TTS
            freeze_embedding_table=False,
            batch_first=True,
            grapheme_embedding_dim=384,
            latent_temp=(2, 0.5, 0.999995),
    ):
        super().__init__()
        self.model_type = 'EncoderRespeller'
        self.embedding = nn.Embedding(n_symbols, d_model)
        self.batch_first = batch_first

        if pretrained_embedding_table is not None:
            initialise_embedding_table(self.embedding, pretrained_embedding_table)
            if freeze_embedding_table:
                # freeze_embedding_table.freeze_weights()
                raise NotImplementedError

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            batch_first=batch_first,
        )
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_layers,
            norm=encoder_norm,
        )
        # self.linear = Linear(d_model, out_vocab_size)
        self.quantiser = GumbelVectorQuantizer(
            in_dim=d_model,
            codebook_size=n_symbols,  # number of codebook entries
            embedding_dim=grapheme_embedding_dim,
            temp=latent_temp,
        )

        # load weights from pretrained tts into gumbel softmax
        init_embedding_weights(pretrained_tts.encoder.word_emb.weight.unsqueeze(0), self.quantiser.vars)

    def forward(self, inputs):
        # print(f"1: before embed {inputs.size()=}")
        inputs = self.embedding(inputs)
        # print(f"2: after embed {inputs.size()=}")
        if self.batch_first:
            inputs = inputs.transpose(0,1)
        inputs = self.pos_encoder(inputs)
        if self.batch_first:
            inputs = inputs.transpose(0,1)
        # print(f"3: after pos encoder {inputs.size()=}")
        logits = self.encoder(inputs)
        # print(f"4: after ff transformer {logits.size()=}")
        # logits = self.linear(inputs)

        quantiser_outdict = self.quantiser(logits, produce_targets=True)
        g_embedding_indices = quantiser_outdict["targets"].squeeze(2)
        g_embeddings = quantiser_outdict["x"]

        return g_embeddings, g_embedding_indices

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def initialise_embedding_table(nn_embedding, weights):
    raise NotImplementedError
