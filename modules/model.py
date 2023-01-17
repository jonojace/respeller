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
            d_embedding=384,
            d_model=512,
            nhead=4,
            num_layers=4,
            pretrained_embedding_table=True, # optional - initialise from pretrained grapheme embedding table from TTS
            freeze_embedding_table=True,
            batch_first=True,
            grapheme_embedding_dim=384,
            latent_temp=(2, 0.5, 0.999995),
    ):
        super().__init__()
        self.model_type = 'EncoderRespeller'
        self.batch_first = batch_first
        self.weights_to_freeze = ['quantiser.vars'] # weights that we do not update during training

        self.embedding = nn.Embedding(n_symbols, d_embedding) # dim of this should match that of fastpitch symbol embedding table if copying over weights
        if pretrained_embedding_table is not None:
            init_embedding_weights(pretrained_tts.encoder.word_emb.weight, self.embedding.weight)
            if freeze_embedding_table:
                self.weights_to_freeze.append('embedding.weight')

        self.proj = nn.Linear(d_embedding, d_model)

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

    def trainable_parameters(self):
        """return the model parameters that we wish to update for respeller training
        note that we ignore self.vars as we wish it to be initialised and frozen to the grapheme
        embedding table from the TTS model"""
        trainable_parameters = []
        for name, param in self.named_parameters():
            if name not in self.weights_to_freeze:
                trainable_parameters.append(param)
            else:
                # print("DEBUG - Frozen weights:", name, param.size())
                pass
        return trainable_parameters

    def forward(self, inputs):
        # print(f"1: before embed {inputs.size()=}")
        inputs = self.embedding(inputs)
        # print(f"2: after embed {inputs.size()=}")
        if self.batch_first:
            inputs = inputs.transpose(0,1)
        inputs = self.proj(inputs)
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
