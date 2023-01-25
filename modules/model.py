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
            d_feedforward=512,
            nhead=4,
            num_layers=4,
            pretrained_embedding_table=True, # optional - initialise from pretrained grapheme embedding table from TTS
            freeze_embedding_table=True,
            grapheme_embedding_dim=384,
            gumbel_temp=(2, 0.5, 0.999995),
            src_key_padding_mask=True,
            dropout_inputs=0.0,
            dropout_layers=0.1,
            concat_pos_encoding=False,
            pos_encoding_dim=384,
    ):
        super().__init__()
        self.model_type = 'EncoderRespeller'
        self.weights_to_freeze = ['quantiser.vars'] # weights that we do not update during training
        self.src_key_padding_mask = src_key_padding_mask
        self.concat_pos_encoding = concat_pos_encoding

        self.embedding = nn.Embedding(n_symbols,
                                      d_embedding, # dim of this should match that of fastpitch symbol embedding table if copying over weights
                                      padding_idx=0) # should match pad idx of text processor used to generate inputs to respeller

        if pretrained_embedding_table:
            init_embedding_weights(pretrained_tts.encoder.word_emb.weight, self.embedding.weight)
            if freeze_embedding_table:
                self.weights_to_freeze.append('embedding.weight')



        # if concat_pos_encoding:
        #     self.pos_encoder = PositionalEncoding(pos_encoding_dim, concat_pos_encoding=concat_pos_encoding, dropout=dropout_inputs)
        #     proj2_indim += pos_encoding_dim
        # else:
        #     self.pos_encoder = PositionalEncoding(d_model, concat_pos_encoding=concat_pos_encoding, dropout=dropout_inputs)


        self.proj = nn.Linear(d_embedding, d_model)

        self.pos_emb = PositionalEmbedding(d_model)

        proj2_indim = d_model
        if concat_pos_encoding:
            proj2_indim += d_model
        self.proj2 = nn.Linear(proj2_indim, d_model)

        self.drop = nn.Dropout(dropout_inputs)

        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            d_feedforward,
            dropout=dropout_layers,
            batch_first=True,
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
            temp=gumbel_temp,
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

    def forward(
            self, inputs,
            src_key_padding_mask # True indicates positions to be padded
    ):
        # print('debug forward() before embed ', f"{inputs.size()}")

        enc_inputs = self.embedding(inputs) # -> [bsz, max_len, d_embed]

        # print('debug forward() after embed', f"{inputs.size()}")

        # inputs = self.proj1(inputs)

        # print('debug forward() after proj', f"{inputs.size()}")

        # inputs = inputs.transpose(0,1)
        # inputs = self.pos_encoder(inputs)
        # inputs = inputs.transpose(0,1)

        enc_inputs = self.proj(enc_inputs) # d_embedding -> d_model

        # incorporate positional information
        # NB we do this after the projection down to the dim of the model as this projection
        # should be learnt in tandem with the positional information
        # normally embeddings are learnt with the PE but in our usecase we sometimes freeze the embedding table
        # so we need to allow flexibility in the projection
        inverted_mask = ~src_key_padding_mask
        pos_seq = torch.arange(enc_inputs.size(1), device=enc_inputs.device).to(enc_inputs.dtype)
        featdim = enc_inputs.size(2)
        # print(f"debug pos emb {self.pos_emb(pos_seq, bsz=enc_inputs.size(0)).size()=}")
        # print(f"debug pos emb {inverted_mask.unsqueeze(2).expand(-1,-1,featdim).size()=}")
        pos_emb = self.pos_emb(pos_seq, bsz=enc_inputs.size(0)) * inverted_mask.unsqueeze(2).expand(-1,-1,featdim)
        if self.concat_pos_encoding:
            enc_inputs = torch.cat([
                enc_inputs,
                pos_emb,
            ], dim=2)
            enc_inputs = self.proj2(enc_inputs) # 2*d_model -> d_model
        else:
            enc_inputs = enc_inputs + pos_emb

        enc_inputs = self.drop(enc_inputs)

        if not self.src_key_padding_mask:
            src_key_padding_mask = None

        logits = self.encoder(enc_inputs, src_key_padding_mask=src_key_padding_mask)

        quantiser_outdict = self.quantiser(logits, produce_targets=True)
        g_embedding_indices = quantiser_outdict["targets"].squeeze(2)
        g_embeddings = quantiser_outdict["x"]

        return g_embeddings, g_embedding_indices

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.matmul(torch.unsqueeze(pos_seq, -1),
                                    torch.unsqueeze(self.inv_freq, 0))
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]

# class PositionalEncoding(nn.Module):
#     r"""Inject some information about the relative or absolute position of the tokens in the sequence.
#         The positional encodings have the same dimension as the embeddings, so that the two can be summed.
#         Here, we use sine and cosine functions of different frequencies.
#     .. math:
#         \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
#         \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
#         \text{where pos is the word position and i is the embed idx)
#     Args:
#         d_model: the embed dim (required).
#         dropout: the dropout value (default=0.1).
#         max_len: the max. length of the incoming sequence (default=5000).
#     Examples:
#         >>> pos_encoder = PositionalEncoding(d_model)
#     """
#
#     def __init__(self, d_model, concat_pos_encoding=False, dropout=0.0, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.concat_pos_encoding = concat_pos_encoding
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         r"""Inputs of forward function
#         Args:
#             x: the sequence fed to the positional encoder model (required).
#         Shape:
#             x: [sequence length, batch size, embed dim]
#             output: [sequence length, batch size, embed dim]
#         Examples:
#             >>> output = pos_encoder(x)
#         """
#         # print('debug pos enc', f"{x.size()=}, {self.pe[:x.size(0), :].size()=}")
#         if self.concat_pos_encoding:
#             bsz = x.size(1)
#             x = torch.cat([
#                 x,
#                 self.pe[:x.size(0), :].expand(x.size(0), bsz, x.size(2))
#             ], dim=2)
#         else:
#             # sum
#             x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)
