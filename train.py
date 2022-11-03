'''
Train respeller model

We backpropagate loss from pretrained TTS model to a Grapheme-to-Grapheme (G2G) respeller model to help it respell words
into a simpler form

Intermediated respellings are discrete character sequences
We can backpropagate through these using gumbel softmax and the straight through estimator
'''

from modules.model import Respeller
from modules.gumbel_vector_quantizer import GumbelVectorQuantizer

def train():
    tts = load_fastpitch()
    tts = convert_embedding_table_to_linear_layer(tts)
    tts.freeze_weights()

    if 'modeltype' == 'autoregressive':
        respeller = EncoderDecoderRespeller(out_dim=len(tts.input_symbols))
    elif 'modeltype' == 'non_autoregressive':
        respeller = EncoderRespeller(out_dim=len(tts.input_symbols))

    quantiser = GumbelVectorQuantizer(
        dim=self.decoder.outdim,
        num_vars=cfg.latent_vars,  # 320 - number of latent variables V in each group of the codebook
        temp=cfg.latent_temp, # (2, 0.5, 0.999995) - temperature for latent variable sampling. can be tuple of 3 values (start, end, decay)
        groups=cfg.latent_groups,  # 2 - number of groups G of latent variables in the codebook
        combine_groups=False,
        vq_dim=vq_dim,
        time_first=True,
        weight_proj_depth=cfg.quantizer_depth,
        weight_proj_factor=cfg.quantizer_factor,
    )

    acoustic_loss_fn = Softdtw()

    # if args.respeller_loss:
    #     respelling_loss_fn = CrossEntropy()
    # else:

    for batch in batches:
        ###############################################################################################################
        # text, ssl_reps, e2e_asr_predictions, gt_log_mel = batch
        text, gt_log_mel = batch

        ###############################################################################################################
        # create inputs
        inputs = text
        # if args.use_acoustic_input:
        #     inputs = inputs.concat(ssl_reps)

        ###############################################################################################################
        # forward pass
        logits = respeller(inputs)
        respelling = quantiser(logits)
        log_mel = tts(respelling)

        ###############################################################################################################
        # calculate losses
        # respelling_loss = respelling_loss_fn(respelling, e2e_asr_predictions)
        acoustic_loss = acoustic_loss_fn(log_mel, gt_log_mel)

        ###############################################################################################################
        # backward pass
        loss = acoustic_loss
        loss.backward()

        ###############################################################################################################
        # log tensorboard metrics

        ###############################################################################################################
        # validation set evaluation
