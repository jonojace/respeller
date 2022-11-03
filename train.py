'''
Train respeller model

We backpropagate loss from pretrained TTS model to a Grapheme-to-Grapheme (G2G) respeller model to help it respell words
into a simpler form

Intermediated respellings are discrete character sequences
We can backpropagate through these using gumbel softmax and the straight through estimator
'''
import argparse
from modules.model import EncoderRespeller
from modules.gumbel_vector_quantizer import GumbelVectorQuantizer

def parse_args(parser):
    """Parse commandline arguments"""
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str, default='./',
                        help='Path to dataset')

    arch = parser.add_argument_group('architecture')
    train.add_argument('--d-model', type=int, required=True,
                       help='Hidden dimension of tranformer')
    train.add_argument('--latent-temp', type=tuple, default=(2, 0.5, 0.999995),
                       help='Temperature annealling parameters for Gumbel-Softmax (start, end, decay)')

def load_pretrained_fastpitch(chkpt_path):
    tts = convert_embedding_table_to_linear_layer(tts)
    tts.freeze_weights()
    vocab_size = len(tts.input_symbols)
    grapheme_embedding_dim = tts.embedding.size()[0]
    return fastpitch

def train(args):
    tts, vocab_size, grapheme_embedding_dim = load_pretrained_fastpitch(args.fastpitch_checkpoint)

    if 'modeltype' == 'autoregressive':
        raise NotImplementedError
    elif 'modeltype' == 'non_autoregressive':
        respeller = EncoderRespeller(in_vocab_size=vocab_size, d_model=args.d_model)

    quantiser = GumbelVectorQuantizer(
        dim=args.d_model, # input dimension to model, will get recast by linear layer to groups * num_vars
        num_vars=vocab_size,  # number of codebook entries
        temp=args.latent_temp,
        embedding_dim=grapheme_embedding_dim,
    )
    quantiser.init_embedding_weights(tts.embedding)

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

def main():
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Training',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    train(args)

if __name__ == '__main__':
    main()
