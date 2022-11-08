'''
Train respeller model

We backpropagate loss from pretrained TTS model to a Grapheme-to-Grapheme (G2G) respeller model to help it respell words
into a simpler form

Intermediated respellings are discrete character sequences
We can backpropagate through these using gumbel softmax and the straight through estimator
'''
import argparse
import numpy as np
import torch
import torch.distributed as dist

from fastpitch import models as fastpitch_model
from modules.model import EncoderRespeller
from modules.gumbel_vector_quantizer import GumbelVectorQuantizer

def parse_args(parser):
    """Parse commandline arguments"""
    # parser.add_argument('-o', '--output', type=str, required=True,
    #                     help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str, default='./',
                        help='Path to dataset')

    train_args = parser.add_argument_group('training setup')
    train_args.add_argument('--cuda', action='store_true',
                      help='Enable GPU training')
    train_args.add_argument('--batch-size', type=int, default=16,
                      help='Batchsize (this is divided by number of GPUs if running Data Distributed Parallel Training)')
    train_args.add_argument('--seed', type=int, default=1337,
                       help='Seed for PyTorch random number generators')
    train_args.add_argument('--grad-accumulation', type=int, default=1,
                       help='Training steps to accumulate gradients for')
    train_args.add_argument('--epochs', type=int, default=100, #required=True,
                       help='Number of total epochs to run')
    train_args.add_argument('--epochs-per-checkpoint', type=int, default=10,
                       help='Number of epochs per checkpoint')

    data_args = parser.add_argument_group('dataset parameters')
    cond_args = parser.add_argument_group('conditioning on additional attributes')
    audio_args = parser.add_argument_group('log generated audio')
    dist_args = parser.add_argument_group('distributed training setup')

    arch_args = parser.add_argument_group('architecture')
    arch_args.add_argument('--d-model', type=int, default=512,
                       help='Hidden dimension of tranformer')
    arch_args.add_argument('--latent-temp', type=tuple, default=(2, 0.5, 0.999995),
                       help='Temperature annealling parameters for Gumbel-Softmax (start, end, decay)')

    pretrained_tts_args = parser.add_argument_group('pretrained tts model')
    # pretrained_tts_args.add_argument('--fastpitch-with-mas', type=bool, default=True,
    #                   help='Whether or not fastpitch was trained with Monotonic Alignment Search (MAS)')
    pretrained_tts_args.add_argument('--fastpitch-chkpt', type=str, required=True,
                      help='Path to pretrained fastpitch checkpoint')
    pretrained_tts_args.add_argument('--input-type', type=str, default='char',
                      choices=['char', 'phone', 'pf', 'unit'],
                      help='Input symbols used, either char (text), phone, pf '
                      '(phonological feature vectors) or unit (quantized acoustic '
                      'representation IDs)')
    pretrained_tts_args.add_argument('--symbol-set', type=str, default='english_basic_lowercase',
                      help='Define symbol set for input sequences. For quantized '
                      'unit inputs, pass the size of the vocabulary.')
    pretrained_tts_args.add_argument('--n-speakers', type=int, default=1,
                      help='Condition on speaker, value > 1 enables trainable '
                      'speaker embeddings.')
    # pretrained_tts_args.add_argument('--use-sepconv', type=bool, default=True,
    #                   help='Use depthwise separable convolutions')

    return parser

def load_checkpoint(args, model, filepath):
    if args.local_rank == 0:
        print(f'Loading model and optimizer state from {filepath}')
    checkpoint = torch.load(filepath, map_location='cpu')
    sd = {k.replace('module.', ''): v
          for k, v in checkpoint['state_dict'].items()}
    getattr(model, 'module', model).load_state_dict(sd)
    return model

def freeze_weights(model):
    # NB wait... won't this stop backprop of gradients?
    # We just don't want to add the fastpitch/quantiser model weights to the optimiser...
    for param in model.parameters():
        param.requires_grad = False

def init_embedding_weights(source_tensor, target_tensor):
    """copy weights inplace from source tensor to target tensor"""
    target_tensor.requires_grad = False
    target_tensor.copy_(source_tensor.clone().detach())
    target_tensor.requires_grad = True

def load_pretrained_fastpitch(args):
    # load chkpt
    device = torch.device('cuda' if args.cuda else 'cpu')
    model_config = fastpitch_model.get_model_config('FastPitch', args)
    fastpitch = fastpitch_model.get_model('FastPitch', model_config, device, forward_mas=args.use_mas)
    load_checkpoint(args, fastpitch, args.fastpitch_chkpt)
    # get information about grapheme embedding table
    n_symbols = fastpitch.encoder.word_emb.weight.size(0)
    grapheme_embedding_dim = fastpitch.encoder.word_emb.weight.size(1)
    return fastpitch, n_symbols, grapheme_embedding_dim

def train(rank, args):
    args.local_rank = rank
    tts, n_symbols, grapheme_embedding_dim = load_pretrained_fastpitch(args)

    respeller = EncoderRespeller(n_symbols=n_symbols, d_model=args.d_model)

    quantiser = GumbelVectorQuantizer(
        in_dim=args.d_model,
        codebook_size=n_symbols,  # number of codebook entries
        embedding_dim=grapheme_embedding_dim,
        temp=args.latent_temp,
    )

    # copy grapheme embeddings from fastpitch to gumbelsoftmax codebook
    init_embedding_weights(tts.encoder.word_emb.weight.unsqueeze(0), quantiser.vars)

    # if init_respeller_grapheme_embeddings:
    #     init_embedding_weights(tts.encoder.word_emb.weight.unsqueeze(0), respeller.embed)

    # skip_emb(fastpitch)

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
        respelling = quantiser(logits, produce_targets=False)["x"]
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
    args, _unk_args = parser.parse_known_args()

    parser = fastpitch_model.parse_model_args('FastPitch', parser)
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    if args.cuda:
        args.num_gpus = torch.cuda.device_count()
        args.distributed_run = args.num_gpus > 1
        args.batch_size = int(args.batch_size / args.num_gpus)
    else:
        args.distributed_run = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.distributed_run:
        mp.spawn(train, nprocs=args.num_gpus, args=(args,))
    else:
        train(0, args)

if __name__ == '__main__':
    main()
