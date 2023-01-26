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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import glob
import re
from collections import defaultdict, Counter
import warnings

from fastpitch import models as fastpitch_model
from fastpitch.common.text.text_processing import TextProcessor

from modules.model import EncoderRespeller
from modules.gumbel_vector_quantizer import GumbelVectorQuantizer
from modules.sdtw_cuda_loss import SoftDTW

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from torch_optimizer import Lamb
import time
from fastpitch.common.utils import mask_from_lens
from collections import OrderedDict

import wandb
from datetime import datetime

from fastpitch.common.text.text_processing import TextProcessor # for respellerdataset

from tqdm import tqdm
import os

def parse_args(parser):
    """Parse commandline arguments"""
    parser.add_argument('-o', '--chkpt-save-dir', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str, default='./',
                        help='Path to dataset')

    train = parser.add_argument_group('training setup')
    train.add_argument('--cuda', action='store_true',
                       help='Enable GPU training')
    train.add_argument('--num-cpus', type=int, default=1,
                       help='Num of cpus on node. Used to optimise number of dataloader workers during training.')
    train.add_argument('--batch-size', type=int, default=16,
                       help='Batchsize (this is divided by number of GPUs if running Data Distributed Parallel Training)')
    train.add_argument('--val-num-to-gen', type=int, default=32,
                      help='Number of samples to generate in validation (determines how many samples show up in wandb')
    train.add_argument('--seed', type=int, default=1337,
                       help='Seed for PyTorch random number generators')
    train.add_argument('--grad-accumulation', type=int, default=1,
                       help='Training steps to accumulate gradients for')
    train.add_argument('--epochs', type=int, default=100,  # required=True,
                       help='Number of total epochs to run')
    train.add_argument('--max-iters-per-epoch', type=int, default=None,
                       help='Number of total batches to iterate through each epoch (reduce this to small number to quickly test whole training loop)')
    train.add_argument('--epochs-per-checkpoint', type=int, default=10,
                       help='Number of epochs per checkpoint')
    train.add_argument('--checkpoint-path', type=str, default=None,
                       help='Checkpoint path to resume train')
    train.add_argument('--resume', action='store_true',
                       help='Resume train from the last available checkpoint')
    train.add_argument('--val-log-interval', type=int, default=5,
                       help='How often to generate melspecs/audio for respellings and log to wandb')
    train.add_argument('--speech-length-penalty-training', action='store_true',
                       help='Whether or not to encourage model to output similar length outputs\
                       as the ground truth. Idea from V2C: Visual Voice Cloning (Chen et al. 2021)')
    train.add_argument('--skip-before-train-loop-validation', action='store_true',
                       help='Skip running validation before model training begins (mostly for speeding up testing of actual training loop)')
    train.add_argument('--avg-loss-by-speech_lens', action='store_true',
                       help='Average the softdtw loss according to number of timesteps in predicted sequence')
    train.add_argument('--softdtw-temp', type=float, default=0.01,
                       help='How hard/soft to make min operation. Minimum is recovered by setting this to 0.')
    train.add_argument('--softdtw-bandwidth', type=int, default=120,
                       help='Bandwidth for pruning paths in alignment matrix when calculating SoftDTW')
    train.add_argument('--dist-func', type=str, default="l1",
                       help='What distance function to use in softdtw loss calculation')

    opt = parser.add_argument_group('optimization setup')
    opt.add_argument('--optimizer', type=str, default='lamb', choices=['adam', 'lamb'],
                     help='Optimization algorithm')
    opt.add_argument('-lr', '--learning-rate', default=0.1, type=float,
                     help='Learning rate')
    opt.add_argument('--weight-decay', default=1e-6, type=float,
                     help='Weight decay')
    opt.add_argument('--grad-clip-thresh', default=1000.0, type=float,
                     help='Clip threshold for gradients')
    opt.add_argument('--warmup-steps', type=int, default=1000,
                     help='Number of steps for lr warmup')

    arch = parser.add_argument_group('architecture')
    arch.add_argument('--dropout-inputs', type=float, default=0.0,
                      help='Dropout prob to apply to sum of word embeddings '
                           'and positional encodings')
    arch.add_argument('--dropout-layers', type=float, default=0.1,
                      help='Dropout prob to apply to each layer of Tranformer')
    arch.add_argument('--d-model', type=int, default=128,
                      help='Hidden dimension of tranformer')
    arch.add_argument('--d-feedforward', type=int, default=512,
                      help='Hidden dimension of tranformer')
    arch.add_argument('--num-layers', type=int, default=4,
                      help='Number of layers for transformer')
    arch.add_argument('--nheads', type=int, default=4,
                      help='Hidden dimension of tranformer')
    arch.add_argument('--embedding-dim', type=int, default=384, # 384 is default value for fastpitch embedding table
                      help='Hidden dimension of grapheme embedding table')
    arch.add_argument('--pretrained-embedding-table', action='store_true',
                      help='Whether or not to initialise embedding table from fastpitchs')
    arch.add_argument('--freeze-embedding-table', action='store_true',
                      help='Whether or not to allow grapheme embedding input table for EncoderRespeller to be updated.')
    arch.add_argument('--gumbel-temp', nargs=3, type=float, default=(2, 0.5, 0.999995),
                      help='Temperature annealling parameters for Gumbel-Softmax (start, end, decay)')
    arch.add_argument('--no-src-key-padding-mask', dest='src_key_padding_mask', action='store_false',
                      help='Whether or not to provide padding attention mask to Transformer Encoder layers')
    arch.add_argument('--respelling-len-modifier', type=int, default=0, # 384 is default value for fastpitch embedding table
                      help='How many letters to remove from or add to original spelling.')
    arch.add_argument('--use-respelling-len-embeddings', action='store_true', # 384 is default value for fastpitch embedding table
                      help='Whether or not to incorporate to respeller input additional embeddings that indicate how long'
                           'the desired respelling should be.')
    arch.add_argument('--concat-pos-encoding', action='store_true',
                      help='Whether or not to concatenate pos encodings to inputs or sum')
    arch.add_argument('--pos-encoding-dim', type=int, default=128,
                      help='Dim of positional encoding module')
    arch.add_argument('--dont-only-predict-alpha', dest='only_predict_alpha', action='store_false',
                      help='Allow gumbel softmax to predict whitespace, padding, and other punctuation symbols')

    pretrained_tts = parser.add_argument_group('pretrained tts model')
    # pretrained_tts.add_argument('--fastpitch-with-mas', type=bool, default=True,
    #                   help='Whether or not fastpitch was trained with Monotonic Alignment Search (MAS)')
    pretrained_tts.add_argument('--fastpitch-chkpt', type=str, required=True,
                                help='Path to pretrained fastpitch checkpoint')
    pretrained_tts.add_argument('--input-type', type=str, default='char',
                                choices=['char', 'phone', 'pf', 'unit'],
                                help='Input symbols used, either char (text), phone, pf '
                                     '(phonological feature vectors) or unit (quantized acoustic '
                                     'representation IDs)')
    pretrained_tts.add_argument('--symbol-set', type=str, default='english_basic_lowercase',
                                help='Define symbol set for input sequences. For quantized '
                                     'unit inputs, pass the size of the vocabulary.')
    pretrained_tts.add_argument('--n-speakers', type=int, default=1,
                                help='Condition on speaker, value > 1 enables trainable '
                                     'speaker embeddings.')
    # pretrained_tts.add_argument('--use-sepconv', type=bool, default=True,
    #                   help='Use depthwise separable convolutions')

    audio = parser.add_argument_group('log generated audio')
    audio.add_argument('--hifigan', type=str,
                       default='/home/s1785140/pretrained_models/hifigan/ljspeech/LJ_V1/generator_v1',
                       help='Path to HiFi-GAN audio checkpoint')
    audio.add_argument('--hifigan-config', type=str,
                       default='/home/s1785140/pretrained_models/hifigan/ljspeech/LJ_V1/config.json',
                       help='Path to HiFi-GAN audio config file')
    audio.add_argument('--sampling-rate', type=int, default=22050,
                       help='Sampling rate for output audio')
    audio.add_argument('--hop-length', type=int, default=256,
                       help='STFT hop length for estimating audio length from mel size')

    data = parser.add_argument_group('dataset parameters')
    data.add_argument('--wordaligned-speechreps', type=str,
                      default='/home/s1785140/data/ljspeech_fastpitch/wordaligned_mels',
                      help='Path to directory of wordaligned speechreps/mels. Inside are folders\
                       each named as a wordtype and containing tensors of word aligned speechreps for each example')
    data.add_argument('--train-wordlist', type=str,
                      default='/home/s1785140/data/ljspeech_fastpitch/respeller_train_words.json',
                      help='Path to words that are used to train respeller')
    data.add_argument('--val-wordlist', type=str,
                      default='/home/s1785140/data/ljspeech_fastpitch/respeller_dev_words.json',
                      help='Path to words that are used to report validation metrics for respeller')
    data.add_argument('--max-examples-per-wordtype', type=int, default=1,
                      help='Path to words that are used to report validation metrics for respeller')
    data.add_argument('--text-cleaners', type=str, nargs='+',
                      default=(),
                      help='What text cleaners to apply to text in order to preproces it before'
                           'its fed to respeller.')

    cond = parser.add_argument_group('conditioning on additional attributes')
    dist = parser.add_argument_group('distributed training setup')

    wandb_logging = parser.add_argument_group('wandb logging')
    data.add_argument('--wandb-project-name', type=str,
                      required=True,
                      help="The name of the wandb project to add this experiment's logs to")
    wandb_logging.add_argument('--keys-to-add-to-exp-name', type=str, nargs='+',
                      default=(),
                      help='Command line arguments that we add their info to the wandb experiment name')

    return parser


def load_checkpoint(args, model, filepath):
    if args.local_rank == 0:
        print(f'Loading model and optimizer state from {filepath}')
    checkpoint = torch.load(filepath, map_location='cpu')
    sd = {k.replace('module.', ''): v
          for k, v in checkpoint['state_dict'].items()}
    getattr(model, 'module', model).load_state_dict(sd)
    return model


def load_respeller_checkpoint(args, model, filepath, optimizer, epoch, total_iter):
    if args.local_rank == 0:
        print(f'Loading model and optimizer state from {filepath}')
    checkpoint = torch.load(filepath, map_location='cpu')
    epoch[0] = checkpoint['epoch'] + 1
    total_iter[0] = checkpoint['iteration']
    sd = {k.replace('module.', ''): v
          for k, v in checkpoint['state_dict'].items()}
    getattr(model, 'module', model).load_state_dict(sd)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model


def last_checkpoint(output):
    saved = sorted(
        glob.glob(f'{output}/respeller_checkpoint_*.pt'),
        key=lambda f: int(re.search('_(\d+).pt', f).group(1)))

    def corrupted(fpath):
        try:
            torch.load(fpath, map_location='cpu')
            return False
        except:
            warnings.warn(f'Cannot load {fpath}')
            return True

    if len(saved) >= 1 and not corrupted(saved[-1]):
        return saved[-1]
    elif len(saved) >= 2:
        return saved[-2]
    else:
        return None

def maybe_save_checkpoint(args, model, optimizer, epoch,
                          total_iter, config):
    if args.local_rank != 0:
        return

    intermediate = (args.epochs_per_checkpoint > 0
                    and epoch % args.epochs_per_checkpoint == 0)

    if not intermediate and epoch < args.epochs:
        return

    os.makedirs(args.chkpt_save_dir, exist_ok=True)
    fpath = os.path.join(args.chkpt_save_dir, f"respeller_checkpoint_{epoch}.pt")
    print(f"Saving model and optimizer state at epoch {epoch} to {fpath}")
    checkpoint = {'epoch': epoch,
                  'iteration': total_iter,
                  'config': config,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, fpath)

def init_embedding_weights(source_tensor, target_tensor):
    """copy weights inplace from source tensor to target tensor"""
    target_tensor.requires_grad = False
    target_tensor.copy_(source_tensor.clone().detach())
    target_tensor.requires_grad = True


def load_pretrained_fastpitch(args):
    # load chkpt
    device = torch.device('cuda' if args.cuda else 'cpu')
    model_config = fastpitch_model.get_model_config('FastPitch', args)
    fastpitch = fastpitch_model.get_model('FastPitch', model_config, device, forward_is_infer=True)
    load_checkpoint(args, fastpitch, args.fastpitch_chkpt)
    # get information about grapheme embedding table
    n_symbols = fastpitch.encoder.word_emb.weight.size(0)
    grapheme_embedding_dim = fastpitch.encoder.word_emb.weight.size(1)
    return fastpitch, n_symbols, grapheme_embedding_dim, model_config

def init_wandb(args, add_datestr_to_exp_name=False):
    # wandb.login() # causes problems when running job on slurm?
    wandb_config = vars(args) # store important information into WANDB config for easier tracking of experiments
    exp_name = args.chkpt_save_dir.split('/')[-1]

    # add more info to experiment name for easieser tracking
    hparam_info_str = ""
    for key in args.keys_to_add_to_exp_name:
        value = wandb_config[key]
        hparam_info_str += f"-{key}={value}"
    exp_name += hparam_info_str

    if add_datestr_to_exp_name:
        dt_string = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
        exp_name += dt_string

    wandb.init(
        project=args.wandb_project_name,
        name=exp_name,
        config=wandb_config,
    )

def load_vocoder(args, device):
    """Load HiFi-GAN vocoder from checkpoint"""
    checkpoint_data = torch.load(args.hifigan)
    vocoder_config = fastpitch_model.get_model_config('HiFi-GAN', args)
    vocoder = fastpitch_model.get_model('HiFi-GAN', vocoder_config, device)
    vocoder.load_state_dict(checkpoint_data['generator'])
    vocoder.remove_weight_norm()
    vocoder.eval()
    return vocoder

def adjust_learning_rate(total_iter, opt, learning_rate, warmup_iters=None):
    if warmup_iters == 0:
        scale = 1.0
    elif total_iter > warmup_iters:
        scale = 1. / (total_iter ** 0.5)
    else:
        scale = total_iter / (warmup_iters ** 1.5)

    for param_group in opt.param_groups:
        param_group['lr'] = learning_rate * scale

def calc_sl_penalty(pred_lens, gt_lens):
    '''speech length mismatch penalty similar to MCD-DTW-SL
    encourages two sequences to be of same length
    M and N are length of each sequence
    coef = Max(M,N) / Min(M,N)'''
    # stack so we can calculate max along batch dimension
    stacked = torch.stack([pred_lens, gt_lens])
    maxs, _ = torch.max(stacked, dim=0)
    mins, _ = torch.min(stacked, dim=0)
    coefs = maxs/mins
    return coefs

class RespellerDataset(torch.utils.data.Dataset):
    """
        1) loads word + word-aligned mel spec for all words in a wordlist
        2) converts text to sequences of one-hot vectors (corresponding to grapheme indices in fastpitch)
    """

    def __init__(
            self,
            wordaligned_speechreps_dir,  # path to directory that contains folders of word aligned speech reps
            wordlist,  # txt file for the words to include speech reps from
            max_examples_per_wordtype=None,
            text_cleaners=[],
            symbol_set="english_basic_lowercase_no_arpabet",
            add_spaces=True,
            eos_symbol="$",
            **kwargs,
    ):
        # load wordlist as a python list
        if type(wordlist) == str:
            if wordlist.endswith('.json'):
                with open(wordlist) as f:
                    wordlist = json.load(f)
            else:
                with open(wordlist) as f:
                    wordlist = f.read().splitlines()
        elif type(wordlist) == list:
            pass  # dont need to do anything, already in expected form
        elif type(wordlist) == set:
            wordlist = list(wordlist)

        wordlist = sorted(wordlist)

        # create list of all word tokens and their word aligned speech reps
        self.word_freq = Counter()
        self.token_and_melfilepaths = []
        print("Initialising respeller dataset")
        for word in tqdm(wordlist):
            # find all word aligned mels for the word
            word_dir = os.path.join(wordaligned_speechreps_dir, word)
            mel_files = os.listdir(word_dir)
            if max_examples_per_wordtype:
                mel_files = mel_files[:max_examples_per_wordtype]
            for mel_file in mel_files:
                mel_file_path = os.path.join(word_dir, mel_file)
                self.token_and_melfilepaths.append((word, mel_file_path))
                self.word_freq[word] += 1

        self.tp = TextProcessor(symbol_set, text_cleaners, add_spaces=add_spaces, eos_symbol=eos_symbol)

    def get_mel(self, filename):
        return torch.load(filename)

    def encode_text(self, text):
        """encode raw text into indices defined by grapheme embedding table of the TTS model"""
        return torch.IntTensor(self.tp.encode_text(text))

    def decode_text(self, encoded):
        if encoded.dim() == 1:
            decodings = [self.tp.id_to_symbol[id] for id in encoded.tolist()]
        else:
            decodings = []
            for batch_idx in range(encoded.size(0)):
                decodings.append(''.join(self.tp.id_to_symbol[idx] for idx in encoded[batch_idx].tolist()))
        return decodings

    @staticmethod
    def get_mel_len(melfilepath):
        return int(melfilepath.split('seqlen')[1].split('.pt')[0])

    def __getitem__(self, index):
        word, mel_filepath = self.token_and_melfilepaths[index]
        encoded_word = self.encode_text(word)
        mel = self.get_mel(mel_filepath)

        return {
            'word': word,
            'encoded_word': encoded_word,
            'mel_filepath': mel_filepath,
            'mel': mel,
        }

    def __len__(self):
        return len(self.token_and_melfilepaths)

class Collate:
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, text_len_modifier=0):
        self.text_len_modifier = text_len_modifier

    def __call__(self, batch):
        """Collate's training batch from encoded word token and its
        corresponding word-aligned mel spectrogram

        batch: [encoded_token, wordaligned_mel]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x['encoded_word']) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        words = []
        mel_filepaths = []
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        text_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            words.append(batch[ids_sorted_decreasing[i]]['word'])
            mel_filepaths.append(batch[ids_sorted_decreasing[i]]['mel_filepath'])
            text = batch[ids_sorted_decreasing[i]]['encoded_word']
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

        # Right zero-pad mel-spec
        num_mels = batch[0]['mel'].size(1)
        max_target_len = max([x['mel'].size(0) for x in batch])

        mel_padded = torch.FloatTensor(len(batch), max_target_len, num_mels)
        mel_padded.zero_()
        mel_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]]['mel']
            mel_padded[i, :mel.size(0), :] = mel
            mel_lengths[i] = mel.size(0)

        def get_desired_text_lens(lens):
            if self.text_len_modifier < 0:
                min_len = torch.tensor(1)
                return torch.max(lens + self.text_len_modifier, min_len)
            elif self.text_len_modifier > 0:
                max_len = torch.max(lens)
                return torch.min(lens + self.text_len_modifier, max_len)
            else:
                return lens

        return {
            'words': words,
            'text_padded': text_padded,
            'text_lengths': text_lengths,
            'desired_text_lengths': get_desired_text_lens(text_lengths),
            'mel_padded': mel_padded,
            'mel_lengths': mel_lengths,
            'mel_filepaths': mel_filepaths
        }
        # input_lengths, mel_padded, output_lengths,
        # len_x, dur_padded, dur_lens, pitch_padded, speaker)


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def batch_to_gpu(collated_batch):
    """put elements that are used throughout training onto gpu"""
    words = collated_batch['words']
    text_padded = collated_batch['text_padded']
    text_lengths = collated_batch['text_lengths']
    mel_padded = collated_batch['mel_padded']
    mel_lengths = collated_batch['mel_lengths']
    desired_text_lengths = collated_batch['desired_text_lengths']

    # no need to put words on gpu, its only used during eval loop
    text_padded = to_gpu(text_padded).long()
    text_lengths = to_gpu(text_lengths).long()
    mel_padded = to_gpu(mel_padded).float()
    mel_lengths = to_gpu(mel_lengths).long()
    desired_text_lengths = to_gpu(desired_text_lengths).long()

    # x: inputs
    x = {
        'words': words,
        'text_padded': text_padded,
        'text_lengths': text_lengths,
        'desired_text_lengths': desired_text_lengths,
    }
    # y: targets
    y = {
        'mel_padded': mel_padded,
        'mel_lengths': mel_lengths,
    }

    return (x, y)

def mean_absolute_error(x, y):
    """for calculating softdtw using L1 loss
    Calculates the Euclidean distance between each element in x and y per timestep
    """
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    return torch.abs(x - y).sum(3)

def log_spectrogram(log_mel, figsize=(15, 5), image_name=""):
    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(log_mel, ax=ax, x_axis='frames', y_axis='linear')
    ax.set_title(image_name)
    fig.colorbar(img, ax=ax)
    return fig


def get_spectrograms_plots(y, fnames, n=4, label='Predicted spectrogram', mas=False, return_figs=False):
    """Plot spectrograms for n utterances in batch"""
    bs = len(fnames)
    n = min(n, bs)
    s = bs // n
    # fnames = fnames[::s]
    # print(f"inside get_spectrograms_plots(), {fnames=}")
    if label == 'Predicted spectrogram':
        # y: mel_padded, mel_lens
        mel_specs = y[0].transpose(1, 2).cpu().numpy()
        mel_lens = y[1].cpu().numpy() - 1
    elif label == 'Reference spectrogram':
        # y: mel_padded, mel_lens
        mel_specs = y[0].cpu().numpy()
        mel_lens = y[1].cpu().numpy()  # output_lengths

    image_names = []
    spectrograms = []
    for mel_spec, mel_len, fname in zip(mel_specs, mel_lens, fnames):
        mel_spec = mel_spec[:, :mel_len]
        utt_id = os.path.splitext(os.path.basename(fname))[0]
        image_name = f'val/{label}/{utt_id}'
        fig = log_spectrogram(mel_spec, image_name=image_name)
        image_names.append(image_name)

        if return_figs:
            spectrograms.append(fig)
        else:
            buf = BytesIO()
            fig.savefig(buf, format='png')
            img = Image.open(buf)
            plt.close(fig)
            spectrograms.append(img)

    return image_names, spectrograms


def generate_audio(y, fnames, vocoder=None, sampling_rate=22050, hop_length=256,
                   n=4, label='Predicted audio', mas=False):
    """Generate audio from spectrograms for n utterances in batch"""
    bs = len(fnames)
    n = min(n, bs)
    s = bs // n
    # fnames = fnames[::s]
    # print(f"inside generate_audio(), {fnames=}")
    with torch.no_grad():
        if label == 'Predicted audio':
            # y: mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred
            audios = vocoder(y[0].transpose(1, 2)).cpu().squeeze(1).numpy() # [bsz, dim, samples ]only squeeze away dim (equals 1 for waveform)
            mel_lens = y[1].cpu().numpy() - 1
        elif label == 'Copy synthesis':
            # y: mel_padded, dur_padded, dur_lens, pitch_padded
            audios = vocoder(y[0]).cpu().squeeze().numpy()
            if mas:
                mel_lens = y[2].cpu().numpy()  # output_lengths
            else:
                mel_lens = y[1].cpu().numpy().sum(axis=1) - 1
        elif label == 'Reference audio':
            audios = []
            for fname in fnames:
                wav = re.sub(r'mels/(.+)\.pt', r'wavs/\1.wav', fname)
                audio, _ = librosa.load(wav, sr=sampling_rate)
                audios.append(audio)
            if mas:
                mel_lens = y[2].cpu().numpy()  # output_lengths
            else:
                mel_lens = y[1].cpu().numpy().sum(axis=1) - 1
    audios_to_return = []
    for audio, mel_len, fname in zip(audios, mel_lens, fnames):
        audio = audio[:mel_len * hop_length]
        audio = audio / np.max(np.abs(audio))
        utt_id = os.path.splitext(os.path.basename(fname))[0]
        audios_to_return.append(audio)

    return audios_to_return

class WandbTable:
    def __init__(self):
        self.table = wandb.Table(columns=[
            "names",
            "orig spell",
            "orig spell spec",
            "orig spell wav",
            "vocoded gt spec",
            "vocoded gt wav",
            "respell",
            "respell spec",
            "respell wav",
            "sl penalty coef",
            "loss",
            "orig loss",
        ])

    def add_rows(
            self,
            names,
            vocoded_gt_specs,
            vocoded_gt_audios,
            orig_words,
            respellings,
            orig_pred_specs,  # either PIL images or matplotlib figures (but might have mem issues!)
            orig_pred_audios,
            pred_specs,  # either PIL images or matplotlib figures (but might have mem issues!)
            pred_audios,
            sl_penalty_coefs,
            losses,
            orig_losses,
            sampling_rate=22050,
    ):
        for (
                name,
                orig_word,
                orig_pred_spec_fig,
                orig_pred_audio,
                vocoded_gt_spec_fig,
                vocoded_gt_audio,
                respelling,
                pred_spec_fig,
                pred_audio,
                sl_penalty_coef,
                loss,
                orig_loss,
        ) in zip(
            names,
            orig_words,
            orig_pred_specs,
            orig_pred_audios,
            vocoded_gt_specs,
            vocoded_gt_audios,
            respellings,
            pred_specs,
            pred_audios,
            sl_penalty_coefs,
            losses,
            orig_losses,
        ):
            self.table.add_data(
                name,
                orig_word,
                wandb.Image(orig_pred_spec_fig, caption=name),
                wandb.Audio(orig_pred_audio, caption=name, sample_rate=sampling_rate),
                wandb.Image(vocoded_gt_spec_fig, caption=name),
                wandb.Audio(vocoded_gt_audio, caption=name, sample_rate=sampling_rate),
                respelling,
                wandb.Image(pred_spec_fig, caption=name),
                wandb.Audio(pred_audio, caption=name, sample_rate=sampling_rate),
                sl_penalty_coef,
                loss,
                orig_loss,
            )

            # close figures to save memory
            if type(orig_pred_spec_fig) == matplotlib.figure.Figure:
                plt.close(orig_pred_spec_fig)
            if type(vocoded_gt_spec_fig) == matplotlib.figure.Figure:
                plt.close(vocoded_gt_spec_fig)
            if type(pred_spec_fig) == matplotlib.figure.Figure:
                plt.close(pred_spec_fig)

    def log(self, is_trainset):
        if is_trainset:
            wandb.log({"train_table": self.table})
        else:
            wandb.log({"val_table": self.table})

def select(x, bsz, n):
    """select items in batch that will be visualised/converted to audio"""
    n = min(n, bsz)
    s = bsz // n
    return x[::s]


def validate(
        respeller_model,
        tts_model,
        vocoder,
        criterion,
        dataset,
        epoch,
        batch_size,
        num_to_gen,
        collate_fn,
        sampling_rate,
        hop_length,
        num_cpus,
        audio_interval=5,
        only_log_table=False,
        is_trainset=False,
        # start_epoch=0,
):
    """Handles all the validation scoring and printing
    GT (beginning of training):
    - log GT mel spec and vocoded audio for several validation set words

    Model outputs:
    - log predicted mel spec and vocoded audio from fastpitch
    - log respelled word from respeller
    """
    was_training = respeller_model.training
    respeller_model.eval()
    wandb_table = WandbTable()

    tik = time.perf_counter()
    with torch.no_grad():
        val_loader = DataLoader(dataset, num_workers=2*num_cpus, shuffle=False,
                                sampler=None,
                                batch_size=batch_size, pin_memory=False,
                                collate_fn=collate_fn)
        val_meta = defaultdict(float)
        val_losses = 0.0
        val_losses_with_sl_penalty = 0.0
        epoch_iter = 0
        sl_penalty_coefs = []

        num_generated = 0

        for i, batch in enumerate(val_loader):
            epoch_iter += 1

            # get loss over batch
            x, y = batch_to_gpu(batch)
            pred_mel, dec_lens, g_embedding_indices = forward_pass(respeller_model, tts_model, x)
            iter_loss = (criterion(pred_mel, y["mel_padded"]))
            val_losses += iter_loss.mean().item()

            coef = calc_sl_penalty(dec_lens, y['mel_lengths'])
            sl_penalty_coefs.append(coef.mean().item())
            val_losses_with_sl_penalty += (coef * iter_loss).mean().item()

            # log spectrograms and generated audio for first few utterances
            log_table = ((epoch-1) % audio_interval == 0 if epoch is not None else True)
            # print(f"DEBUG audio interval {audio_interval=} {epoch=} {start_epoch=}")
            # print(f"DEBUG audio interval {((epoch-1) % audio_interval == 0)=}")
            should_generate = num_generated < num_to_gen
            if log_table and should_generate:
                fnames = batch['mel_filepaths']
                bsz = len(fnames)

                num_to_generate_this_batch = min(bsz, num_to_gen - num_generated)

                # get original word and respellings for logging
                original_words = dataset.decode_text(x['text_padded'])
                respellings = dataset.decode_text(g_embedding_indices)

                # vocode original recorded speech
                gt_mel = y['mel_padded']
                gt_mel_lens = y['mel_lengths']
                _orig_token_names, gt_specs = get_spectrograms_plots(
                    (gt_mel.transpose(1, 2), gt_mel_lens), fnames,
                    n=num_to_generate_this_batch, label='Reference spectrogram', mas=False)
                vocoded_gt = generate_audio((gt_mel, gt_mel_lens), fnames, vocoder,
                                            sampling_rate, hop_length, n=num_to_generate_this_batch,
                                            label='Predicted audio', mas=True)

                # get melspec + generated audio for original spellings
                orig_pred_mel, orig_dec_lens, _dur_pred, _pitch_pred = tts_model(
                    inputs=x['text_padded'],
                    skip_embeddings=False,
                )
                # print(f'DEBUG VALIDATE ORIG {orig_pred_mel.size()=} {y["mel_padded"].size()}')
                orig_pred_mel = orig_pred_mel.transpose(1, 2)
                orig_iter_loss = (criterion(orig_pred_mel, y["mel_padded"]))
                _orig_token_names, orig_pred_specs = get_spectrograms_plots(
                    (orig_pred_mel, orig_dec_lens), fnames,
                    n=num_to_generate_this_batch, label='Predicted spectrogram', mas=True)
                orig_pred_audios = generate_audio(
                    (orig_pred_mel, orig_dec_lens), fnames,
                    vocoder, sampling_rate, hop_length,
                    n=num_to_generate_this_batch, label='Predicted audio', mas=True)

                # get melspec + generated audio for respellings
                token_names, pred_specs = get_spectrograms_plots(
                    (pred_mel, dec_lens), fnames,
                    n=num_to_generate_this_batch, label='Predicted spectrogram', mas=True)
                pred_audios = generate_audio(
                    (pred_mel, dec_lens), fnames, vocoder, sampling_rate, hop_length,
                    n=num_to_generate_this_batch, label='Predicted audio', mas=True)

                # log everything to wandb table
                token_names = [tok_name.split('/')[-1] for tok_name in token_names]

                wandb_table.add_rows(
                    names=token_names,
                    vocoded_gt_specs=gt_specs,
                    vocoded_gt_audios=vocoded_gt,
                    orig_words=original_words,
                    orig_pred_specs=orig_pred_specs,
                    orig_pred_audios=orig_pred_audios,
                    respellings=respellings,
                    pred_specs=pred_specs,
                    pred_audios=pred_audios,
                    sl_penalty_coefs=coef,
                    losses=iter_loss,
                    orig_losses=orig_iter_loss,
                    sampling_rate=sampling_rate,
                )

                num_generated += num_to_generate_this_batch

            if log_table and only_log_table and num_generated > num_to_gen:
                break  #  leave for loop after first iteration

        if not only_log_table:
            val_logs = {}
            val_logs['val/epoch_loss'] = val_losses / epoch_iter
            if val_losses_with_sl_penalty != 0.0:
                val_logs['val/epoch_loss_with_sl_penalty'] = val_losses_with_sl_penalty / epoch_iter
                val_logs['val/epoch_sl_penalty_coef'] = sum(sl_penalty_coefs) / len(sl_penalty_coefs)
            wandb.log(val_logs)

    if log_table:
        wandb_table.log(is_trainset=is_trainset)

    if was_training:
        respeller_model.train()

def get_src_key_padding_mask(bsz, max_len, lens, device):
    """return a Boolean mask for a list or tensor of sequence lengths
    True for values in tensor greater than sequence length

    bsz (int)
    max_len (int): max seq len of item in batch
    lens [bsz]: list or tensor of lengths
    """
    if type(lens) == list:
        lens = torch.tensor(lens, device=device)
    assert lens.dim() == 1

    lens = lens.unsqueeze(1)  # [bsz] -> [bsz, seq_len]
    m = torch.arange(max_len, device=device)
    m = m.expand(bsz, max_len)  # repeat along batch dimension
    m = (m < lens)
    return ~m  # tilde inverts a bool tensor

def forward_pass(respeller, tts, x):
    """x: inputs
    x = {
        'words': words,
        'text_padded': text_padded,
        'text_lengths': text_lengths,
    }"""
    text_lens = x['text_lengths']
    max_len = max(text_lens).item()
    bsz = len(text_lens)
    mask = get_src_key_padding_mask(bsz, max_len, text_lens, x['text_padded'].device)

    g_embeddings, g_embedding_indices = respeller(x['text_padded'], mask)

    # print(f'DEBUG desiredtextlen1 {g_embedding_indices.size()=} {g_embeddings.size()=} {g_embedding_indices=}')
    # print(f'DEBUG desiredtextlen2 {max_len=} {x["desired_text_lengths"].size()=} {x["desired_text_lengths"]}')
    # respellings = dataset.decode_text(g_embedding_indices)
    # print(f'DEBUG desiredtextlen2 {respellings=}')

    # use text lens to zero out/pad the output of the respeller so that repelling matches the length of the original spelling
    for i, desired_text_len in enumerate(x['desired_text_lengths']):
        g_embedding_indices[i, desired_text_len:] = 0
        g_embeddings[i, desired_text_len:, :] = 0.0

    # print(f'DEBUG desiredtextlen3 AFTER zero padding {g_embedding_indices=}')

    # quantiser_outdict = quantiser(logits, produce_targets=True)
    # g_embedding_indices = quantiser_outdict["targets"].squeeze(2)
    # g_embeddings = quantiser_outdict["x"]

    log_mel, dec_lens, _dur_pred, _pitch_pred = tts(
        inputs=g_embeddings,
        ids=g_embedding_indices,
        skip_embeddings=True,
    )

    # log_mel: [bsz, dim, seqlen]
    log_mel = log_mel.transpose(1, 2)
    # log_mel: [bsz, seqlen, dim]

    # return mask for masking acoustic loss
    # padding_idx = 0
    # mask = (g_embedding_indices != padding_idx).unsqueeze(2)
    # mask.size()
    # dec_mask = mask_from_lens(dec_lens).unsqueeze(2)

    return log_mel, dec_lens, g_embedding_indices


def byte_to_gigabyte(bytes):
    return bytes / 1000000000

def pretraining_prep(args, rank):
    args.local_rank = rank
    device = torch.device('cuda' if args.cuda else 'cpu')

    # load models
    tts, n_symbols, grapheme_embedding_dim, model_config = load_pretrained_fastpitch(args)
    respeller = EncoderRespeller(n_symbols=n_symbols,
                                 pretrained_tts=tts,
                                 d_embedding=args.embedding_dim,
                                 d_model=args.d_model,
                                 d_feedforward=args.d_feedforward,
                                 nhead=args.nheads,
                                 num_layers=args.num_layers,
                                 pretrained_embedding_table=args.pretrained_embedding_table,
                                 freeze_embedding_table=args.freeze_embedding_table,
                                 src_key_padding_mask=args.src_key_padding_mask,
                                 dropout_inputs=args.dropout_inputs,
                                 dropout_layers=args.dropout_layers,
                                 gumbel_temp=args.gumbel_temp,
                                 concat_pos_encoding=args.concat_pos_encoding,
                                 pos_encoding_dim=args.pos_encoding_dim,
                                 only_predict_alpha=args.only_predict_alpha,
                                 )
    if args.dist_func == 'l1':
        dist_func = mean_absolute_error
    elif args.dist_func == 'l2':
        dist_func = None  # softdtw package uses L2 as default
    else:
        dist_func = None  # softdtw package uses L2 as default

    criterion = SoftDTW(use_cuda=True, gamma=args.softdtw_temp, bandwidth=args.softdtw_bandwidth,
                        dist_func=dist_func)  # input should be size [bsz, seqlen, dim]

    tts.to(device)
    respeller.to(device)
    criterion.to(device)

    # load optimiser and assign to it the weights to be trained
    kw = dict(lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9,
              weight_decay=args.weight_decay)
    optimizer = Lamb(respeller.trainable_parameters(), **kw)

    # (optional) load checkpoint for respeller
    start_epoch = [1]
    start_iter = [0]
    assert args.checkpoint_path is None or args.resume is False, (
        "Specify a single checkpoint source")
    if args.checkpoint_path is not None:
        ch_fpath = args.checkpoint_path
    elif args.resume:
        ch_fpath = last_checkpoint(args.chkpt_save_dir)
    else:
        ch_fpath = None
    if ch_fpath is not None:
        load_respeller_checkpoint(args, respeller, ch_fpath, optimizer, start_epoch, start_iter)

    start_epoch = start_epoch[0]
    total_iter = start_iter[0]

    # create datasets, collate func, dataloader
    train_dataset = RespellerDataset(
        wordaligned_speechreps_dir=args.wordaligned_speechreps,
        wordlist=args.train_wordlist,
        max_examples_per_wordtype=args.max_examples_per_wordtype,
        add_spaces=args.add_spaces,
        symbol_set=args.symbol_set,
        text_cleaners=args.text_cleaners
    )
    val_dataset = RespellerDataset(
        wordaligned_speechreps_dir=args.wordaligned_speechreps,
        wordlist=args.val_wordlist,
        add_spaces=args.add_spaces,
        symbol_set=args.symbol_set,
        text_cleaners=args.text_cleaners
    )
    num_cpus = args.num_cpus  # TODO change to CLA? detect from wandb or some automatic way???
    collate_fn = Collate(text_len_modifier=args.respelling_len_modifier)
    train_loader = DataLoader(train_dataset, num_workers=2 * num_cpus, shuffle=True,
                              sampler=None, batch_size=args.batch_size,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, num_workers=2 * num_cpus, shuffle=False,
                            sampler=None, batch_size=args.batch_size,
                            pin_memory=False, collate_fn=collate_fn)

    # load pretrained hifigan
    vocoder = load_vocoder(args, device)

    # train loop
    respeller.train()
    # quantiser.train()
    tts.eval()

    print('Finished setting up models + dataloaders')

    return {
        "start_epoch": start_epoch,
        "total_iter": total_iter,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "collate_fn": collate_fn,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "respeller": respeller,
        "tts": tts,
        "vocoder": vocoder,
        "optimizer": optimizer,
        "criterion": criterion,
    }

def run_val(
    args,
    epoch,
    train_dataset,
    val_dataset,
    collate_fn,
    respeller,
    tts,
    vocoder,
    criterion,
    start_epoch,
):
    """wrap in fn so that we can call at:
    1. before training model
    2. at end of every X epochs"""
    # log audio and respellings for training set words
    validate(
        respeller_model=respeller,
        tts_model=tts,
        vocoder=vocoder,
        criterion=criterion,
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_to_gen=args.val_num_to_gen,
        collate_fn=collate_fn,
        epoch=epoch,
        sampling_rate=args.sampling_rate,
        hop_length=args.hop_length,
        num_cpus=args.num_cpus,
        audio_interval=args.val_log_interval,
        only_log_table=True,
        is_trainset=True,
    )

    # log audio and respellings for val set words
    validate(
        respeller_model=respeller,
        tts_model=tts,
        vocoder=vocoder,
        criterion=criterion,
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_to_gen=args.val_num_to_gen,
        collate_fn=collate_fn,
        epoch=epoch,
        sampling_rate=args.sampling_rate,
        hop_length=args.hop_length,
        num_cpus=args.num_cpus,
        audio_interval=args.val_log_interval,
    )

def train_loop(
    args,
    start_epoch,
    total_iter,
    train_dataset,
    val_dataset,
    collate_fn,
    train_loader,
    respeller,
    tts,
    vocoder,
    optimizer,
    criterion,
):
    print(f"\n *** Starting training! (from epoch {start_epoch}) ***")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Training loop: epoch {epoch}/{args.epochs}")

        # logging metrics
        epoch_start_time = time.perf_counter()
        iter_loss = 0
        epoch_loss = 0.0
        epoch_iter = 0
        num_iters = len(train_loader)
        mean_sl_penalty_coefs = []
        # epoch_mel_loss = 0.0
        # epoch_num_frames = 0
        # epoch_frames_per_sec = 0.0
        # iter_num_frames = 0
        # iter_meta = {}

        # iterate over all batches in epoch
        for batch in train_loader:
            if args.max_iters_per_epoch:
                if epoch_iter > args.max_iters_per_epoch:
                    print("quit training loop, FOR DEVELOPMENT!!!")
                    break
                print(f'DEBUG mode iter {epoch_iter} of {args.max_iters_per_epoch}')

            if epoch_iter == num_iters:  # useful for gradient accumulation
                break

            total_iter += 1
            epoch_iter += 1
            iter_start_time = time.perf_counter()

            adjust_learning_rate(total_iter, optimizer, args.learning_rate,
                                 args.warmup_steps)

            # adjust gumbelsoftmax temperature by updating the total number of iterations
            respeller.quantiser.set_num_updates(total_iter)

            optimizer.zero_grad()

            x, y = batch_to_gpu(batch)  # x: inputs, y: targets
            gt_mel = y["mel_padded"]

            # # y: targets
            # y = {
            #     'mel_padded': mel_padded,
            #     'mel_lengths': mel_lengths,
            # }

            # forward pass through models (respeller -> quantiser -> tts)
            pred_mel, dec_lens, _g_embedding_indices = forward_pass(respeller, tts, x)

            # TODO: DO WE NEED MASK IF WE USE SOFTDTW LOSS?
            # I THINK IT AUTOMATICALLY WILL ALIGN PADDED FRAMES WITH EACH OTHER???

            # print(f'inputs to loss {pred_mel.size()}, {gt_mel.size()}')

            # calculate loss
            loss = criterion(pred_mel, gt_mel)
            # print('raw loss from softdtw', loss.size())

            if args.avg_loss_by_speech_lens:
                loss = loss / dec_lens  # needed because softdtw code doesn't return avg loss by default TODO check this!
                # TODO also add gt lens? maybe shud normalise according to path len?
                # print('loss avg according to dec seqlens', loss.size())

            # penalise length mismatch
            coef = calc_sl_penalty(dec_lens, y['mel_lengths'])
            if args.speech_length_penalty_training:
                loss_no_sl_penalty = loss.clone().detach()
                loss = coef * loss

            loss = loss.mean()
            # print('loss avged across batch', loss.size())

            # backpropagation of loss
            loss.backward()

            # clip gradients and run optimizer
            torch.nn.utils.clip_grad_norm_(respeller.trainable_parameters(), args.grad_clip_thresh)
            optimizer.step()

            # log metrics to terminal and to wandb
            iter_loss = loss.item()
            iter_time = time.perf_counter() - iter_start_time
            epoch_loss += iter_loss

            # values to be logged by WANDB
            iter_logs = {}

            iter_logs["train/iter_loss"] = iter_loss
            iter_logs["train/iter_time"] = iter_time

            mean_sl_penalty_coef = coef.mean().item()
            mean_sl_penalty_coefs.append(mean_sl_penalty_coef)
            iter_logs["train/iter_sl_penalty_coef"] = mean_sl_penalty_coef

            if args.speech_length_penalty_training:
                iter_logs["train/iter_loss_no_sl_penalty"] = loss_no_sl_penalty.mean().item()

            iter_logs["train/iter_gumbel_temp"] = respeller.quantiser.curr_temp

            if True:
                # log memory usage
                t = torch.cuda.get_device_properties(0).total_memory
                r = torch.cuda.memory_reserved(0)
                a = torch.cuda.memory_allocated(0)
                f = r - a  # free inside reserved
                iter_logs['memory/total'] = byte_to_gigabyte(t)
                iter_logs['memory/reserved'] = byte_to_gigabyte(r)
                iter_logs['memory/allocated'] = byte_to_gigabyte(a)
                iter_logs['memory/free'] = byte_to_gigabyte(f)

            wandb.log(iter_logs)

            ### Finished Epoch!

        epoch_time = time.perf_counter() - epoch_start_time

        epoch_logs = {
            "train/epoch_num": epoch,
            "train/epoch_time": epoch_time,
            "train/epoch_loss": epoch_loss / epoch_iter,
        }
        if args.speech_length_penalty_training:
            epoch_logs["train/epoch_sl_penalty_coef"] = sum(mean_sl_penalty_coefs) / len(mean_sl_penalty_coefs)

        wandb.log(epoch_logs)

        run_val(
            args,
            epoch=epoch,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            respeller=respeller,
            tts=tts,
            vocoder=vocoder,
            criterion=criterion,
            start_epoch=1,
        )

        respeller_model_config = None # TODO fix this! replace with a model config similar to how fastpitch saves checkpoints in maybe_save_checkpoint!!!
        maybe_save_checkpoint(args, respeller, optimizer,
                              epoch, total_iter, respeller_model_config)

    print("\n *** Finished training! ***")

    # wandb.finish() # useful in jupyter notebooks

def train(rank, args):
    d =  pretraining_prep(args, rank)

    if not args.skip_before_train_loop_validation and d['start_epoch'] == 1 and d['total_iter'] == 0:
        print("Starting pre-training loop validation")
        run_val(
            args,
            epoch=d['start_epoch'],
            train_dataset=d['train_dataset'],
            val_dataset=d['val_dataset'],
            collate_fn=d['collate_fn'],
            respeller=d['respeller'],
            tts=d['tts'],
            vocoder=d['vocoder'],
            criterion=d['criterion'],
            start_epoch=1,
        )
        print("Finished pre-training loop validation")
    else:
        print("Skipping pre-training loop validation")

    train_loop(
        args,
        start_epoch=d["start_epoch"],
        total_iter=d["total_iter"],
        train_dataset=d['train_dataset'],
        val_dataset=d['val_dataset'],
        collate_fn=d['collate_fn'],
        train_loader=d["train_loader"],
        respeller=d["respeller"],
        tts=d["tts"],
        vocoder=d['vocoder'],
        optimizer=d["optimizer"],
        criterion=d["criterion"],
    )

def main():
    parser = argparse.ArgumentParser(description='PyTorch Respeller Training', allow_abbrev=False)
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

    init_wandb(args)

    # if args.distributed_run:
    #     mp.spawn(train, nprocs=args.num_gpus, args=(args,))
    # else:
    train(0, args)

if __name__ == '__main__':
    main()
