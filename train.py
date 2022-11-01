'''
Train respeller model

We backpropagate loss from pretrained TTS model to a Grapheme-to-Grapheme (G2G) respeller model to help it respell words
into a simpler form

Intermediated respellings are discrete character sequences
We can backpropagate through these using gumbel softmax and the straight through estimator
'''

from modules.model import Respeller

def train():
    # models
    tts = Fastpitch()
    tts.freeze weights()
    respeller = Respeller(out_dim=len(tts.input_symbols), audio=use_audio, quantisation='gumbel_softmax')

    # forward pass
    respelling = respeller(text, audio=ssl_reps)
    log_mel = tts(respelling)
    loss = l2(log_mel, gt_log_mel)

    # backward pass
    loss.backward()

    # log tensorboard metrics

    # validation set evaluation
