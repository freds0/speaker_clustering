import argparse
from os.path import join, exists, basename, splitext
from os import makedirs
from tqdm import tqdm
from glob import glob
import numpy as np
import torch
import torchaudio
from speechbrain.pretrained.interfaces import Pretrained
from speechbrain.pretrained import EncoderClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_filenames_embeddings(input_dir):
    embeddings = []
    filenames = []
    for filepath in glob(join(input_dir, "*.pt")):
        emb = torch.load(filepath)
        embeddings.append(emb.numpy())
        filenames.append(splitext(basename(filepath))[0])

    return np.array(filenames).squeeze(), np.array(embeddings).squeeze()


class Encoder(Pretrained):
    '''
    Source: https://huggingface.co/yangwang825/ecapa-tdnn-vox2
    '''

    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model"
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mods.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings,
                torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings


def extract_embeddings(filelist, output_dir):

    classifier = Encoder.from_hparams(
       source="yangwang825/ecapa-tdnn-vox2"
    )
    for filepath in tqdm(filelist):
        # Load audio file
        if not exists(filepath):
            print("file {} doesnt exist!".format(filepath))
            continue
        filename = basename(filepath)
        signal, fs = torchaudio.load(filepath)
        if fs != 16000:
            print(filepath)
            fn_resample = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000, resampling_method='sinc_interp_hann')
            signal = fn_resample(signal)            
        embedding = classifier.encode_batch(signal)
        # Saving embedding
        output_filename = filename.split(".")[0] + ".pt"
        output_filepath = join(output_dir, output_filename)
        makedirs(output_dir, exist_ok=True)
        torch.save(embedding, output_filepath)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', help='Wavs folder')
    parser.add_argument('-o', '--output_dir', default='output_embeddings', help='Name of csv file')
    args = parser.parse_args()

    filelist = glob(args.input_dir + '/*.wav')

    makedirs(args.output_dir, exist_ok=True)
    extract_embeddings(filelist, args.output_dir, args.model_name)


if __name__ == "__main__":
    main()
