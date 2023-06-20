import argparse
import torch
import nemo.collections.asr as nemo_asr
from os.path import join, exists, basename, splitext
from os import makedirs
from tqdm import tqdm
from glob import glob
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_name="speakernet"):
    speaker_model = None
    if (model_name == "speakernet"):
        speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name="speakerverification_speakernet"
        )
    elif (model_name == "titanet"):
        speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name="titanet_large"
        )
    return speaker_model


def get_filenames_embeddings(input_dir):
    embeddings = []
    filenames = []
    for filepath in glob(join(input_dir, "*.pt")):
        emb = torch.load(filepath)
        embeddings.append(emb.numpy())
        filenames.append(splitext(basename(filepath))[0])
        
    return np.array(filenames).squeeze(), np.array(embeddings).squeeze()


def extract_embeddings(filelist, output_dir, model_name='speakernet'):
    model = load_model(model_name)
    for filepath in tqdm(filelist):
        # Load audio file
        if not exists(filepath):
            print("file {} doesnt exist!".format(filepath))
            continue
        filename = basename(filepath)
        file_embedding = model.get_embedding(filepath).cpu().detach().numpy()
        embedding = torch.tensor(file_embedding)
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
    parser.add_argument('-m', '--model_name', help='Available Models: speakernet and titanet.')
    args = parser.parse_args()

    output_dir = join(args.base_dir, args.output_dir)
    input_dir = join(args.base_dir, args.input_dir)
    filelist = glob(input_dir + '/*.wav')

    makedirs(output_dir, exist_ok=True)
    extract_embeddings(filelist, output_dir, args.model_name)


if __name__ == "__main__":
    main()
