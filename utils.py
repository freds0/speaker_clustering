import argparse
import torch
from os.path import basename, join, exists, dirname, sep
from os import makedirs
from glob import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import csv
import shutil
from tqdm import tqdm
import librosa 
import soundfile as sf


def normalize_audios(input_path, output_path, file_ext='wav', force=False):
    makedirs(output_path, exist_ok=True)
    for input_filepath in tqdm(glob(join(input_path, '*.{}'.format(file_ext)))):
        #folder = dirname(input_filepath).split(sep)[1:]
        #folder = join(*folder)
        filename = basename(input_filepath)
        output_filepath = join(output_path, filename)

        #if (not(exists(dirname(output_filepath))) and (force)):
        #    makedirs(dirname(output_filepath))

        if force:
            waveform, sr = librosa.load(input_filepath, sr=None)
            norm_waveform = librosa.util.normalize(waveform)
            if file_ext == 'wav':
                sf.write(output_filepath, norm_waveform, sr, 'PCM_16')
            elif file_ext == 'flac':
                sf.write(output_filepath, norm_waveform, sr, 'PCM_24')  # Use 'PCM_24' for FLAC

        else:
            print("norm {} {}".format(input_filepath, output_filepath))


def move_files(input_csv, input_dir, output_dir, file_ext='wav'):
    # Criar a pasta de destino se não existir
    #if not exists(output_dir):
    #    makedirs(output_dir)

    # Ler o arquivo CSV
    with open(input_csv, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Pular a linha de cabeçalho, se houver

        # Percorrer as linhas do arquivo CSV
        for row in reader:
            filename = row[0]
            filename = f"{filename}.{file_ext}"
            folder = row[1]
            filepath = join(input_dir, filename)

            # Verificar se o arquivo existe
            if exists(filepath):
                # Criar a pasta com o nome da coluna 1, se não existir
                ### folder_path = join(output_dir, folder)
                folder = str(int(folder) + 1)
                folder_path = join(output_dir + '_' + folder)
                if not exists(folder_path):
                    makedirs(folder_path)

                # Mover o arquivo para a pasta de destino
                shutil.copy(filepath, folder_path)
                #print(f"Arquivo {file_name} movido para a pasta {folder_path}.")
            else:
                print(f"Arquivo {filepath} não encontrado!")


def save_plot(embeddings, labels, output_filepath='clusters.png', show_plot=False):


    # Reduzir a dimensionalidade dos dados usando o PCA
    pca = PCA(n_components=2)
    embeddings_with_reduced_dimensionality = pca.fit_transform(embeddings)

    # Plotar os pontos coloridos pelos labels dos clusters
    plt.scatter(embeddings_with_reduced_dimensionality[:, 0], 
                embeddings_with_reduced_dimensionality[:, 1], 
                c=labels
    )
    plt.title('Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(output_filepath)
    if show_plot:
        plt.show()
