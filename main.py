#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import argparse
from os.path import join
from os import makedirs, listdir
from glob import glob
from embeddings import get_filenames_embeddings, extract_embeddings
from clustering import get_hdbscan_clusters, save_clusters_to_file
from utils import save_plot, move_files, normalize_audios
from shutil import rmtree

def execute_pileline(input_dir, output_dir, model_name):

    temp_dir = 'temp'
    clusters_csv = join(temp_dir, 'clusters.csv')
    print("Normalizing audios...")
    norm_audios_dir = join(temp_dir, 'norm')
    normalize_audios(input_dir, norm_audios_dir, force=True)

    print("Extracting embeddings...")
    filelist = glob(join(norm_audios_dir, '*.wav'))
    emb_output_dir = join(temp_dir, 'embeddings')
    extract_embeddings(filelist, emb_output_dir, model_name)

    print("Loading embeddings...")
    filenames, embeddings = get_filenames_embeddings(emb_output_dir)

    print("Clustering...")
    labels = get_hdbscan_clusters(embeddings)
    
    print("Saving...")
    #output_csv_filepath = join(output_dir, clusters_csv)
    save_clusters_to_file(filenames, labels, clusters_csv)

    print("Moving files...")
    move_files(clusters_csv, input_dir, output_dir)
    
    print("Plotting...")
    output_img_filepath = join(temp_dir, "clusters.png")
    save_plot(embeddings, labels, output_img_filepath, show_plot=False)
    
    rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default='input', help='Input folder.')
    parser.add_argument('-o', '--output_dir', default='output', help='Output folder.')
    parser.add_argument('-m', '--model', default='speakernet', help='Embedding model: speakernet or titanet.')
    parser.add_argument('-f', '--force', action='store_true', default=False)    
    args = parser.parse_args()

    for input_folder in listdir(args.input_dir):

        input_dir = join(args.input_dir, input_folder)
        output_dir = join(args.output_dir, input_folder)

        #makedirs(output_dir, exist_ok=True)
        
        execute_pileline(input_dir, output_dir, args.model)


if __name__ == "__main__":
    main()
