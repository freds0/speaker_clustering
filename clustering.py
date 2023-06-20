#!/usr/bin/env python
# coding: utf-8
import argparse
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
import hdbscan

def get_hdbscan_clusters(embeddings):
    hdb = hdbscan.HDBSCAN(
        algorithm='best', 
        alpha=1.0, 
        approx_min_span_tree=True,
        gen_min_span_tree=False, 
        leaf_size=40, 
        metric='euclidean', 
        min_cluster_size=5, 
        min_samples=None, 
        p=None
    )
    normalized_embeddings = normalize(embeddings)
    hdb.fit(normalized_embeddings)
    return hdb.labels_

def get_dbscan_clusters(embeddings):
    db = DBSCAN(
        eps=0.5, 
        min_samples=3)
    normalized_embeddings = normalize(embeddings)
    db.fit(normalized_embeddings)
    return db.labels_


def get_kmeans_clusters(embeddings, total_clusters):
    kmeans = KMeans(n_clusters=total_clusters)
    normalized_embeddings = normalize(embeddings)
    kmeans.fit(normalized_embeddings)
    return kmeans.labels_


def save_clusters_to_file(filenames, labels, output_filename='output.csv'):
    matrix = np.column_stack((filenames, labels))
    df = pd.DataFrame(matrix)
    df.to_csv(output_filename, index=False)


def main():
    from embeddings import get_filenames_embeddings

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default='input', type=str, help='Input dir')
    parser.add_argument('-f', '--output_file', default='output.csv', type=str, help='Output csv file')
    parser.add_argument('-n', '--n_clusters', default=3, type=int, help='Total de Clusters.')
    args = parser.parse_args()

    print("Loading embeddings...")
    filenames, embeddings = get_filenames_embeddings(args.input_dir)
    print("Clustering...")
    labels = get_kmeans_clusters(embeddings, args.n_clusters)
    print("Saving...")
    save_clusters_to_file(filenames, labels, args.output_file)


if __name__ == "__main__":
    main()
