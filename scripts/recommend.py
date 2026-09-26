"""
Sistema de recomendacao por similaridade visual: dado um produto (imagem),
recomenda os K produtos com aparencia mais parecida, usando a distancia de
cosseno entre os embeddings extraidos por scripts/extract_features.py.

A busca e restrita a produtos da MESMA categoria da consulta: um relogio
sempre recomenda outros relogios, um oculos de sol sempre recomenda outros
oculos de sol, nunca um produto de tipo diferente por acaso ter aparencia
parecida. A categoria aqui funciona como o "tipo de produto" (um relogio
continua sendo relogio independente da marca ou preco), nao como um dado
textual de marketing, o ranking dentro dela e feito 100% por aparencia
visual, sem olhar preco, marca ou nome do produto.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors

FEATURES_PATH = "data/features.npz"
TOP_K = 5


def load_features():
    npz = np.load(FEATURES_PATH, allow_pickle=True)
    return npz["features"], npz["ids"], npz["classes"], npz["paths"]


def recommend(query_idx, features, classes, k=TOP_K):
    query_class = classes[query_idx]
    same_class_idxs = np.where(classes == query_class)[0]

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(same_class_idxs)), metric="cosine")
    nn.fit(features[same_class_idxs])
    distances, local_indices = nn.kneighbors(features[query_idx:query_idx + 1])

    global_indices = same_class_idxs[local_indices[0]]
    # remove a propria imagem da lista (vizinho mais proximo de si mesma, distancia 0)
    mask = global_indices != query_idx
    return global_indices[mask][:k], distances[0][mask][:k]


def plot_recommendation(query_idx, neighbor_idxs, distances, features, classes, paths, out_path):
    fig, axes = plt.subplots(1, len(neighbor_idxs) + 1, figsize=(2.2 * (len(neighbor_idxs) + 1), 2.6))

    query_img = Image.open(paths[query_idx])
    axes[0].imshow(query_img)
    axes[0].set_title(f"Consulta\n{classes[query_idx]}", fontsize=9)
    axes[0].axis("off")
    for ax in axes[0].spines.values():
        pass

    for ax, idx, dist in zip(axes[1:], neighbor_idxs, distances):
        img = Image.open(paths[idx])
        ax.imshow(img)
        match = "✓" if classes[idx] == classes[query_idx] else "✗"
        ax.set_title(f"{classes[idx]} {match}\ndist={dist:.3f}", fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    features, ids, classes, paths = load_features()
    query_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    neighbor_idxs, distances = recommend(query_idx, features, classes)
    print(f"Consulta: {paths[query_idx]} (classe: {classes[query_idx]})")
    for idx, dist in zip(neighbor_idxs, distances):
        print(f"  recomendado: {paths[idx]} (classe: {classes[idx]}, distância cosseno: {dist:.4f})")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/recomendacao_{ids[query_idx]}.png"
    plot_recommendation(query_idx, neighbor_idxs, distances, features, classes, paths, out_path)
    print(f"Salvo {out_path}")


if __name__ == "__main__":
    main()
