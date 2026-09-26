"""
Sistema de recomendacao por similaridade visual: dado um produto (imagem),
recomenda os K produtos com aparencia mais parecida, usando a distancia de
cosseno entre os embeddings extraidos por scripts/extract_features.py.

Nao usa nenhum dado textual do produto (preco, marca, categoria), so a
similaridade entre os vetores de features visuais.
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


def recommend(query_idx, features, k=TOP_K):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(features)
    distances, indices = nn.kneighbors(features[query_idx:query_idx + 1])
    return indices[0][1:], distances[0][1:]  # exclui a propria imagem (vizinho mais proximo de si mesma)


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

    neighbor_idxs, distances = recommend(query_idx, features)
    print(f"Consulta: {paths[query_idx]} (classe: {classes[query_idx]})")
    for idx, dist in zip(neighbor_idxs, distances):
        print(f"  recomendado: {paths[idx]} (classe: {classes[idx]}, distância cosseno: {dist:.4f})")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/recomendacao_{ids[query_idx]}.png"
    plot_recommendation(query_idx, neighbor_idxs, distances, features, classes, paths, out_path)
    print(f"Salvo {out_path}")


if __name__ == "__main__":
    main()
