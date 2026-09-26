"""
Avaliacao quantitativa do sistema de recomendacao: para cada imagem do
dataset, busca os K vizinhos mais proximos por similaridade visual e mede
que fracao deles pertence a mesma categoria da consulta.

Isso nao e uma metrica perfeita, similaridade visual de verdade nao precisa
respeitar categoria (um sapato branco pode ser mais parecido visualmente com
uma bolsa branca do que com um sapato preto), mas serve como um sinal
quantitativo razoavel: se o extrator de features capturasse só ruido, o
acerto por categoria seria em torno de 1/5 = 20% (5 classes). Um valor bem
acima disso indica que os embeddings realmente capturam alguma nocao visual
coerente que tende a se alinhar com a categoria do produto.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

FEATURES_PATH = "data/features.npz"
K_VALUES = [1, 3, 5, 10]


def main():
    npz = np.load(FEATURES_PATH, allow_pickle=True)
    features, classes = npz["features"], npz["classes"]

    max_k = max(K_VALUES)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine")
    nn.fit(features)
    _, indices = nn.kneighbors(features)

    precisions = {}
    for k in K_VALUES:
        hits = []
        for i in range(len(features)):
            neighbor_idxs = indices[i][1:k + 1]  # exclui a propria imagem
            match_frac = np.mean(classes[neighbor_idxs] == classes[i])
            hits.append(match_frac)
        precisions[k] = np.mean(hits)
        print(f"precision@{k}: {precisions[k]:.4f}")

    chance_level = 1 / len(set(classes))
    print(f"nivel de chance (5 classes balanceadas): {chance_level:.4f}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ks = list(precisions.keys())
    vals = list(precisions.values())
    ax.bar([str(k) for k in ks], vals, color="#1f5fa8")
    ax.axhline(chance_level, color="#c62828", linestyle="--", label=f"nível de chance ({chance_level:.2f})")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("K (número de recomendações)")
    ax.set_ylabel("Precisão por categoria")
    ax.set_title("Precision@K do sistema de recomendação")
    ax.legend()
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/precision_at_k.png", dpi=600)
    plt.close(fig)
    print("Salvo results/precision_at_k.png")


if __name__ == "__main__":
    main()
