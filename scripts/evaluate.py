"""
Avaliacao quantitativa do sistema de recomendacao.

Como scripts/recommend.py agora restringe a busca a produtos da mesma
categoria da consulta, a antiga metrica de "precisao por categoria" fica
100% por construcao, deixa de medir qualquer coisa interessante. A pergunta
que importa agora e outra: dentro da categoria, o ranking por aparencia
visual e realmente melhor do que escolher vizinhos ao acaso? Para responder,
comparo, para cada imagem, a distancia media aos seus K vizinhos mais
proximos (dentro da mesma categoria) contra a distancia media a K itens
aleatorios da mesma categoria. Se o extrator de features nao capturasse
nada util, as duas distancias seriam parecidas.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances

FEATURES_PATH = "data/features.npz"
K_VALUES = [1, 3, 5, 10]
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    npz = np.load(FEATURES_PATH, allow_pickle=True)
    features, classes = npz["features"], npz["classes"]

    max_k = max(K_VALUES)
    nn_dist_by_k = {k: [] for k in K_VALUES}
    random_dist_by_k = {k: [] for k in K_VALUES}

    for cls in sorted(set(classes)):
        idxs = np.where(classes == cls)[0]
        cls_features = features[idxs]

        nn = NearestNeighbors(n_neighbors=min(max_k + 1, len(idxs)), metric="cosine")
        nn.fit(cls_features)
        distances, _ = nn.kneighbors(cls_features)
        distances = distances[:, 1:]  # remove a propria imagem (distancia 0)

        for k in K_VALUES:
            nn_dist_by_k[k].extend(distances[:, :k].mean(axis=1))

            for i in range(len(idxs)):
                choices = rng.choice([j for j in range(len(idxs)) if j != i],
                                      size=min(k, len(idxs) - 1), replace=False)
                d = cosine_distances(cls_features[i:i + 1], cls_features[choices])
                random_dist_by_k[k].append(d.mean())

    print("Distância de cosseno média (menor = mais parecido):")
    print(f"{'K':>4} {'Vizinhos por aparência':>24} {'Itens aleatórios (mesma categoria)':>36}")
    means_nn, means_random = [], []
    for k in K_VALUES:
        m_nn, m_rand = np.mean(nn_dist_by_k[k]), np.mean(random_dist_by_k[k])
        means_nn.append(m_nn)
        means_random.append(m_rand)
        print(f"{k:>4} {m_nn:>24.4f} {m_rand:>36.4f}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })

    x = np.arange(len(K_VALUES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(x - width / 2, means_nn, width, label="Recomendado (por aparência)", color="#1f5fa8")
    ax.bar(x + width / 2, means_random, width, label="Aleatório (mesma categoria)", color="#9e9e9e")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("K (número de recomendações)")
    ax.set_ylabel("Distância de cosseno média")
    ax.set_title("Recomendado x aleatório, dentro da mesma categoria")
    ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/recomendado_vs_aleatorio.png", dpi=600)
    plt.close(fig)
    print("\nSalvo results/recomendado_vs_aleatorio.png")


if __name__ == "__main__":
    main()
