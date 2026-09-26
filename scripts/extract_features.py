"""
Extrai um vetor de features (embedding) de cada imagem do dataset usando a
MobileNetV2 pre-treinada na ImageNet, sem a camada de classificacao final,
so a parte convolucional seguida de global average pooling. A rede nunca viu
essas classes de produto especificamente, mas as camadas convolucionais
aprenderam a reconhecer forma, textura e cor de forma generica o suficiente
para servir como um "descritor visual" de qualquer imagem, exatamente o que
um sistema de recomendacao por aparencia precisa: duas imagens com embeddings
proximos tendem a ser visualmente parecidas, mesmo sem nenhum dado textual.
"""

import csv
import os

os.environ.setdefault("USE_TORCH_XLA", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

IMG_SIZE = 224
METADATA_PATH = "dataset/metadata.csv"
OUT_PATH = "data/features.npz"


def load_metadata():
    with open(METADATA_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.array(img, dtype="float32")


def main():
    os.makedirs("data", exist_ok=True)
    rows = load_metadata()
    print(f"{len(rows)} imagens para processar")

    model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg",
                         input_shape=(IMG_SIZE, IMG_SIZE, 3))

    batch_size = 32
    all_features = []
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        batch_imgs = np.stack([load_image(r["path"]) for r in batch_rows])
        batch_imgs = preprocess_input(batch_imgs)
        feats = model.predict(batch_imgs, verbose=0)
        all_features.append(feats)
        print(f"{min(i + batch_size, len(rows))}/{len(rows)} processadas")

    features = np.concatenate(all_features, axis=0)
    ids = [r["id"] for r in rows]
    classes = [r["class"] for r in rows]
    paths = [r["path"] for r in rows]

    np.savez(OUT_PATH, features=features, ids=ids, classes=classes, paths=paths)
    print(f"\nFeatures salvas em {OUT_PATH}, shape: {features.shape}")


if __name__ == "__main__":
    main()
