"""
Monta um subconjunto do Fashion Product Images Dataset (Kaggle) com N imagens
de cada uma das classes escolhidas, copiando os arquivos e salvando um CSV
de metadados usado pelos proximos scripts.

O dataset baixado do Kaggle (nao incluso neste repositorio, veja o README)
deve estar extraido em RAW_DATA_DIR antes de rodar este script:

    kaggle datasets download -d paramaggarwal/fashion-product-images-small
    unzip fashion-product-images-small.zip -d raw_data/

Os arquivos dataset/ ja commitados neste repositorio sao o resultado deste
script, entao rodar de novo so e necessario para gerar um subconjunto
diferente (outras classes, outro N_PER_CLASS).
"""

import csv
import os
import random
import shutil

random.seed(42)

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "raw_data")
SRC_IMAGES = os.path.join(RAW_DATA_DIR, "images")
SRC_STYLES = os.path.join(RAW_DATA_DIR, "styles.csv")
DST_ROOT = "dataset"
N_PER_CLASS = 150

CLASSES = ["Watches", "Tshirts", "Casual Shoes", "Handbags", "Sunglasses"]


def main():
    by_class = {c: [] for c in CLASSES}
    with open(SRC_STYLES, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            article = row.get("articleType", "")
            if article in by_class:
                by_class[article].append(row["id"])

    os.makedirs(DST_ROOT, exist_ok=True)
    metadata = []
    for cls, ids in by_class.items():
        random.shuffle(ids)
        chosen = ids[:N_PER_CLASS]
        cls_dir = os.path.join(DST_ROOT, cls.replace(" ", "_").lower())
        os.makedirs(cls_dir, exist_ok=True)
        n_copied = 0
        for img_id in chosen:
            src = os.path.join(SRC_IMAGES, f"{img_id}.jpg")
            if not os.path.exists(src):
                continue
            dst = os.path.join(cls_dir, f"{img_id}.jpg")
            shutil.copy(src, dst)
            metadata.append({"id": img_id, "class": cls, "path": dst})
            n_copied += 1
        print(f"{cls}: {n_copied} imagens")

    with open(os.path.join(DST_ROOT, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "class", "path"])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\nTotal: {len(metadata)} imagens em {len(CLASSES)} classes")
    print(f"Metadados salvos em {DST_ROOT}/metadata.csv")


if __name__ == "__main__":
    main()
