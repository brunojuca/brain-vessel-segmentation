import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm
import glob
import joblib
import argparse

PATCH_SIZE = 7
HALF = PATCH_SIZE // 2


def extract_patches(img, label, num_samples_per_class=5000):
    h, w = img.shape
    X, y = [], []

    # Coletar amostras de fundo (0)
    while len(y) < num_samples_per_class:
        i = np.random.randint(HALF, h - HALF)
        j = np.random.randint(HALF, w - HALF)
        if label[i, j] < 128:
            patch = img[i - HALF:i + HALF + 1, j - HALF:j + HALF + 1]
            X.append(patch.flatten())
            y.append(0)

    # Coletar amostras de vasos (1)
    while len(y) < 2 * num_samples_per_class:
        i = np.random.randint(HALF, h - HALF)
        j = np.random.randint(HALF, w - HALF)
        if label[i, j] >= 128:
            patch = img[i - HALF:i + HALF + 1, j - HALF:j + HALF + 1]
            X.append(patch.flatten())
            y.append(1)

    return np.array(X), np.array(y)


def load_data(image_dir, label_dir, limit=None):
    X_all, y_all = [], []
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    label_paths = sorted(glob.glob(os.path.join(label_dir, "*.png")))

    if limit:
        image_paths = image_paths[:limit]
        label_paths = label_paths[:limit]

    for img_path, label_path in tqdm(zip(image_paths, label_paths), total=len(image_paths), desc="Extraindo patches"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if img is None or label is None:
            print(f"Erro ao carregar {img_path} ou {label_path}")
            continue

        X, y = extract_patches(img, label)
        X_all.append(X)
        y_all.append(y)

    return np.vstack(X_all), np.hstack(y_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina um Random Forest para segmentação de vasos.")
    parser.add_argument("--image_dir", type=str, required=True, help="Diretório com as imagens de entrada.")
    parser.add_argument("--label_dir", type=str, required=True, help="Diretório com as máscaras binárias.")
    parser.add_argument("--output_model", type=str, default="models/random_forest_patch.pkl", help="Caminho para salvar o modelo treinado.")
    parser.add_argument("--limit", type=int, default=None, help="Número máximo de imagens a carregar (padrão: todas).")

    args = parser.parse_args()

    X, y = load_data(args.image_dir, args.label_dir, limit=args.limit)
    print(f"Treinando com {X.shape[0]} amostras, {X.shape[1]} features cada.")

    clf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    print("Classification Report no mesmo dataset de treino:")
    print(classification_report(y, y_pred))

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump(clf, args.output_model)
    print(f"Modelo salvo em: {args.output_model}")
