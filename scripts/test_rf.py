import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.metrics import classification_report
import argparse

PATCH_SIZE = 7
HALF = PATCH_SIZE // 2

def predict_image(model, image, batch_size=10000):
    h, w = image.shape
    padded = np.pad(image, HALF, mode='reflect')
    patches = []

    for i in range(h):
        for j in range(w):
            patch = padded[i:i+PATCH_SIZE, j:j+PATCH_SIZE].flatten()
            patches.append(patch)

    patches = np.array(patches)
    output = np.zeros(h * w, dtype=np.uint8)

    for i in range(0, len(patches), batch_size):
        batch = patches[i:i+batch_size]
        preds = model.predict(batch)
        output[i:i+batch_size] = preds

    label = (output.reshape(h, w) * 255).astype(np.uint8)
    return label

def evaluate_dataset(model, image_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    label_paths = sorted(glob.glob(os.path.join(label_dir, "*.png")))

    y_true_all = []
    y_pred_all = []

    for img_path, label_path in tqdm(zip(image_paths, label_paths), total=len(image_paths), desc="Processando imagens"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if img is None or label is None:
            print(f"Erro ao ler: {img_path} ou {label_path}")
            continue

        pred_label = predict_image(model, img)

        # Binariza máscara real
        label_bin = (label >= 128).astype(np.uint8) * 255

        # Salva predição
        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_dir, filename), pred_label)

        # Flatten para métricas
        y_true_all.extend(label_bin.flatten() // 255)
        y_pred_all.extend(pred_label.flatten() // 255)

    print("Classification Report:")
    print(classification_report(y_true_all, y_pred_all))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia e segmenta imagens com modelo Random Forest.")
    parser.add_argument("--model_path", type=str, required=True, help="Caminho para o modelo treinado (.pkl).")
    parser.add_argument("--image_dir", type=str, required=True, help="Diretório com imagens para segmentar.")
    parser.add_argument("--label_dir", type=str, required=True, help="Diretório com máscaras reais binárias.")
    parser.add_argument("--output_dir", type=str, required=True, help="Diretório para salvar máscaras preditas.")

    args = parser.parse_args()

    print("Carregando modelo...")
    model = joblib.load(args.model_path)

    print("Segmentando imagens e avaliando...")
    evaluate_dataset(model, args.image_dir, args.label_dir, args.output_dir)
