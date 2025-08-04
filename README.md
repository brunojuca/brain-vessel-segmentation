# Segmentação de Vasos Sanguíneos Cerebrais

Este projeto aplica técnicas de aprendizado de máquina para realizar a **segmentação de vasos sanguíneos cerebrais** em angiogramas.

O código está dividido em dois scripts principais:

- [`scripts/train_rd.py`](scripts/train_rf.py): treinamento do modelo Random Forest.
- [`scripts/test_rf.py`](scripts/test_rf.py): avaliação e geração das segmentações a partir do modelo treinado.

## 📂 Estrutura esperada de diretórios

```
brain-vessel-segmentation/
│
├── data/
│ └── DIAS/
│ ├── training/
│ │ ├── last_seq_images/
│ │ └── labels/
│ └── test/
│ ├── last_seq_images/
│ └── labels/
│
├── models/
│ └── random_forest_patch.pkl
│
├── outputs/
│ └── rf_segmented_test/
│
├── scripts/
│ ├── train_rf.py
│ └── test_rf.py
│
├── uv.lock
└── pyproject.toml
```

## 📦 Ambiente e Dependências

Este projeto utiliza `uv` como gerenciador de pacotes rápido e compatível com `pip`.

### Instalar o `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Instalar as dependências

```bash
uv sync
source .venv/bin/activate
```

## 🚀 Treinamento

### Executar

```bash
python scripts/train_rf.py \
  --image_dir data/DIAS/training/last_seq_images \
  --label_dir data/DIAS/training/labels \
  --output_model models/random_forest_patch.pkl
```

#### Argumentos

| Parâmetro      | Descrição                                               |
| -------------- | ------------------------------------------------------- |
| `--image_dir`  | Caminho para as imagens de entrada (grayscale PNG).     |
| `--label_dir`  | Caminho para as labels de referência (grayscale PNG). |
| `--output_model` | Caminho para salvar o modelo treinado `.pkl`.           |
| `--limit`      | Quantidade máxima de pares imagem/label a usar (Default: None).       |

O modelo é treinado com amostras balanceadas de patches extraídas das regiões de fundo (classe 0) e de vasos (classe 1), com PATCH_SIZE = 7 por padrão.

## 🧪 Avaliação / Segmentação

### Executar

```bash
python scripts/test_rf.py \
  --model_path models/random_forest_patch.pkl \
  --image_dir data/DIAS/test/last_seq_images \
  --label_dir data/DIAS/test/labels \
  --output_dir outputs/rf_segmented_output
```

#### Argumentos

| Parâmetro      | Descrição                                      |
| -------------- | ---------------------------------------------- |
| `--model_path` | Caminho para o modelo `.pkl` treinado.         |
| `--image_dir`  | Caminho das imagens a serem segmentadas.       |
| `--label_dir`  | Caminho das máscaras reais (para avaliação).   |
| `--output_dir` | Diretório onde salvar as segmentações geradas. |

## 📊 Métricas

Após a segmentação, o script exibe um `classification_report` com métricas como:

- Precision
- Recall
- F1-score
- Accuracy

## 📍 Dataset

As imagens e máscaras vêm do dataset DIAS (DIAS: A dataset and benchmark for intracranial artery segmentation in DSA sequences).

Liu W, Tian T, Wang L, Xu W, Li L, Li H, Zhao W, Tian S, Pan X, Deng Y, Gao F, Yang H, Wang X, Su R. DIAS: A dataset and benchmark for intracranial artery segmentation in DSA sequences. Med Image Anal. 2024 Oct;97:103247. doi: 10.1016/j.media.2024.103247. Epub 2024 Jun 18. PMID: 38941857.
