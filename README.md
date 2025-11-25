🩺 Breast Cancer Detector — YOLO11

Detector de anomalias em mamografias utilizando Deep Learning (YOLO)

Este projeto implementa um modelo YOLO11 treinado para detectar possíveis tumores malignos e benignos em mamografias, utilizando máscaras para gerar as labels e diversas técnicas de augmentação para melhorar o desempenho em cenários reais.

📌 📁 Estrutura do Projeto
breast-cancer-detector/
│
├── data/
│   ├── data.yaml
│   └── datasets/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       ├── labels/
│       │   ├── train/
│       │   └── val/
│       └── classes.txt
│
├── models/
│   └── yolo11.pt
│
├── scripts/
│   ├── detector_tumores.py
│   ├── augment.py
│   ├── prepare_dataset.py
│   └── utils.py (opcional)
│
├── results/
│   └── training/
│       ├── weights/
│       ├── results.csv
│       ├── confusion_matrix.png
│       └── ...
│
└── README.md

🧠 Objetivo do Projeto

O objetivo central é construir um modelo YOLO capaz de:

✔ Detectar lesões suspeitas em mamografias
✔ Diferenciar normal, benigno e maligno
✔ Ajudar pesquisadores a automatizar diagnósticos
✔ Melhorar a análise de exames com imagens claras ou fora do padrão do dataset de treino

🧩 Classes Utilizadas

O projeto utiliza 3 classes:

ID	Classe
0	Normal
1	Benigno
2	Maligno

Essas classes são definidas em:

data/data.yaml
data/datasets/classes.txt

🗂 Dataset

O dataset foi organizado no formato padrão YOLO:

images/train/
images/val/
labels/train/
labels/val/


As labels foram geradas automaticamente a partir das máscaras originais utilizando o script:

scripts/prepare_dataset.py

🔄 Data Augmentation

Para balancear as classes e aumentar a robustez do modelo, utilizamos:

Rotação

Flip horizontal e vertical

Alteração de brilho

Alteração de contraste

Ruído

Recorte aleatório

Script utilizado:

scripts/augment.py

🤖 Treinamento

O treinamento foi realizado utilizando YOLO11:

yolo train \
    model=models/yolo11.pt \
    data=data/data.yaml \
    epochs=80 \
    imgsz=640 \
    batch=4 \
    device=0


Os resultados ficam salvos em:

results/training/


Inclui:

pesos treinados

matriz de confusão

curvas P/R

curvas Box P/R

batches de validação

CSV com métricas completas

📈 Resultados

📌 Os gráficos gerados pelo YOLO incluem:

Confusion Matrix

Normalized Confusion Matrix

Precision x Recall

F1 Curve

Loss Curve

Batch Predictions

Esses arquivos estão em:

results/training/

🔍 Inferência

O script de inferência está em:

scripts/detector_tumores.py


Exemplo de uso:

python3 scripts/detector_tumores.py --image caminho/da/imagem.jpg


O script:

✔ Carrega o modelo
✔ Faz a predição
✔ Salva a imagem anotada
✔ Gera as coordenadas das detecções
✔ Salva o relatório JSON

🧪 Exemplo de Execução
from ultralytics import YOLO

model = YOLO("models/yolo11.pt")

results = model("exemplo.jpg")

results[0].show()

📄 Arquivo data.yaml
train: data/datasets/images/train
val: data/datasets/images/val

nc: 3
names: ["normal", "benigno", "maligno"]



