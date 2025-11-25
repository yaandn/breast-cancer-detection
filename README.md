🩺 Breast Cancer Detector — YOLO11
Detector de anomalias em mamografias utilizando Deep Learning

Este projeto implementa um modelo YOLO11 treinado para identificar possíveis tumores malignos, benignos ou padrões normais em mamografias.
O pipeline utiliza máscaras para gerar labels, divide as imagens em classes e aplica augmentations para aumentar a robustez do modelo em cenários reais.

📁 Estrutura do Projeto
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
│   └── utils.py
│
├── results/
│   └── training/
│       ├── weights/
│       ├── results.csv
│       ├── confusion_matrix.png
│       └── ...
│
└── README.md

🎯 Objetivo do Projeto

O objetivo principal deste repositório é:

✔ Detectar lesões suspeitas em mamografias

✔ Diferenciar normal, benigno e maligno

✔ Auxiliar pesquisadores e profissionais na análise das imagens

✔ Aumentar a precisão em imagens muito claras ou fora do padrão do dataset

🧩 Classes Utilizadas

O modelo trabalha com 3 classes:

ID	Classe
0	Normal
1	Benigno
2	Maligno

Essas classes estão definidas em:

data/data.yaml

data/datasets/classes.txt

🗂 Dataset

O dataset está estruturado no formato oficial do YOLO:

images/
   train/
   val/
labels/
   train/
   val/


As labels foram geradas automaticamente pelas máscaras originais utilizando:

scripts/prepare_dataset.py

🔄 Data Augmentation

Para balancear as classes e aumentar a generalização, foram aplicadas:

Rotação

Flip horizontal e vertical

Ajuste de brilho

Ajuste de contraste

Adição de ruído

Crop aleatório

Script usado:

scripts/augment.py

🤖 Treinamento

O treinamento do modelo YOLO11 foi executado com:

yolo train \
    model=models/yolo11.pt \
    data=data/data.yaml \
    epochs=80 \
    imgsz=640 \
    batch=4 \
    device=0


Os resultados ficam em:

results/training/


Incluindo:

✔ Pesos finais

✔ Matriz de confusão

✔ Curvas P/R

✔ Curva F1

✔ Predições de validação

✔ CSV com métricas completas

📈 Resultados Incluídos

O YOLO gera automaticamente:

📊 Confusion Matrix

📊 Normalized Confusion Matrix

📈 Precision × Recall Curve

📈 F1 Curve

📉 Loss Curve

🖼 Batch Predictions

Localização:

results/training/

🔍 Inferência

Para realizar predições em novas imagens:

Script:

scripts/detector_tumores.py


Exemplo de execução:

python3 scripts/detector_tumores.py --image caminho/da/imagem.jpg


O script:

Carrega o modelo

Faz a predição

Salva a imagem anotada

Exporta coordenadas

Gera relatório JSON

🧪 Exemplo simples de uso com Python
from ultralytics import YOLO

model = YOLO("models/yolo11.pt")

results = model("exemplo.jpg")

results[0].show()

📄 Arquivo data.yaml
train: data/datasets/images/train
val: data/datasets/images/val

nc: 3
names: ["normal", "benigno", "maligno"]
