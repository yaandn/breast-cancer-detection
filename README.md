<h1>🩺 Breast Cancer Detector — YOLO11</h1>
<p>Detector de anomalias em mamografias utilizando Deep Learning</p>

<p>Este projeto implementa um modelo <strong>YOLO11</strong> treinado para identificar possíveis tumores <strong>malignos</strong>, <strong>benignos</strong> ou padrões <strong>normais</strong> em mamografias. O pipeline utiliza máscaras para gerar labels, divide as imagens em classes e aplica <strong>augmentations</strong> para aumentar a robustez do modelo em cenários reais.</p>

<hr />

<h2>📁 Estrutura do Projeto</h2>

<pre>
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
</pre>

<hr />

<h2>🎯 Objetivo do Projeto</h2>

<ul>
  <li>✔ Detectar lesões suspeitas em mamografias</li>
  <li>✔ Diferenciar <strong>normal</strong>, <strong>benigno</strong> e <strong>maligno</strong></li>
  <li>✔ Auxiliar pesquisadores e profissionais na análise das imagens</li>
  <li>✔ Aumentar a precisão em imagens claras ou fora do padrão</li>
</ul>

<hr />

<h2>🧩 Classes Utilizadas</h2>

<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Classe</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>0</td><td>Normal</td></tr>
    <tr><td>1</td><td>Benigno</td></tr>
    <tr><td>2</td><td>Maligno</td></tr>
  </tbody>
</table>

<p>Definidas em:</p>
<ul>
  <li><code>data/data.yaml</code></li>
  <li><code>data/datasets/classes.txt</code></li>
</ul>

<hr />

<h2>🗂 Dataset</h2>

<p>Organização no formato padrão YOLO:</p>

<pre>
images/
   train/
   val/
labels/
   train/
   val/
</pre>

<p>As labels foram geradas automaticamente a partir das máscaras com:</p>
<code>scripts/prepare_dataset.py</code>

<hr />

<h2>🔄 Data Augmentation</h2>

<p>Para melhorar desempenho e balancear classes, foram usadas:</p>
<ul>
  <li>Rotação</li>
  <li>Flip horizontal/vertical</li>
  <li>Ajuste de brilho</li>
  <li>Ajuste de contraste</li>
  <li>Ruído</li>
  <li>Crop aleatório</li>
</ul>

<p>Script:</p>
<code>scripts/augment.py</code>

<hr />

<h2>🤖 Treinamento</h2>

<p>Comando utilizado:</p>

<pre>
yolo train \
    model=models/yolo11.pt \
    data=data/data.yaml \
    epochs=80 \
    imgsz=640 \
    batch=4 \
    device=0
</pre>

<p>Resultados salvos em:</p>
<code>results/training/</code>

<ul>
  <li>Pesos finais</li>
  <li>Matriz de confusão</li>
  <li>Curvas P/R</li>
  <li>Curva F1</li>
  <li>Predições de validação</li>
  <li>CSV com métricas</li>
</ul>

<hr />

<h2>📈 Resultados Gerados</h2>

<p>YOLO gera automaticamente:</p>
<ul>
  <li>📊 Confusion Matrix</li>
  <li>📊 Normalized Confusion Matrix</li>
  <li>📈 Precision × Recall</li>
  <li>📈 F1 Curve</li>
  <li>📉 Loss Curve</li>
  <li>🖼 Batch Predictions</li>
</ul>

<p>Local:</p>
<code>results/training/</code>

<hr />

<h2>🔍 Inferência</h2>

<p>Script:</p>
<code>scripts/detector_tumores.py</code>

<p>Uso:</p>

<pre>
python3 scripts/detector_tumores.py --image caminho/da/imagem.jpg
</pre>

<p>O script:</p>
<ul>
  <li>Carrega o modelo</li>
  <li>Executa a predição</li>
  <li>Salva a imagem anotada</li>
  <li>Exporta coordenadas</li>
  <li>Gera relatório JSON</li>
</ul>

<hr />

<h2>🧪 Exemplo em Python</h2>

<pre>
from ultralytics import YOLO

model = YOLO("models/yolo11.pt")
results = model("exemplo.jpg")
results[0].show()
</pre>

<hr />

<h2>📄 Arquivo data.yaml</h2>

<pre>
train: data/datasets/images/train
val: data/datasets/images/val

nc: 3
names: ["normal", "benigno", "maligno"]
</pre>

