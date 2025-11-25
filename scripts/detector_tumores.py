from ultralytics import YOLO
import os
import json

MODEL_PATH = 'runs/train/weights/best.pt'

PASTA_IMAGENS = 'datasets/mamografias/images/val'

CONF_THRESHOLD = 0.7

ARQUIVO_SAIDA_JSON = 'resultados_deteccao.json'
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

try:
    results_list = model.predict(
        source=PASTA_IMAGENS,
        conf=CONF_THRESHOLD
    )
    print(f"Processamento concluído\n{len(results_list)} imagens analisadas.")
except Exception as e:
    print(f"Erro durante a predição: {e}")
    exit()

todos_os_resultados = []

for result in results_list:
    
    nome_imagem = os.path.basename(result.path)
    
    deteccoes_nesta_imagem = []
    
    print(f"  -> Processando: {nome_imagem}")
    
    for box in result.boxes:
        cls = int(box.cls[0]) 
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist() 
        
        deteccao = {
            "classe_id": cls,
            "confianca": conf,
            "coordenadas_xyxy": [round(c, 2) for c in coords] 
        }
        
        deteccoes_nesta_imagem.append(deteccao)
    
    todos_os_resultados.append({
        "imagem": nome_imagem,
        "total_deteccoes": len(deteccoes_nesta_imagem),
        "deteccoes": deteccoes_nesta_imagem
    })

try:
    with open(ARQUIVO_SAIDA_JSON, 'w', encoding='utf-8') as f:
        json.dump(todos_os_resultados, f, indent=4, ensure_ascii=False)
    
    print("Resultados salvos com sucesso!")
    print(f"\nTotal de imagens processadas: {len(todos_os_resultados)}")

except Exception as e:
    print(f"Erro ao salvar o arquivo JSON: {e}")