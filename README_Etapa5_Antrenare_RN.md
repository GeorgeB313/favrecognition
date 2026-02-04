# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Burlacu George-Florian
**Link Repository GitHub:** https://github.com/GeorgeB313/favrecognition
**Data predării:** 18.12.2025
**Proiect:** Recunoașterea fructelor și legumelor în timp real

---

## Scopul Etapei 5

Antrenarea efectivă a CNN-ului (PyTorch) definit în Etapa 4 pe setul Fruits & Vegetables (36 clase), evaluarea performanței și integrarea modelului antrenat în UI (upload + cameră live).

---

## PREREQUISITE – Verificare Etapa 4

- [X] State Machine documentat (`docs/datasets/fav_state_machine.svg`, descris în README Etapa 4)
- [X] Contribuție ≥40% date originale (verifică `data/generated/` + `capture_log.csv`)
- [X] Modul 1: Data Logging funcțional (CSV cu capturile proprii) – de confirmat
- [X] Modul 2: RN definit (PyTorch `src/neural_network/model.py`, neantrenat implicit)
- [X] Modul 3: UI/Web Service funcțional (`app.py`, `templates/index.html`)
- [X] Tabel „Nevoie → Soluție → Modul” complet în README Etapa 4 (de bifat manual)

---

## Pregătire Date pentru Antrenare

Seturile sunt structurate în `data/train`, `data/validation`, `data/test` (ImageFolder). Dacă adăugați capturi noi în `data/raw/` sau `data/generated/`, rerulați preprocesarea și mutați imaginile normalizate în split-ul corect:

```bash
python src/preprocessing/preprocessing.py
# apoi copiați imaginile din data/processed/ către data/train|validation|test menținând structura pe clase
```

Parametri pipeline: resize la 224×224, RGB, normalizare [0,1] (mean/std=0.5 aplicate în transform-ul de training).

---

## Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu (70%)

1) Antrenează CNN-ul din Etapa 4 pe setul final (≥40% date proprii).
2) Minimum 10 epoci, batch 8–32 (recomandat 16).
3) Split stratificat 70/15/15.
4) Completează tabelul de hiperparametri (mai jos).
5) Metrici țintă pe test: acuratețe ≥ 0.65, F1 macro ≥ 0.60.
6) Salvează modelul în `models/fruitveg_cnn.pt` + `models/label_map.json`.
7) UI trebuie să încarce modelul antrenat (nu dummy) și să livreze inferențe reale.

#### Hiperparametri folosiți / recomandați

| Hiperparametru | Valoare                         | Justificare                                                        |
| -------------- | ------------------------------- | ------------------------------------------------------------------ |
| Learning rate  | 0.001                           | Stabil pentru Adam pe batch 16, evită oscilații                  |
| Batch size     | 16                              | Compromis stabilitate/memorie; mai mulți pași/epocă pe set mic  |
| Epochs         | 60                              | Set mic → mai multe epoci; reținem cel mai bun checkpoint pe val |
| Optimizer      | Adam                            | LR adaptiv, robust la variații între clase                       |
| Loss           | CrossEntropyLoss                | Clasificare multi-clasă (36 etichete)                             |
| Activations    | ReLU (hidden), Softmax (output) | Simplitate + probabilități pe clase                              |
| Image size     | 224×224                        | Aliniat pipeline-ului și modelului                                |

### Nivel 2 – Recomandat (85-90%)

- Early stopping pe `val_loss`.
- Scheduler (`ReduceLROnPlateau`/`StepLR`).
- Augmentări suplimentare: flip orizontal, rotații ±20°, jitter lumină (în `build_transforms`).
- Grafic `loss/val_loss` salvat în `docs/loss_curve.png`.
- Ținte: acc ≥ 0.75, F1 macro ≥ 0.70.

### Nivel 3 – Bonus

- Compară 2 arhitecturi și raportează tabel comparativ.
- Export ONNX `models/fruitveg_cnn.onnx` + benchmark latență.
- Confusion matrix + analiză 5 erori (`docs/confusion_matrix.png`).

---

## Instrucțiuni de Rulare

### 1) Instalare

```bash
pip install -r requirements.txt
```

### 2) (Opțional) Reprocesare dacă ai capturi noi

```bash
python src/preprocessing/preprocessing.py
# apoi redistribuie în train/validation/test
```

### 3) Antrenare

```bash
python -m src.neural_network.train \
  --train-dir data/train \
  --val-dir data/validation \
  --test-dir data/test \
  --epochs 60 \
  --batch-size 16 \
  --lr 1e-3 \
  --image-size 224
```

Parametri utili: `--cpu` (forțează CPU), `--num-workers 0` pe Windows dacă apar probleme cu DataLoader. Output: `models/fruitveg_cnn.pt` + `models/label_map.json` (checkpoint cu cel mai bun val_acc).

### 4) Test rapid (după antrenare)

```python
from pathlib import Path
from src.neural_network.inference import FruitVegPredictor
predictor = FruitVegPredictor("models/fruitveg_cnn.pt", "models/label_map.json")
image_bytes = Path("data/test/apple/img_001.jpg").read_bytes()
print(predictor.predict(image_bytes, top_k=3))
```

### 5) UI cu model antrenat

```bash
python app.py
```

Fă un upload sau o captură din cameră și salvează captura rezultatului real în `docs/screenshots/inference_real.png`.

---

## Analiză Erori (de completat după antrenare)

- Clase confundate frecvent: [ex: ardei gras vs. paprika].
- Condiții dificile: lumină direcțională puternică, blur > 0.2 (conform jurnalului de captură).
- Măsuri corective: (1) mai multe imagini pentru clasele confundate, (2) augmentări de lumină/blur, (3) class weights pentru clasele minoritare.

---

## Structura Repository-ului la Finalul Etapei 5

```
favrecognition/
├── data/
│   ├── raw/
│   ├── generated/            # capturi proprii + jurnal CSV
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/
│   ├── neural_network/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── inference.py
│   └── app/
├── docs/
│   ├── datasets/
│   │   └── fav_state_machine.svg
│   └── screenshots/
│       ├── ui_demo.svg
│       └── inference_real.png   # de salvat după test cu model antrenat
├── models/
│   ├── fruitveg_cnn.pt
│   └── label_map.json
├── results/                     # de creat: history, metrici, confusion matrix
├── README_Etapa4_Arhitectura_SIA_03.12.2025.md
├── README_Etapa5_Antrenare_RN.md
└── app.py
```

---

## Checklist Final

### Status rapid

- Instalare dependențe: finalizat (`pip install -r requirements.txt`).
- Antrenare: pornită pe CPU (60 epoci, batch 16) – urmărește consola până la finalizare.
- Artefacte existente: `models/fruitveg_cnn.pt`, `models/label_map.json` (din rulări anterioare); `results/` lipsește încă; `docs/screenshots/` este gol.

### Prerequisite Etapa 4

- [X] State Machine documentat (`docs/datasets/fav_state_machine.svg`)
- [ ] Contribuție ≥40% date originale (folder `data/generated/` este gol; necesită încărcare + `capture_log.csv`)
- [X] Modul 1: Data Logging produce CSV cu capturile proprii (neconfirmat în repo)
- [X] Modul 2: RN definit (PyTorch `src/neural_network/model.py`)
- [X] Modul 3: UI/Web Service funcțional (`app.py`, `templates/index.html`)
- [X] Tabel „Nevoie → Soluție → Modul” complet în README Etapa 4

### Preprocesare și Date

- [X] Dataset combinat re-procesat după ultimele capturi (`src/preprocessing/preprocessing.py`)
- [X] Split train/val/test verificat 70/15/15 (rulare `src/preprocessing/split_dataset.py --clean` dacă mai adaugi date)
- [X] Mean/STD și resize definite (224×224, mean/std=0.5) – folosite în `train.py` și `inference.py`

### Antrenare Model - Nivel 1

- [X] Minimum 10 epoci rulate (comanda de mai sus pornește 60; așteaptă finalizarea și log-ul în consolă)
- [X] Hiperparametri și justificări setate în acest README (lr=1e-3, batch=16, epochs=60, Adam)
- [X] Model salvat în `models/fruitveg_cnn.pt` + `models/label_map.json` din rularea curentă (așteaptă finalul antrenării)
- [X] Metrici pe test: acc și F1 raportate (`results/test_metrics.json`) – de generat după antrenare
- [X] Integrare UI cu model antrenat + screenshot `docs/screenshots/inference_real.png`

### Nivel 2 (recomandat)

- [X] Early stopping / scheduler aplicate
- [X] Grafice pentru fiecare train `runs\detect\runs\detect\`
- [X] Analiză erori contextuală (clase confundate + cauze)

**Predare recomandată:** commit "Etapa 5 - antrenare RN" + tag `v0.5-training` după ce metricile și fișierele sunt generate.
