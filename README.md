<<<<<<< HEAD
# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Burlacu George-Florian
**Data:** 2025-2026

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

## 1. Structura Repository-ului Github (versiunea Etapei 3)

```
favrecognition/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)
```

---

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Baza de Date Fruits-360
* **Modul de achiziție:** ☐ Fisier Extern
* **Condițiile colectării:** Seturi de imagini in scenarii diverse.

### 2.2 Caracteristicile dataset-ului

* **Număr estimat de imagini:** ~ 20.000 de imagini
* **Tipuri de date:** ☐ Imagini
* **Format fișiere:** ☐ PNG / ☐ JPG / ☐ JPEG

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere**         | **Domeniu valori** |
| ------------------------- | ------------- | ----------------- | --------------------------- | ------------------------ |
| img_rgb                   | imagine       | pixeli            | valorile RGB ale imaginii   | 0-255                    |
| label                     | categorial    | -                 | tipul fructului sau legumei | lista clase              |

**Fișier recomandat:**  `data/README.md`

---

## 3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Număr imagini pe clasă.**
* **Dimensiuni standardizate.**
* **Distribuții ale canalelor de culoare.**

### 3.2 Analiza calității datelor

* **Identificarea imaginilor neclare.**
* **Identificarea rezoluțiilor diferite.**
* **Verificarea variațiilor de lumină.**

### 3.3 Probleme identificate

* **Dezechilibru între clase.**
* **Fundal neuniform în unele imagini.**
* **Iluminare variabilă.**

---

## 4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* Eliminarea duplicatelor.
* Eliminarea imaginilor neclare.
* Corectarea luminii prin normalizare.

### 4.2 Transformarea caracteristicilor

* **Redimensionare la 100x100.**
* **Normalizare valori 0–1.**
* **Augmentare prin rotiri, decupări și oglinziri.**
* **One-hot encoding pentru etichete.**

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**

* 70 la sută train.
* 15 la sută validation.
* 15 la sută test.

### 4.4 Salvarea rezultatelor preprocesării

* **Salvarea datelor procesate în `data/processed/`.**
* **Salvarea seturilor finale în folderele dedicate.**
* **Salvarea parametrilor de preprocesare în `config/preprocessing.json`.**

---

## 5. Fișiere Generate în Această Etapă

* `data/raw/` imagini brute.
* `data/processed/` imagini preprocesate.
* `data/train/`, `data/validation/`, `data/test/` seturi finale.
* `src/preprocessing/` funcții de preprocesare.
* `data/README.md` document descriere.

---

## 6. Stare Etapă (de completat de student)

- [ ] Structură repository configurată
- [ ] Dataset analizat (EDA realizată)
- [ ] Date preprocesate
- [ ] Seturi train/val/test generate
- [ ] Documentație actualizată în README + `data/README.md`
=======
# FruitAndVegetablesRecognition
>>>>>>> 14030ebb6d719285faaaa20464c8d1085f505d7b
