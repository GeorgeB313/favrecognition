# Data Acquisition / Logging

Acest modul acoperă capturile proprii pentru cele 36 de clase.

## Flux
1. Captură manuală cu camera (laptop/telefon) – 5+ imagini/clasă.
2. Organizare în `data/raw/<clasa>/`.
3. Rulare preprocesare pentru normalizare și split:
   ```bash
   python src/preprocessing/preprocessing.py
   ```
4. Jurnalizarea capturilor în `data/generated/capture_log.csv` (metadate: clasă, rezoluție, blur_score estimat, condiții de lumină).

## Parametri capturați în jurnal
- `timestamp` – ora ISO a capturii
- `class` – eticheta aleasă manual
- `file_path` – locația imaginii brute
- `width` / `height` – rezoluția înainte de resize
- `blur_score` – scor simplu estimat (0-1, mai mic e mai clar)
- `lighting` – condiții (indoor-led / natural-window / mixed)

## Extindere
- Automatizare blur_score: integrați un laplacian variance check înainte de acceptarea cadrului.
- Captură asistată: adăugați un mic script în `src/data_acquisition/` care deschide webcam-ul și salvează cadrele direct structurat pe clase.
