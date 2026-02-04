# Dataset YOLO pentru detecția fructelor/legumelor

## Structură (deja creată)
```
data/yolo/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

## Format etichete YOLO
Pentru fiecare imagine `images/.../nume.jpg` trebuie să existe un fișier `labels/.../nume.txt` cu linii de forma:
```
<class_id> <x_center> <y_center> <width> <height>
```
Toate valorile sunt normalizate între 0–1 (raportate la lățimea/înălțimea imaginii).

## Clase
Ordinea claselor este definită în [config/fruitveg_detect.yaml](../../config/fruitveg_detect.yaml).

## Workflow recomandat
1. Etichetează imaginile cu **LabelImg** sau **CVAT** în format YOLO.
2. Pune toate imaginile în `data/yolo_raw/images/` și toate etichetele `.txt` în `data/yolo_raw/labels/`.
  - (Opțional) Pre-etichetare automată cu YOLOv8:
    ```
    python -m src.neural_network.prelabel_yolo
    ```
    Apoi deschide în LabelImg și corectează etichetele (boxurile sunt doar sugestii).
3. Creează split-ul train/val/test:
  ```
  python -m src.neural_network.prepare_yolo_dataset
  ```
4. Validează datasetul:
  ```
  python -m src.neural_network.validate_yolo_dataset
  ```
5. Rulează antrenarea detectorului:
  ```
  python -m src.neural_network.train_detector --data config/fruitveg_detect.yaml
  ```

## Verificare rapidă
- Numărul de fișiere din `images/*` trebuie să corespundă cu cele din `labels/*`.
- Un fișier `.txt` poate avea mai multe linii (mai multe obiecte per imagine).
