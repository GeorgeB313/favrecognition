# UI / Web Service

Aplicația Flask este definită în rădăcină (`app.py`) și folosește șabloanele din `templates/` și stilurile din `static/`.

## Pornire server
```bash
python app.py
```
Serverul rulează la `http://127.0.0.1:5000/`.

## Funcționalități
- Upload imagine și returnare top-3 probabilități.
- Cameră live în browser cu preview oglindit, captură și trimitere către `/api/predict`.
- Endpoint `POST /api/predict`: trimiteți fișierul în câmpul `image`; răspuns JSON cu lista de predicții.

## Cerințe
- Dependențe instalate: `pip install -r requirements.txt`
- Model și etichete în `models/fruitveg_cnn.pt` și `models/label_map.json`.

## Captură UI
O captură este disponibilă în `docs/screenshots/ui_demo.svg`.
