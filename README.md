# Vehicle 3D Tracking from Tail Lights

Sistema di computer vision per il tracking 3D di veicoli notturni tramite rilevamento dei fari posteriori e stima della posa con algoritmo PnP.

## 📋 Caratteristiche

- ✅ Rilevamento automatico fari posteriori rossi
- ✅ Tracking multi-frame robusto
- ✅ Stima posa 3D (rotazione + traslazione)
- ✅ Proiezione bounding box 3D orientata
- ✅ Pipeline completamente containerizzata con Docker

## 🚀 Quick Start

### 1. Preparazione Dati

Organizza i tuoi file come segue:

```
data/
├── videos/input/          # Inserisci qui i tuoi video .mp4
├── calibration/
│   ├── images/            # Immagini scacchiera per calibrazione
│   └── camera1.npy        # Parametri intrinseci camera
```

### 2. Build Docker Image

```bash
docker-compose build
```

### 3. Avvia il Container

```bash
docker-compose up -d vehicle-tracker
docker-compose exec vehicle-tracker bash
```

### 4. Processa un Video

```bash
# Singolo video
python scripts/process_video.py --input data/videos/input/video1.mp4 --output data/videos/output/video1_tracked.mp4

# Batch processing
python scripts/batch_process.py
```

## 📊 Jupyter Notebooks

Per sperimentazione e tuning parametri:

```bash
docker-compose up jupyter
```

Apri browser su: `http://localhost:8888`

Notebooks disponibili:
- `01_test_detection.ipynb` - Test rilevamento fari
- `02_tune_parameters.ipynb` - Tuning parametri HSV
- `03_analyze_results.ipynb` - Analisi risultati

## ⚙️ Configurazione

Modifica i file YAML in `config/`:

### `vehicle_model.yaml`
Definisci dimensioni veicolo e posizione fari nel sistema di riferimento del veicolo

### `detection_params.yaml`
Parametri di rilevamento (HSV, threshold, ecc.)

### `camera_config.yaml`
Path al file di calibrazione camera

## 📁 Output

Risultati salvati in `data/results/`:

- **tracked_points/**: Coordinate 2D dei fari per ogni frame (NumPy arrays)
- **poses/**: Matrici di rotazione e vettori di traslazione (formato .npz)
- **bbox_3d/**: Vertici della bounding box 3D proiettata

## 🔧 Calibrazione Camera

Se devi ricalcolare i parametri intrinseci:

```bash
python scripts/calibrate_camera.py \
  --images data/calibration/images/*.jpg \
  --pattern-size 9 6 \
  --square-size 0.025 \
  --output data/calibration/camera1.npy
```

## 📝 Struttura Progetto

```
vehicle-3d-tracking/
├── src/                   # Codice sorgente modulare
│   ├── detection/         # Rilevamento fari
│   ├── tracking/          # Tracking temporale
│   ├── pose_estimation/   # PnP solver
│   └── visualization/     # Rendering risultati
├── scripts/               # Entry points
├── config/                # File di configurazione
├── data/                  # Dati e risultati
└── notebooks/             # Jupyter notebooks
```

## 🧪 Testing

```bash
pytest tests/
```

## 📖 Algoritmo

1. **Rilevamento**: Filtro HSV per luci rosse/bianche + blob detection
2. **Selezione**: Euristica geometrica per identificare coppia fari posteriori
3. **Tracking**: CSRT tracker OpenCV con re-detection automatica
4. **Posa 3D**: cv2.solvePnP() con corrispondenze 2D-3D note
5. **Proiezione**: Rendering bounding box 3D orientata

## 📄 Licenza

MIT License