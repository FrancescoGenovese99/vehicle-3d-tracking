# Vehicle 3D Localization System

Sistema di **localizzazione 3D** per la stima della posa (posizione + orientamento) e dell'occupazione di spazio di un veicolo (Toyota Aygo X) osservato da una telecamera calibrata fissa.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

---

## 🎯 Scopo del Progetto

Sviluppare un sistema informatico in grado di **stimare, per ogni istante di tempo**, la posizione e l'orientamento tridimensionale (posa 3D) di un veicolo in movimento osservato da una singola telecamera fissa.

A partire da questa stima, il sistema determina lo **spazio occupato dal veicolo** sulla strada, rappresentandolo come una **bounding box 3D** (parallelepipedo) con dimensioni note, allineata alla posa stimata del veicolo.

### Obiettivo

**Stima frame-by-frame** di:
- Posa 3D del veicolo (posizione + orientamento)
- Bounding box 3D allineata con dimensioni reali
- Occupazione di spazio nel sistema di riferimento della telecamera

**Non richiesto**: Ricostruzione 3D completa della scena, SLAM, mappatura globale.

---

## 📋 Task Implementati

Il progetto implementa **tre metodi alternativi** di localizzazione, selezionabili dall'utente all'avvio:

### 🌙 **Task 1 - Localizzazione da Omografia (4 Punti Complanari)**

**Scenario**: Ambiente diurno, targa posteriore visibile

**Metodo**:
1. Rilevamento automatico dei **4 angoli della targa** posteriore
2. Calcolo **omografia H** tra piano targa 3D e piano immagine 2D
3. Decomposizione H per estrarre posa: **[r₁ r₂ t] = K⁻¹·H**
4. Ricostruzione matrice rotazione completa **R = [r₁ r₂ r₃]**
5. Proiezione bounding box 3D

**Vantaggi**:
- ✅ Funziona con singolo frame
- ✅ Robusto se targa ben visibile
- ✅ Non richiede tracking temporale

**Limitazioni**:
- ⚠️ Instabile con prospettiva debole (telecamera molto lontana)
- ⚠️ Richiede targa visibile e non occluded

---

### 🌃 **Task 2 - Localizzazione Notturna da Punto di Fuga**

**Scenario**: Ambiente notturno, solo luci posteriori visibili

**Metodo** (implementazione secondo specifiche):
1. Rilevamento **luci posteriori** frame N: L₁′, R₁′
2. Tracking luci frame N+1: L₂′, R₂′
3. Calcolo **punto di fuga Vₓ** = intersezione(L₁′L₂′, R₁′R₂′)
4. Conversione Vₓ in **direzione 3D moto** tramite K⁻¹
5. Verifica perpendicolarità: segmento_luci ⊥ direzione_moto
6. **Stima distanza piano π** usando vincolo metrico (distanza luci = 1.40m)
7. Posizionamento modello 3D veicolo
8. Proiezione bounding box 3D

**Vantaggi**:
- ✅ Funziona in condizioni notturne (scarsa illuminazione)
- ✅ Usa solo luci posteriori (sempre visibili)
- ✅ Sfrutta movimento tra frame

**Limitazioni**:
- ⚠️ Richiede almeno 2 frame consecutivi
- ⚠️ Assume moto traslatorio rettilineo

---

### 🔧 **Metodo PnP (Opzionale - Confronto)**

**Metodo alternativo/di confronto** disponibile per validazione:

1. Rilevamento 2 luci posteriori (coordinate 2D pixel)
2. Uso coordinate 3D note dal modello CAD
3. Risoluzione diretta con **cv2.solvePnP()**
4. Output: [R | t] diretto

**Vantaggi PnP**:
- ✅ Più accurato (usa geometria 3D esatta)
- ✅ Funziona anche con singolo frame
- ✅ Robusto al rumore

**Perché è opzionale**: Non segue le specifiche del Task 2 (che richiedono il metodo del punto di fuga), ma fornisce un utile termine di confronto per validare gli altri metodi.

---

## 🏗️ Architettura Sistema
```
┌─────────────────────────────────────────────────────────────┐
│              MENU INTERATTIVO ALL'AVVIO                      │
│  1. Selezione video da data/videos/input/                   │
│  2. Scelta metodo: Task 1 | Task 2 | PnP                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │   TASK 1: OMOGRAFIA          │
          │   - Detection targa (4 pt)   │
          │   - Calcolo H                │
          │   - Decomposizione [R|t]     │
          └──────────┬───────────────────┘
                     │
          ┌──────────────────────────────┐
          │   TASK 2: PUNTO DI FUGA      │
          │   - Detection luci (HSV)     │
          │   - Tracking CSRT            │
          │   - Calcolo Vₓ               │
          │   - Stima distanza π         │
          └──────────┬───────────────────┘
                     │
          ┌──────────────────────────────┐
          │   METODO PnP (opzionale)     │
          │   - Detection luci           │
          │   - solvePnP diretto         │
          └──────────┬───────────────────┘
                     │
                     ▼
          ┌──────────────────────────────┐
          │   PROIEZIONE BBOX 3D         │
          │   - 8 vertici → 2D           │
          │   - Wireframe rendering      │
          └──────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────┐
│  OUTPUT: Video Annotato + Dati Numerici (pose, bbox, ...)  │
└────────────────────────────────────────────────────────────┘
```

---

## 📊 Input Noti del Sistema

Come da specifiche, il sistema dispone di:

### 1. Matrice Calibrazione Intrinseca K

File: `data/calibration/camera1.npy`
```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

### 2. Modello Geometrico Veicolo

File: `config/vehicle_model.yaml`

**Dimensioni Toyota Aygo X**:
- Lunghezza: 3.70 m
- Larghezza: 1.74 m
- Altezza: 1.525 m

**Coordinate 3D luci posteriori** (sistema riferimento veicolo):
- Luce sinistra: [-0.27, 0.70, 0.50] m
- Luce destra: [-0.27, -0.70, 0.50] m
- Distanza tra luci: 1.40 m

**Coordinate 3D angoli targa** (per Task 1):
- Top-left: [-0.30, 0.25, 0.45] m
- Top-right: [-0.30, -0.25, 0.45] m
- Bottom-right: [-0.30, -0.25, 0.35] m
- Bottom-left: [-0.30, 0.25, 0.35] m

### 3. Video Telecamera Fissa

Formato: `.mp4`, posizionati in `data/videos/input/`

---

## 🚀 Quick Start con Docker

### Prerequisiti

- Docker >= 20.10
- Docker Compose >= 1.29

### 1. Clona Repository
```bash
git clone https://github.com/FrancescoGenovese99/vehicle-3d-tracking.git
cd vehicle-3d-tracking
```

### 2. Prepara i Dati
```
data/
├── videos/input/          # Inserisci qui i tuoi video .mp4
├── calibration/
│   ├── images/            # Immagini scacchiera per calibrazione
│   └── camera1.npy        # Parametri camera (generato da calibrazione)
```

### 3. Build e Avvio Docker
```bash
docker-compose build
docker-compose up -d vehicle-tracker
docker-compose exec vehicle-tracker bash
```

### 4. Calibra Camera (Prima Esecuzione)
```bash
python src/scripts/calibrate_camera.py \
  --images data/calibration/images/*.jpeg \
  --pattern-size 9 6 \
  --square-size 0.025 \
  --output data/calibration/camera1.npy
```

### 5. Avvia Sistema con Menu Interattivo
```bash
python scripts/vehicle_localization_system.py
```

Il sistema mostrerà un **menu pop-up** che permette di:
1. Selezionare il video da processare (lista automatica da `data/videos/input/`)
2. Scegliere il metodo di localizzazione:
   - **Task 1**: Omografia da targa (4 punti complanari)
   - **Task 2**: Punto di fuga notturno (luci posteriori)
   - **PnP**: Metodo diretto (confronto)

---

## ⚙️ Configurazione del Sistema

### 1. `vehicle_model.yaml` - Modello CAD Veicolo
```yaml
vehicle:
  make: "Toyota"
  model: "Aygo X"
  
  dimensions:
    length: 3.70
    width: 1.74
    height: 1.525
  
  # Coordinate 3D luci posteriori
  tail_lights:
    left: [-0.27, 0.70, 0.50]
    right: [-0.27, -0.70, 0.50]
    distance_between: 1.40
  
  # Coordinate 3D angoli targa (Task 1)
  license_plate:
    corners:
      top_left: [-0.30, 0.25, 0.45]
      top_right: [-0.30, -0.25, 0.45]
      bottom_right: [-0.30, -0.25, 0.35]
      bottom_left: [-0.30, 0.25, 0.35]
    width: 0.50  # metri
    height: 0.10
  
  bbox_vertices:
    auto_generate: true
```

### 2. `detection_params.yaml` - Parametri Rilevamento
```yaml
# Range HSV per luci rosse (Task 2)
hsv_ranges:
  red:
    lower1: [0, 100, 100]
    upper1: [10, 255, 255]
    lower2: [170, 100, 100]
    upper2: [180, 255, 255]

# Blob detection luci
blob_detection:
  min_area: 50
  max_area: 5000
  min_circularity: 0.4

# Selezione coppia luci
tail_lights_selection:
  min_horizontal_distance: 50
  max_horizontal_distance_ratio: 0.8
  max_vertical_offset: 50

# Tracking (Task 2)
tracking:
  tracker_type: "CSRT"
  max_frames_lost: 10

# Detection targa (Task 1)
license_plate_detection:
  edge_detection:
    canny_low: 50
    canny_high: 150
  contour_filter:
    min_area: 1000
    max_area: 50000
    aspect_ratio_min: 3.0  # Targa italiana ~520x110mm
    aspect_ratio_max: 6.0
```

### 3. `camera_config.yaml` - Calibrazione Camera
```yaml
camera:
  calibration_file: "data/calibration/camera1.npy"
  
  resolution:
    width: 1280
    height: 720

calibration:
  pattern:
    size: [9, 6]
    square_size: 0.025

# Configurazione metodi
methods:
  pnp:
    solver: "ITERATIVE"
    refine_iterations: 10
  
  homography:
    method: "RANSAC"
    ransac_threshold: 5.0
  
  vanishing_point:
    line_intersection_tolerance: 10  # pixel
```

---

## 🔬 Pipeline Algoritmica Dettagliata

### Task 1: Omografia da Targa
```
Frame N
↓
1. Detection Targa
   - Conversione grayscale
   - Edge detection (Canny)
   - Trova contorni
   - Filtra per area e aspect ratio
   - Identifica 4 angoli targa
   ↓
   Output: 4 punti 2D [(u,v)] pixel

2. Preparazione Punti 3D
   - Carica coordinate angoli targa da vehicle_model.yaml
   - Ordina: top-left, top-right, bottom-right, bottom-left
   ↓
   Output: 4 punti 3D [(X,Y,Z)] metri

3. Calcolo Omografia H
   cv2.findHomography(points_3d_plane, points_2d, RANSAC)
   ↓
   Output: H (3×3)

4. Decomposizione Omografia
   [r₁ r₂ t] = K⁻¹ · H
   r₃ = r₁ × r₂  (prodotto vettoriale)
   R = [r₁ r₂ r₃]
   ↓
   Output: R (3×3), t (3×1)

5. Proiezione Bbox 3D
   (come Task 2)
```

### Task 2: Punto di Fuga Notturno
```
Frame N
↓
1. Detection Luci
   [IDENTICO A IMPLEMENTAZIONE CORRENTE]
   ↓
   Output: L₁′(u₁,v₁), R₁′(u₁,v₁)

Frame N+1
↓
2. Tracking Luci
   CSRT tracker update
   ↓
   Output: L₂′(u₂,v₂), R₂′(u₂,v₂)

3. Calcolo Punto di Fuga Vₓ
   - Retta 1: passa per L₁′ e L₂′
   - Retta 2: passa per R₁′ e R₂′
   - Vₓ = intersezione(Retta1, Retta2)
   ↓
   Output: Vₓ(uᵥ, vᵥ) pixel

4. Direzione 3D Moto
   d_pixel = [uᵥ - cx, vᵥ - cy, f]
   d_world = K⁻¹ · d_pixel
   d_normalized = d_world / ||d_world||
   ↓
   Output: direzione moto 3D unitaria

5. Verifica Perpendicolarità
   segmento_luci = R₁′ - L₁′
   dot_product = segmento_luci · direzione_moto
   ↓
   Verifica: |dot_product| ≈ 0 (perpendicolare)

6. Stima Distanza Piano π
   - Usa vincolo: distanza_reale_luci = 1.40m
   - Calcola distanza_pixel_luci = ||R₁′ - L₁′||
   - Relazione prospettica: d = (f · distanza_reale) / distanza_pixel
   ↓
   Output: distanza piano z = d metri

7. Ricostruzione Posa
   - Centro veicolo: tvec = punto_medio_luci_3D
   - Orientamento: R calcolato da direzione_moto
   ↓
   Output: [R | t]

8. Proiezione Bbox 3D
   [IDENTICO A IMPLEMENTAZIONE CORRENTE]
```

### Metodo PnP (Opzionale)
```
[IMPLEMENTAZIONE GIÀ PRESENTE - INVARIATA]

Frame N
↓
1. Detection Luci → 2 punti 2D
2. Punti 3D dal modello CAD
3. cv2.solvePnP(points_3d, points_2d, K, dist)
4. Output: [rvec, tvec] → [R | t]
5. Proiezione Bbox 3D
```

---

## 📁 Struttura Codice
```
vehicle-3d-tracking/
│
├── config/
│   ├── camera_config.yaml
│   ├── detection_params.yaml
│   └── vehicle_model.yaml
│
├── data/
│   ├── videos/
│   │   ├── input/                   # Video da processare
│   │   └── output/                  # Video con tracking
│   ├── calibration/
│   │   ├── images/
│   │   └── camera1.npy
│   └── results/
│       ├── task1_homography/
│       ├── task2_vanishing_point/
│       └── pnp_comparison/
│
├── src/
│   ├── calibration/
│   │   ├── camera_calibration.py
│   │   └── load_calibration.py
│   │
│   ├── detection/
│   │   ├── light_detector.py        # Luci posteriori (HSV + blob)
│   │   ├── plate_detector.py        # NEW: Detection targa
│   │   └── candidate_selector.py
│   │
│   ├── tracking/
│   │   ├── tracker.py               # CSRT tracker
│   │   └── redetection.py
│   │
│   ├── pose_estimation/
│   │   ├── homography_solver.py     # NEW: Task 1
│   │   ├── vanishing_point_solver.py # NEW: Task 2 (specifiche)
│   │   ├── pnp_solver.py            # Esistente (confronto)
│   │   └── bbox_3d_projector.py
│   │
│   ├── visualization/
│   │   ├── draw_utils.py
│   │   └── video_writer.py
│   │
│   ├── ui/                          # NEW
│   │   └── interactive_menu.py      # Menu selezione video/metodo
│   │
│   └── utils/
│       ├── config_loader.py
│       └── data_io.py
│
├── scripts/
│   └── vehicle_localization_system.py  # Entry point con menu
│
└── notebooks/
    ├── 01_test_detection.ipynb
    ├── 02_tune_parameters.ipynb
    ├── 03_analyze_results.ipynb
    └── 04_compare_methods.ipynb     # NEW: Confronto Task 1/2/PnP
```

---

## 📈 Output del Sistema

### 1. Video Annotato

**Task 1 (Omografia)**:
- 4 angoli targa (cerchi gialli)
- Bounding box 3D wireframe (blu)
- Info: frame, metodo, distanza

**Task 2 (Punto di Fuga)**:
- Luci posteriori tracciate (cerchi verdi)
- Punto di fuga Vₓ (croce rossa)
- Traiettorie L₁′→L₂′, R₁′→R₂′ (linee gialle)
- Bounding box 3D wireframe (blu)

**PnP**:
- Luci posteriori (cerchi verdi)
- Bounding box 3D wireframe (viola)

### 2. Dati Numerici

Salvati in `data/results/{metodo}/`:
```python
# Posa frame N
pose_data = np.load(f'results/task2_vanishing_point/frame_{N:04d}.npz')
R = pose_data['R']        # (3,3)
t = pose_data['tvec']     # (3,1)
method = pose_data['method']  # 'vanishing_point' | 'homography' | 'pnp'
```

### 3. Analisi Comparativa

Notebook `04_compare_methods.ipynb` per confrontare:
- Traiettorie 3D dei 3 metodi sovrapposti
- Errori relativi tra metodi
- Robustezza al rumore
- Performance temporali

---

## 🧪 Testing
```bash
pytest src/tests/ -v --cov=src

# Test specifici
pytest src/tests/test_homography.py          # Task 1
pytest src/tests/test_vanishing_point.py     # Task 2
pytest src/tests/test_pnp_solver.py          # PnP
```

---

## 🔧 Assunzioni Sistema

✅ **Strada piana** (z=0 suolo)  
✅ **Veicolo simmetrico** (piano verticale)  
✅ **Movimento semplice** (traslazione rettilinea/curvatura costante)  
✅ **Geometria veicolo nota** (dimensioni, coordinate elementi)  

---

## 📝 Limitazioni

**Implementazione corrente**:
- **Task 3 non implementato**: Metodo simmetria per angolo θ (skip)
- **Singolo veicolo**: Sistema ottimizzato per 1 veicolo
- **Task 1**: Richiede targa visibile e non occluded
- **Task 2**: Richiede entrambe le luci posteriori visibili

---

## 🚀 Miglioramenti Futuri

**Algoritmi**:
- [ ] **Task 3**: Simmetria per stima θ con prospettiva debole
- [ ] **Fusione multi-metodo**: Kalman filter per combinare Task 1 + Task 2
- [ ] **Deep learning**: YOLO custom per detection targa/luci

**Sistema**:
- [ ] Multi-vehicle tracking
- [ ] Real-time processing (GPU acceleration)
- [ ] Auto-selezione metodo basata su condizioni (lighting, visibilità)

---

## 📄 Licenza

MIT License

---

## 👤 Autore

**Francesco Genovese**

- GitHub: [@FrancescoGenovese99](https://github.com/FrancescoGenovese99)
- Repository: [vehicle-3d-tracking](https://github.com/FrancescoGenovese99/vehicle-3d-tracking)

---

## 📚 Riferimenti

- Documentazione task: `task/tasks descritption.txt`
- Specifiche occupazione spazio: `task/Space occupancy.pdf`
- OpenCV solvePnP: [cv2.solvePnP()](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)
- OpenCV findHomography: [cv2.findHomography()](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780)

---

**Per domande o problemi, apri una [Issue](https://github.com/FrancescoGenovese99/vehicle-3d-tracking/issues) su GitHub.**