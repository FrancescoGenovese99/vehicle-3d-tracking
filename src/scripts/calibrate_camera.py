#!/usr/bin/env python3
"""
Script per calibrare la camera usando scacchiere in legno.
Ottimizzato per rilevare pattern con basso contrasto.

Usage:
    python calibrate_camera.py -s camera1 folder_images
    python calibrate_camera.py -p folder_images
"""

import numpy as np
import cv2 as cv
import sys
from pathlib import Path


def enhance_image_for_chessboard(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    gray = cv.GaussianBlur(gray, (7, 7), 0)
    return gray


def main():
    """Funzione principale."""

    # ========== PARSING ARGOMENTI ==========
    if len(sys.argv) < 3:
        print('\n❌ ERRORE: Devi specificare:')
        print('   -s <nome_output> <folder_immagini>  (per salvare)')
        print('   -p <folder_immagini>                (per stampare)')
        print('\nEsempio: python calibrate_camera.py -s camera1 my_images\n')
        sys.exit(1)

    flag = sys.argv[1]

    if flag not in ['-s', '-p']:
        print(f'\n❌ ERRORE: Flag {flag} non valido')
        print('   Usa -s per salvare o -p per stampare\n')
        sys.exit(1)

    if flag == '-s' and len(sys.argv) < 4:
        print('\n❌ ERRORE: Con -s devi specificare il nome del file di output\n')
        sys.exit(1)

    folder_name = sys.argv[3] if flag == '-s' else sys.argv[2]
    output_name = sys.argv[2] if flag == '-s' else None

    # ========== SETUP PATHS ==========
    project_root = Path('/app')  # radice del progetto

    images_path = project_root / 'data' / 'calibration' / 'images' / folder_name

    if not images_path.exists():
        print(f'\n❌ ERRORE: La cartella {images_path} non esiste!\n')
        sys.exit(1)

    # ========== CARICA IMMAGINI ==========
    image_files = list(images_path.glob('*.jpg')) + \
                  list(images_path.glob('*.jpeg')) + \
                  list(images_path.glob('*.png'))

    if not image_files:
        print(f'\n❌ ERRORE: Nessuna immagine trovata in {images_path}\n')
        sys.exit(1)

    print(f'\n{"=" * 60}')
    print(f'CALIBRAZIONE CAMERA - Scacchiera in Legno')
    print(f'{"=" * 60}')
    print(f'Cartella immagini: {images_path}')
    print(f'Immagini trovate: {len(image_files)}')
    print(f'{"=" * 60}\n')

    # ========== PARAMETRI CALIBRAZIONE ==========

    # Dimensione corner interni della scacchiera
    # (7 colonne di corner, 3 righe di corner)
    chessboard_size = (7, 3)

    # Prepara i punti 3D della scacchiera
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Array per memorizzare i punti
    objpoints = []
    imgpoints = []

    used_images = 0

    # ========== PROCESSING IMMAGINI ==========
    print('Processamento immagini in corso...\n')

    for img_file in sorted(image_files):
        img = cv.imread(str(img_file))

        if img is None:
            print(f'⚠️  Impossibile leggere: {img_file.name}')
            continue

        gray = enhance_image_for_chessboard(img)

        # Prova a trovare i corner con il metodo classico
        ret, corners = cv.findChessboardCorners(
            gray,
            chessboard_size,
            cv.CALIB_CB_ADAPTIVE_THRESH |
            cv.CALIB_CB_NORMALIZE_IMAGE |
            cv.CALIB_CB_FILTER_QUADS
        )

        if ret:
            # Raffina i corner a subpixel
            corners = cv.cornerSubPix(
                gray, corners,
                (11, 11), (-1, -1),
                (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            used_images += 1
            print(f'✓ {img_file.name}')

            objpoints.append(objp.copy())
            imgpoints.append(corners)

            # Visualizza i corner rilevati
            img_display = img.copy()
            cv.drawChessboardCorners(img_display, chessboard_size, corners, ret)

            cv.namedWindow('Corner Rilevati', cv.WINDOW_NORMAL)
            cv.resizeWindow('Corner Rilevati', 1000, 750)
            cv.imshow('Corner Rilevati', img_display)

            if cv.waitKey(500) & 0xFF == ord('q'):
                break
        else:
            print(f'✗ {img_file.name} - Pattern non trovato')

   # cv.destroyAllWindows()

    # ========== VERIFICA IMMAGINI ==========
    print(f'\n{"=" * 60}')
    print(f'Immagini utilizzabili: {used_images}/{len(image_files)}')
    print(f'{"=" * 60}\n')

    if used_images < 3:
        print('❌ ERRORE: Servono almeno 3 immagini valide per calibrare!\n')
        sys.exit(1)

    # ========== CALIBRAZIONE ==========
    print('Esecuzione calibrazione...\n')

    img_shape = cv.imread(str(image_files[0])).shape[:2][::-1]

    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    print(f'RMS error: {ret:.4f}')

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)

    print(f'{"=" * 60}')
    print('✓ CALIBRAZIONE COMPLETATA!')
    print(f'{"=" * 60}\n')
    print('Matrice intrinseca K:')
    print(K)
    print(f'\nErrore di riproiezione medio: {mean_error:.4f} pixel')

    if flag == '-s':
        output_dir = project_root / 'data' / 'calibration' / 'matrices'
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f'{output_name}.npy'
        np.save(output_file, K.astype(np.float32))

        print(f'\n✓ Matrice salvata in: {output_file}')

    print(f'\n{"=" * 60}\n')


if __name__ == '__main__':
    main()
