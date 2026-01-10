#!/usr/bin/env python3
"""
Script per calibrare la camera usando immagini di scacchiera.

Usage:
    python scripts/calibrate_camera.py --images "data/calibration/images/*.jpg" \
                                       --pattern-size 9 6 \
                                       --square-size 0.025 \
                                       --output data/calibration/camera1.npy \
                                       --visualize
"""

import argparse
import sys
from pathlib import Path

# Aggiungi la directory root al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration.camera_calibrator import CameraCalibrator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calibra camera usando immagini di scacchiera',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Pattern glob per le immagini di calibrazione (es: "data/calibration/images/*.jpg")'
    )
    
    parser.add_argument(
        '--pattern-size',
        type=int,
        nargs=2,
        required=True,
        metavar=('COLS', 'ROWS'),
        help='Dimensione pattern: numero di angoli interni (colonne righe)'
    )
    
    parser.add_argument(
        '--square-size',
        type=float,
        required=True,
        help='Dimensione di un quadrato della scacchiera in metri'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/calibration/camera_calibration.npz',
        help='Path del file di output (.npz)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualizza i corner rilevati su ogni immagine'
    )
    
    parser.add_argument(
        '--min-images',
        type=int,
        default=3,
        help='Numero minimo di immagini richieste per calibrare'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("="*60)
    print("CALIBRAZIONE CAMERA")
    print("="*60)
    print(f"Pattern immagini: {args.images}")
    print(f"Pattern size: {args.pattern_size[0]}x{args.pattern_size[1]}")
    print(f"Square size: {args.square_size}m")
    print(f"Output: {args.output}")
    print(f"Visualizza: {args.visualize}")
    print("="*60)
    print()
    
    try:
        # Esegui calibrazione
        camera_matrix, dist_coeffs, mean_error = CameraCalibrator.calibrate_from_images(
            image_pattern=args.images,
            pattern_size=tuple(args.pattern_size),
            square_size=args.square_size,
            output_path=args.output,
            visualize=args.visualize
        )
        
        print("\n" + "="*60)
        print("CALIBRAZIONE COMPLETATA CON SUCCESSO!")
        print("="*60)
        print("\nParametri Camera:")
        print(f"  fx: {camera_matrix[0, 0]:.2f}")
        print(f"  fy: {camera_matrix[1, 1]:.2f}")
        print(f"  cx: {camera_matrix[0, 2]:.2f}")
        print(f"  cy: {camera_matrix[1, 2]:.2f}")
        print(f"\nCoefficienti di distorsione:")
        print(f"  k1: {dist_coeffs[0]:.6f}")
        print(f"  k2: {dist_coeffs[1]:.6f}")
        print(f"  p1: {dist_coeffs[2]:.6f}")
        print(f"  p2: {dist_coeffs[3]:.6f}")
        print(f"  k3: {dist_coeffs[4]:.6f}")
        print(f"\nErrore di riproiezione medio: {mean_error:.4f} pixel")
        print(f"\nFile salvato in: {args.output}")
        print("="*60)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERRORE: {e}")
        return 1
    except ValueError as e:
        print(f"\n❌ ERRORE: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERRORE IMPREVISTO: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())