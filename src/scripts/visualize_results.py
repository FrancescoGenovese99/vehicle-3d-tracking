#!/usr/bin/env python3
"""
Script per visualizzare e analizzare i risultati salvati.

Usage:
    python scripts/visualize_results.py --video video1 --plot-trajectory
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

# Aggiungi la directory root al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_io import load_tracked_points, load_pose


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualizza e analizza risultati salvati',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Nome del video (senza estensione) di cui visualizzare i risultati'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='data/results',
        help='Directory contenente i risultati'
    )
    
    parser.add_argument(
        '--plot-trajectory',
        action='store_true',
        help='Plotta la traiettoria 2D dei punti tracciati'
    )
    
    parser.add_argument(
        '--plot-distance',
        action='store_true',
        help='Plotta la distanza del veicolo nel tempo'
    )
    
    parser.add_argument(
        '--plot-orientation',
        action='store_true',
        help='Plotta gli angoli di orientamento (yaw, pitch, roll)'
    )
    
    parser.add_argument(
        '--plot-position-3d',
        action='store_true',
        help='Plotta la posizione 3D del veicolo'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Genera tutti i plot disponibili'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Salva i plot come immagini invece di mostrarli'
    )
    
    return parser.parse_args()


def load_video_results(video_name: str, results_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Carica tutti i risultati per un video.
    
    Args:
        video_name: Nome del video
        results_dir: Directory risultati
        
    Returns:
        Tuple (tracked_points, rotations, translations)
    """
    results_path = Path(results_dir)
    
    # Carica punti tracciati
    points_file = results_path / 'tracked_points' / f'{video_name}_points.npz'
    tracked_points = None
    if points_file.exists():
        tracked_points, _ = load_tracked_points(str(points_file))
        print(f"✓ Caricati punti tracciati: {tracked_points.shape}")
    else:
        print(f"⚠️  File punti non trovato: {points_file}")
    
    # Carica pose
    poses_file = results_path / 'poses' / f'{video_name}_poses.npz'
    rotations = None
    translations = None
    if poses_file.exists():
        data = np.load(poses_file)
        rotations = data['rotations']
        translations = data['translations']
        print(f"✓ Caricate pose: {len(rotations)} frame")
    else:
        print(f"⚠️  File pose non trovato: {poses_file}")
    
    return tracked_points, rotations, translations


def plot_trajectory_2d(tracked_points: np.ndarray, video_name: str, save: bool = False):
    """
    Plotta la traiettoria 2D dei punti tracciati.
    
    Args:
        tracked_points: Array (n_frames, 2, 2) con i punti
        video_name: Nome del video
        save: Se True, salva invece di mostrare
    """
    if tracked_points is None or len(tracked_points) == 0:
        print("⚠️  Nessun punto da plottare")
        return
    
    # Calcola centro tra i due fari
    centers = tracked_points.mean(axis=1)
    
    plt.figure(figsize=(12, 8))
    
    # Plot centro
    plt.plot(centers[:, 0], centers[:, 1], 'b-', linewidth=2, label='Centro veicolo')
    plt.plot(centers[0, 0], centers[0, 1], 'go', markersize=10, label='Inizio')
    plt.plot(centers[-1, 0], centers[-1, 1], 'ro', markersize=10, label='Fine')
    
    # Plot fari individuali
    plt.plot(tracked_points[:, 0, 0], tracked_points[:, 0, 1], 
             'c--', alpha=0.5, label='Faro sinistro')
    plt.plot(tracked_points[:, 1, 0], tracked_points[:, 1, 1], 
             'm--', alpha=0.5, label='Faro destro')
    
    plt.xlabel('X (pixel)')
    plt.ylabel('Y (pixel)')
    plt.title(f'Traiettoria 2D - {video_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # Inverti Y per coordinate immagine
    
    if save:
        output_path = f"data/results/{video_name}_trajectory_2d.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot salvato: {output_path}")
    else:
        plt.show()


def plot_distance_over_time(translations: np.ndarray, video_name: str, fps: float = 30.0, save: bool = False):
    """
    Plotta la distanza del veicolo nel tempo.
    
    Args:
        translations: Array (n_frames, 3, 1) con i vettori di traslazione
        video_name: Nome del video
        fps: Frame rate per convertire frame in tempo
        save: Se True, salva invece di mostrare
    """
    if translations is None or len(translations) == 0:
        print("⚠️  Nessuna traslazione da plottare")
        return
    
    # Calcola distanza euclidea
    distances = np.linalg.norm(translations.reshape(-1, 3), axis=1)
    
    # Tempo in secondi
    time = np.arange(len(distances)) / fps
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, distances, 'b-', linewidth=2)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Distanza (m)')
    plt.title(f'Distanza dal veicolo nel tempo - {video_name}')
    plt.grid(True, alpha=0.3)
    
    # Aggiungi statistiche
    mean_dist = distances.mean()
    plt.axhline(mean_dist, color='r', linestyle='--', 
                label=f'Media: {mean_dist:.2f}m')
    plt.legend()
    
    if save:
        output_path = f"data/results/{video_name}_distance.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot salvato: {output_path}")
    else:
        plt.show()


def plot_orientation(rotations: np.ndarray, video_name: str, fps: float = 30.0, save: bool = False):
    """
    Plotta gli angoli di orientamento (yaw, pitch, roll).
    
    Args:
        rotations: Array (n_frames, 3, 1) con i vettori di rotazione (Rodrigues)
        video_name: Nome del video
        fps: Frame rate
        save: Se True, salva invece di mostrare
    """
    if rotations is None or len(rotations) == 0:
        print("⚠️  Nessuna rotazione da plottare")
        return
    
    import cv2
    
    # Converti Rodrigues in angoli di Eulero
    yaws = []
    pitches = []
    rolls = []
    
    for rvec in rotations:
        R, _ = cv2.Rodrigues(rvec)
        
        # Estrai angoli di Eulero
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        rolls.append(np.degrees(roll))
        pitches.append(np.degrees(pitch))
        yaws.append(np.degrees(yaw))
    
    time = np.arange(len(rotations)) / fps
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    axes[0].plot(time, yaws, 'b-', linewidth=2)
    axes[0].set_ylabel('Yaw (°)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Orientamento veicolo - {video_name}')
    
    axes[1].plot(time, pitches, 'g-', linewidth=2)
    axes[1].set_ylabel('Pitch (°)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time, rolls, 'r-', linewidth=2)
    axes[2].set_ylabel('Roll (°)')
    axes[2].set_xlabel('Tempo (s)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        output_path = f"data/results/{video_name}_orientation.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot salvato: {output_path}")
    else:
        plt.show()


def plot_position_3d(translations: np.ndarray, video_name: str, save: bool = False):
    """
    Plotta la posizione 3D del veicolo.
    
    Args:
        translations: Array (n_frames, 3, 1) con i vettori di traslazione
        video_name: Nome del video
        save: Se True, salva invece di mostrare
    """
    if translations is None or len(translations) == 0:
        print("⚠️  Nessuna traslazione da plottare")
        return
    
    positions = translations.reshape(-1, 3)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot traiettoria
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', linewidth=2, label='Traiettoria')
    
    # Punto iniziale e finale
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
              c='g', marker='o', s=100, label='Inizio')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
              c='r', marker='o', s=100, label='Fine')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Posizione 3D del veicolo - {video_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save:
        output_path = f"data/results/{video_name}_position_3d.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot salvato: {output_path}")
    else:
        plt.show()


def print_statistics(tracked_points: np.ndarray, rotations: np.ndarray, translations: np.ndarray):
    """
    Stampa statistiche sui risultati.
    
    Args:
        tracked_points: Punti tracciati
        rotations: Rotazioni
        translations: Traslazioni
    """
    print("\n" + "="*60)
    print("STATISTICHE")
    print("="*60)
    
    if tracked_points is not None:
        print(f"\nPunti tracciati:")
        print(f"  Frames totali: {len(tracked_points)}")
        
        # Calcola distanza tra fari
        distances = np.linalg.norm(tracked_points[:, 0] - tracked_points[:, 1], axis=1)
        print(f"  Distanza fari (pixel):")
        print(f"    Media: {distances.mean():.1f}")
        print(f"    Min: {distances.min():.1f}")
        print(f"    Max: {distances.max():.1f}")
    
    if translations is not None:
        positions = translations.reshape(-1, 3)
        distances_3d = np.linalg.norm(positions, axis=1)
        
        print(f"\nDistanza veicolo:")
        print(f"  Media: {distances_3d.mean():.2f}m")
        print(f"  Min: {distances_3d.min():.2f}m")
        print(f"  Max: {distances_3d.max():.2f}m")
        
        print(f"\nPosizione (media):")
        print(f"  X: {positions[:, 0].mean():.2f}m")
        print(f"  Y: {positions[:, 1].mean():.2f}m")
        print(f"  Z: {positions[:, 2].mean():.2f}m")
    
    print("="*60)


def main():
    """Main function."""
    args = parse_args()
    
    print("="*60)
    print("VISUALIZZAZIONE RISULTATI")
    print("="*60)
    print(f"Video: {args.video}")
    print(f"Results dir: {args.results_dir}")
    print("="*60)
    
    # Carica risultati
    tracked_points, rotations, translations = load_video_results(
        args.video, 
        args.results_dir
    )
    
    if tracked_points is None and rotations is None and translations is None:
        print("\n❌ Nessun risultato trovato per questo video")
        return 1
    
    # Stampa statistiche
    print_statistics(tracked_points, rotations, translations)
    
    # Genera plot richiesti
    if args.all or args.plot_trajectory:
        if tracked_points is not None:
            plot_trajectory_2d(tracked_points, args.video, args.save_plots)
    
    if args.all or args.plot_distance:
        if translations is not None:
            plot_distance_over_time(translations, args.video, save=args.save_plots)
    
    if args.all or args.plot_orientation:
        if rotations is not None:
            plot_orientation(rotations, args.video, save=args.save_plots)
    
    if args.all or args.plot_position_3d:
        if translations is not None:
            plot_position_3d(translations, args.video, save=args.save_plots)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())