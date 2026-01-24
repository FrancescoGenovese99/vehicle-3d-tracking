"""
Data I/O - Funzioni per salvare e caricare risultati.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json


def save_tracked_points(points: np.ndarray, output_path: str, metadata: Optional[dict] = None):
    """
    Salva i punti tracciati nel tempo.
    
    Args:
        points: Array numpy shape (n_frames, n_points, 2) con coordinate (x, y)
        output_path: Path del file di output (.npz)
        metadata: Metadati opzionali da salvare
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {'points': points}
    if metadata:
        save_dict['metadata'] = np.array([json.dumps(metadata)])
    
    np.savez_compressed(output_file, **save_dict)
    print(f"Punti salvati in: {output_path}")


def load_tracked_points(input_path: str) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Carica i punti tracciati da file.
    
    Args:
        input_path: Path del file .npz
        
    Returns:
        Tuple (points, metadata) dove:
        - points: Array numpy con i punti
        - metadata: Dizionario con metadati (se presenti)
    """
    data = np.load(input_path, allow_pickle=True)
    points = data['points']
    
    metadata = None
    if 'metadata' in data:
        metadata = json.loads(str(data['metadata'][0]))
    
    return points, metadata


def save_pose(rotation: np.ndarray, translation: np.ndarray, output_path: str, 
              frame_idx: Optional[int] = None):
    """
    Salva la posa 3D (rotazione e traslazione).
    
    Args:
        rotation: Matrice di rotazione 3x3 o vettore di rotazione (Rodrigues)
        translation: Vettore di traslazione 3x1
        output_path: Path del file di output (.npz)
        frame_idx: Indice del frame (opzionale)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'rotation': rotation,
        'translation': translation
    }
    
    if frame_idx is not None:
        save_dict['frame_idx'] = frame_idx
    
    np.savez_compressed(output_file, **save_dict)


def load_pose(input_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """
    Carica una posa 3D da file.
    
    Args:
        input_path: Path del file .npz
        
    Returns:
        Tuple (rotation, translation, frame_idx) dove:
        - rotation: Matrice/vettore di rotazione
        - translation: Vettore di traslazione
        - frame_idx: Indice frame (se presente)
    """
    data = np.load(input_path)
    rotation = data['rotation']
    translation = data['translation']
    
    frame_idx = None
    if 'frame_idx' in data:
        frame_idx = int(data['frame_idx'])
    
    return rotation, translation, frame_idx


def save_bbox_3d(vertices: np.ndarray, output_path: str):
    """
    Salva i vertici della bounding box 3D.
    
    Args:
        vertices: Array numpy shape (n_frames, 8, 2) con coordinate 2D proiettate
                  oppure (8, 3) per singolo frame con coordinate 3D
        output_path: Path del file di output (.npy)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_file, vertices)
    print(f"Bounding box 3D salvata in: {output_path}")


def load_bbox_3d(input_path: str) -> np.ndarray:
    """
    Carica i vertici della bounding box 3D.
    
    Args:
        input_path: Path del file .npy
        
    Returns:
        Array numpy con i vertici
    """
    return np.load(input_path)


def save_all_poses(poses: List[Tuple[np.ndarray, np.ndarray]], output_path: str, 
                   video_name: str):
    """
    Salva tutte le pose di un video in un unico file.
    
    Args:
        poses: Lista di tuple (rotation, translation) per ogni frame
        output_path: Path della directory di output
        video_name: Nome del video (senza estensione)
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rotations = []
    translations = []
    
    for rot, trans in poses:
        rotations.append(rot)
        translations.append(trans)
    
    rotations = np.array(rotations)
    translations = np.array(translations)
    
    output_file = output_dir / f"{video_name}_poses.npz"
    np.savez_compressed(output_file, rotations=rotations, translations=translations)
    print(f"Pose salvate in: {output_file}")


def ensure_output_dirs(base_dir: str = "data/results"):
    """
    Crea le directory di output se non esistono.
    
    Args:
        base_dir: Directory base per i risultati
    """
    base_path = Path(base_dir)
    
    subdirs = [
        'tracked_points',
        'poses',
        'bbox_3d'
    ]
    
    for subdir in subdirs:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"Directory di output create in: {base_dir}")