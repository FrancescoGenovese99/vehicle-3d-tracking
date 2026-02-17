"""
Confronto quantitativo PnP vs VP
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def load_comparison_data(results_dir):
    """Carica dati dai .npz files"""
    files = sorted(glob.glob(str(results_dir / "frame_*.npz")))
    
    data = {
        'frames': [],
        'dist_pnp': [],
        'dist_vp': [],
        'reproj_error': []
    }
    
    for f in files:
        try:
            npz = np.load(f, allow_pickle=True)
            frame = int(Path(f).stem.split('_')[1])
            
            if 'distance_pnp' in npz and 'distance_vp' in npz:
                data['frames'].append(frame)
                data['dist_pnp'].append(float(npz['distance_pnp']))
                data['dist_vp'].append(float(npz['distance_vp']))
                data['reproj_error'].append(float(npz.get('reproj_error', 0)))
        except:
            continue
    
    return data

def plot_comparison(data, output_path):
    """Genera grafici di confronto"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Grafico 1: distanze
    ax1 = axes[0]
    ax1.plot(data['frames'], data['dist_pnp'], 'b-', label='PnP', linewidth=2)
    ax1.plot(data['frames'], data['dist_vp'], 'r--', label='VP (geometric)', linewidth=2)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Distance (m)')
    ax1.set_title('Distance Estimation: PnP vs VP')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Grafico 2: differenza assoluta
    ax2 = axes[1]
    diff = np.abs(np.array(data['dist_pnp']) - np.array(data['dist_vp']))
    ax2.plot(data['frames'], diff, 'g-', linewidth=2)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Absolute difference (m)')
    ax2.set_title('|Distance_PnP - Distance_VP|')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Grafico salvato: {output_path}")
    
    # Statistiche
    print(f"\n=== STATISTICHE ===")
    print(f"Mean PnP distance: {np.mean(data['dist_pnp']):.2f} m")
    print(f"Mean VP distance:  {np.mean(data['dist_vp']):.2f} m")
    print(f"Mean difference:   {np.mean(diff):.2f} m")
    print(f"Std PnP: {np.std(data['dist_pnp']):.2f} m")
    print(f"Std VP:  {np.std(data['dist_vp']):.2f} m")
    print(f"Mean reproj error: {np.mean(data['reproj_error']):.2f} px")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "data" / "results" / "vanishing_point"
    output_path = project_root / "data" / "videos" / "output" / "comparison_pnp_vs_vp.png"
    
    data = load_comparison_data(results_dir)
    if len(data['frames']) < 10:
        print("Dati insufficienti. Processa il video prima.")
    else:
        plot_comparison(data, output_path)