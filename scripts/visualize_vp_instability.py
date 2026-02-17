"""
Script to visualize the instability of the VP method compared to PnP.
Generates comparative images for the report.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def load_frame_data(results_dir):
    """Load data from .npz files"""
    files = sorted(glob.glob(str(results_dir / "frame_*.npz")))
    
    data = {
        'frames': [],
        'vx': [],
        'vy': [],
        'dist_pnp': [],
        'dist_vp': [],
        'reproj_error': []
    }
    
    for f in files:
        try:
            npz = np.load(f, allow_pickle=True)
            frame_num = int(Path(f).stem.split('_')[1])
            
            # Extract data if available
            if 'distance_pnp' in npz:
                data['frames'].append(frame_num)
                data['dist_pnp'].append(float(npz['distance_pnp']))
                
                # VP data (if available)
                if 'distance_vp' in npz:
                    data['dist_vp'].append(float(npz['distance_vp']))
                else:
                    data['dist_vp'].append(np.nan)
                
                # Vanishing points (if available)
                if 'vx' in npz and npz['vx'] is not None:
                    vx = npz['vx']
                    if isinstance(vx, np.ndarray) and len(vx) >= 2:
                        data['vx'].append([float(vx[0]), float(vx[1])])
                    else:
                        data['vx'].append(None)
                else:
                    data['vx'].append(None)
                
                # Reprojection error
                if 'reproj_error' in npz:
                    data['reproj_error'].append(float(npz['reproj_error']))
                else:
                    data['reproj_error'].append(0.0)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
    
    return data

def plot_vp_instability(data, output_dir):
    """
    Generate plots showing the instability of the VP method
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames = np.array(data['frames'])
    dist_pnp = np.array(data['dist_pnp'])
    dist_vp = np.array(data['dist_vp'])
    
    # Remove NaN values for VP
    valid_vp = ~np.isnan(dist_vp)
    
    # =========================================================================
    # Figure 1: Comparison of PnP vs VP distances
    # =========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Subplot 1: Distances
    ax1 = axes[0]
    ax1.plot(frames, dist_pnp, 'b-', linewidth=2, label='PnP (stable)', alpha=0.9)
    ax1.plot(frames[valid_vp], dist_vp[valid_vp], 'r--', linewidth=1.5, 
             label='VP (unstable)', alpha=0.7)
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('Estimated Distance (m)', fontsize=12)
    ax1.set_title('Distance Estimation: PnP vs VP Method', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(np.nanmax(dist_vp), np.max(dist_pnp)) * 1.1])
    
    # Subplot 2: Absolute difference
    ax2 = axes[1]
    diff = np.abs(dist_pnp[valid_vp] - dist_vp[valid_vp])
    ax2.plot(frames[valid_vp], diff, 'g-', linewidth=2, label='|Distance_PnP - Distance_VP|')
    ax2.axhline(y=np.mean(diff), color='orange', linestyle='--', 
                label=f'Mean difference: {np.mean(diff):.2f} m', linewidth=2)
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Absolute Difference (m)', fontsize=12)
    ax2.set_title('Distance Estimation Error: VP vs PnP', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "comparison_pnp_vs_vp_detailed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # =========================================================================
    # Figure 2: Vanishing Point Trajectory (shows jumping)
    # =========================================================================
    vx_coords = []
    vx_frames = []
    for i, vx in enumerate(data['vx']):
        if vx is not None:
            vx_coords.append(vx)
            vx_frames.append(data['frames'][i])
    
    if len(vx_coords) > 2:
        vx_array = np.array(vx_coords)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot trajectory
        scatter = ax.scatter(vx_array[:, 0], vx_array[:, 1], 
                           c=vx_frames, cmap='viridis', s=50, alpha=0.6)
        
        # Connect consecutive points to show jumping
        for i in range(len(vx_array) - 1):
            ax.plot([vx_array[i, 0], vx_array[i+1, 0]], 
                   [vx_array[i, 1], vx_array[i+1, 1]], 
                   'r-', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('Image X (pixels)', fontsize=12)
        ax.set_ylabel('Image Y (pixels)', fontsize=12)
        ax.set_title('Motion Vanishing Point Vx Trajectory\n(shows instability and jumping)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Frame', fontsize=11)
        
        # Statistics text box
        std_x = np.std(vx_array[:, 0])
        std_y = np.std(vx_array[:, 1])
        textstr = f'Std Dev X: {std_x:.1f} px\nStd Dev Y: {std_y:.1f} px'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = output_dir / "vp_trajectory_instability.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    # =========================================================================
    # Figure 3: Statistics Summary
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram comparison
    ax1 = axes[0]
    ax1.hist(dist_pnp, bins=20, alpha=0.7, color='blue', label='PnP', edgecolor='black')
    ax1.hist(dist_vp[valid_vp], bins=20, alpha=0.7, color='red', label='VP', edgecolor='black')
    ax1.set_xlabel('Estimated Distance (m)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distance Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Box plot comparison
    ax2 = axes[1]
    bp = ax2.boxplot([dist_pnp, dist_vp[valid_vp]], 
                     labels=['PnP', 'VP'],
                     patch_artist=True,
                     widths=0.6)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Estimated Distance (m)', fontsize=12)
    ax2.set_title('Distance Estimation Variability', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "statistics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # =========================================================================
    # Print Summary Statistics
    # =========================================================================
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)
    print(f"PnP Method:")
    print(f"  Mean distance:     {np.mean(dist_pnp):.2f} m")
    print(f"  Std deviation:     {np.std(dist_pnp):.2f} m")
    print(f"  Coefficient of variation: {(np.std(dist_pnp)/np.mean(dist_pnp))*100:.1f}%")
    print(f"\nVP Method:")
    print(f"  Mean distance:     {np.nanmean(dist_vp):.2f} m")
    print(f"  Std deviation:     {np.nanstd(dist_vp):.2f} m")
    print(f"  Coefficient of variation: {(np.nanstd(dist_vp)/np.nanmean(dist_vp))*100:.1f}%")
    print(f"\nDifference:")
    print(f"  Mean absolute difference: {np.mean(diff):.2f} m")
    print(f"  Relative error:           {(np.mean(diff)/np.mean(dist_pnp))*100:.1f}%")
    print(f"\nReprojection Error (PnP):")
    print(f"  Mean:  {np.mean(data['reproj_error']):.2f} px")
    print(f"  Max:   {np.max(data['reproj_error']):.2f} px")
    print("="*70 + "\n")

if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent.parent if Path(__file__).parent.name == "scripts" else Path(__file__).parent
    results_dir = project_root / "data" / "results" / "vanishing_point"
    output_dir = project_root / "data" / "videos" / "output"
    
    print("Loading frame data...")
    data = load_frame_data(results_dir)
    
    if len(data['frames']) < 10:
        print("❌ Insufficient data. Process video first with option 2.")
    else:
        print(f"✓ Loaded {len(data['frames'])} frames")
        print("\nGenerating visualization plots...")
        plot_vp_instability(data, output_dir)
        print("\n✅ All plots generated successfully!")