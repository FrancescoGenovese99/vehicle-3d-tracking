#!/usr/bin/env python3
"""
Unified launcher for Vehicle 3D Localization.
Uses CLI menu to select video and method, then runs processing.
"""

import sys
from pathlib import Path
sys.path.insert(0, '/app/src')

# Import menu CLI
from ui.interactive_menu_cli import select_video_and_method_cli

# Import processing functions
from scripts.vehicle_localization_system import (
    process_video_task1,
    process_video_task2,
    process_video_pnp
)
from utils.config_loader import load_config
from calibration.load_calibration import load_camera_calibration_simple as load_camera_calibration


def main():
    """Main launcher with CLI menu."""
    
    # Show menu and get selection
    video_path, method = select_video_and_method_cli()
    
    if not video_path or not method:
        print("\n✖ Operazione annullata dall'utente.")
        return
    
    print("\n" + "="*60)
    print("🚀 AVVIO PROCESSING")
    print("="*60)
    print(f"📹 Video: {video_path}")
    print(f"🔧 Metodo: {method}")
    
    # Verify video exists
    if not Path(video_path).exists():
        print(f"\n❌ Errore: Video non trovato!")
        return
    
    # Load configurations
    print("\n📂 Caricamento configurazioni...")
    try:
        config = {}
        config['camera_config'] = load_config('config/camera_config.yaml')
        config['detection_params'] = load_config('config/detection_params.yaml')
        config['vehicle_model'] = load_config('config/vehicle_model.yaml')
        config['tracking'] = config['detection_params'].get('tracking', {})
        
        calib_file = config['camera_config']['camera']['calibration_file']
        camera_matrix, dist_coeffs = load_camera_calibration(calib_file)
        
        print(f"✅ Configurazioni caricate")
        print(f"   Camera matrix: {camera_matrix.shape}")
        
    except Exception as e:
        print(f"\n❌ Errore caricamento config: {e}")
        return
    
    # Route to appropriate method
    print(f"\n▶ Inizio processing con metodo: {method}")
    
    try:
        if method == "homography":
            success = process_video_task1(video_path, camera_matrix, dist_coeffs, config)
        elif method == "vanishing_point":
            success = process_video_task2(video_path, camera_matrix, dist_coeffs, config)
        elif method == "pnp":
            success = process_video_pnp(video_path, camera_matrix, dist_coeffs, config)
        else:
            print(f"\n❌ Metodo sconosciuto: {method}")
            return
        
        if success:
            print("\n" + "="*60)
            print("🎉 PROCESSING COMPLETATO CON SUCCESSO!")
            print("="*60)
            print("\n📁 Output salvato in: data/videos/output/")
            print("📊 Dati salvati in: data/results/")
            print()
        else:
            print("\n❌ Errore durante il processing.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
