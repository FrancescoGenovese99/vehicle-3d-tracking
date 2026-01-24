#!/usr/bin/env python3
"""
Unified launcher for Vehicle 3D Localization.
Uses CLI menu to select video and method, then runs processing.

This is a lightweight wrapper around vehicle_localization_system.py
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Import menu CLI
from ui.interactive_menu_cli import select_video_and_method_cli

# Import processing functions from vehicle_localization_system
from scripts.vehicle_localization_system import (
    process_video_task1,
    process_video_task2,
    process_video_pnp,
    load_all_configs
)

from calibration.load_calibration import load_camera_calibration_simple as load_camera_calibration


def main():
    """Main launcher with CLI menu."""
    
    print("\n" + "=" * 70)
    print("🚀 VEHICLE 3D LOCALIZATION - LAUNCHER")
    print("=" * 70)
    
    # Show menu and get selection
    video_path, method = select_video_and_method_cli()
    
    if not video_path or not method:
        print("\n✖ Operation cancelled by user.")
        return
    
    print("\n" + "=" * 70)
    print("🚀 STARTING PROCESSING")
    print("=" * 70)
    print(f"📹 Video: {video_path}")
    print(f"🔧 Method: {method}")
    
    # Verify video exists
    if not Path(video_path).exists():
        print(f"\n❌ Error: Video not found: {video_path}")
        return
    
    # Load configurations
    print("\n📂 Loading configurations...")
    try:
        config = load_all_configs()
        
        calib_file = config['camera_config']['camera']['calibration_file']
        camera_matrix, dist_coeffs = load_camera_calibration(calib_file)
        
        print(f"✅ Configurations loaded")
        print(f"   Camera matrix: {camera_matrix.shape}")
        print(f"   Calibration file: {calib_file}")
        
    except Exception as e:
        print(f"\n❌ Error loading configuration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Route to appropriate method
    print(f"\n▶ Starting processing with method: {method}")
    print("=" * 70)
    
    try:
        if method == "homography":
            success = process_video_task1(video_path, camera_matrix, dist_coeffs, config)
        elif method == "vanishing_point":
            success = process_video_task2(video_path, camera_matrix, dist_coeffs, config)
        elif method == "pnp":
            success = process_video_pnp(video_path, camera_matrix, dist_coeffs, config)
        else:
            print(f"\n❌ Unknown method: {method}")
            return
        
        if success:
            print("\n" + "=" * 70)
            print("🎉 PROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print("\n📁 Output saved in: data/videos/output/")
            print("📊 Data saved in: data/results/")
            print()
        else:
            print("\n❌ Error during processing.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()