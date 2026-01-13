"""
Interactive CLI menu for video and method selection (Docker-compatible).
NO tkinter - pure terminal interface.
"""

import os
from pathlib import Path
from typing import Optional, Tuple


class VideoMethodSelectorCLI:
    """
    Interactive CLI for selecting:
    1. Input video from data/videos/input/
    2. Localization method (Task 1, Task 2, PnP)
    
    Returns selected video path and method.
    """
    
    def __init__(self, videos_dir: str = "data/videos/input"):
        self.videos_dir = Path(videos_dir)
        self.selected_video: Optional[str] = None
        self.selected_method: Optional[str] = None
        
        # Method mapping
        self.methods = {
            "1": ("Task 1: Omografia (4 punti targa)", "homography"),
            "2": ("Task 2: Punto di Fuga (luci notturne)", "vanishing_point"),
            "3": ("PnP: Metodo Diretto (confronto)", "pnp")
        }
        
    def get_available_videos(self):
        """Scan videos directory for .mp4 files."""
        if not self.videos_dir.exists():
            return []
        
        videos = list(self.videos_dir.glob("*.mp4"))
        return sorted([v.name for v in videos])
    
    def print_header(self):
        """Print fancy header."""
        print("\n" + "="*60)
        print("🚗  VEHICLE 3D LOCALIZATION SYSTEM")
        print("="*60 + "\n")
    
    def select_video(self):
        """Interactive video selection."""
        videos = self.get_available_videos()
        
        if not videos:
            print("❌ Nessun video trovato in data/videos/input/")
            print("   Aggiungi file .mp4 nella cartella e riprova.\n")
            return False
        
        print("📹 VIDEO DISPONIBILI:\n")
        for i, video in enumerate(videos, 1):
            print(f"  [{i}] {video}")
        
        print(f"\n  [0] Annulla\n")
        
        while True:
            try:
                choice = input("Seleziona video (numero): ").strip()
                
                if choice == "0":
                    return False
                
                idx = int(choice) - 1
                if 0 <= idx < len(videos):
                    self.selected_video = videos[idx]
                    print(f"\n✓ Video selezionato: {self.selected_video}\n")
                    return True
                else:
                    print(f"⚠️  Numero non valido. Scegli tra 1 e {len(videos)}")
            except ValueError:
                print("⚠️  Inserisci un numero valido")
            except KeyboardInterrupt:
                print("\n\n❌ Operazione annullata\n")
                return False
    
    def select_method(self):
        """Interactive method selection."""
        print("🔬 METODI DI LOCALIZZAZIONE:\n")
        
        for key, (name, _) in self.methods.items():
            print(f"  [{key}] {name}")
        
        print("\n  Descrizioni:")
        print("    • Task 1: Ambiente diurno, targa visibile (omografia 4 punti)")
        print("    • Task 2: Ambiente notturno, solo luci posteriori (punto di fuga)")
        print("    • PnP: Metodo alternativo per confronto (solvePnP diretto)")
        print("\n  [0] Annulla\n")
        
        while True:
            try:
                choice = input("Seleziona metodo (numero): ").strip()
                
                if choice == "0":
                    return False
                
                if choice in self.methods:
                    method_name, method_code = self.methods[choice]
                    self.selected_method = method_code
                    print(f"\n✓ Metodo selezionato: {method_name}\n")
                    return True
                else:
                    print("⚠️  Numero non valido. Scegli 1, 2 o 3")
            except KeyboardInterrupt:
                print("\n\n❌ Operazione annullata\n")
                return False
    
    def run(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Run the interactive CLI menu.
        
        Returns:
            (video_path, method) tuple or (None, None) if cancelled
        """
        self.print_header()
        
        # Step 1: Select video
        if not self.select_video():
            return None, None
        
        # Step 2: Select method
        if not self.select_method():
            return None, None
        
        # Build full path
        video_path = str(self.videos_dir / self.selected_video)
        
        return video_path, self.selected_method


def select_video_and_method_cli() -> Tuple[Optional[str], Optional[str]]:
    """
    Convenience function to run the CLI selector.
    
    Returns:
        (video_path, method) tuple
        - video_path: full path to selected video or None
        - method: 'homography' | 'vanishing_point' | 'pnp' or None
    """
    selector = VideoMethodSelectorCLI()
    return selector.run()


if __name__ == "__main__":
    try:
        video, method = select_video_and_method_cli()
        
        if video and method:
            print(f"\n✓ Video: {video}")
            print(f"✓ Metodo: {method}")
            print("\nPer eseguire il processing, usa:")
            print(f"  python run_task*.py\n")
        else:
            print("\n👋 Arrivederci!\n")
    except KeyboardInterrupt:
        print("\n\n👋 Arrivederci!\n")
